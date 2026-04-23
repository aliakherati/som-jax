"""Unit tests for :mod:`som_jax.oh` — OH trajectory helpers and simulate-time
support for time-varying OH.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from som_jax import (
    build_initial,
    oh_constant,
    oh_exponential_decay,
    oh_linear_ramp,
    oh_piecewise_linear,
    simulate,
)
from som_jax.mechanism import SOMNetwork
from som_jax.oh import as_oh_callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"


@pytest.fixture(scope="module")
def network() -> SOMNetwork:
    return SOMNetwork.from_json(_GENSOMG_JSON)


# --- helper-function unit tests -----------------------------------------


def test_oh_constant_ignores_time() -> None:
    f = oh_constant(1.5)
    for t in (0.0, 1.0, 1e6):
        assert float(f(jnp.asarray(t))) == pytest.approx(1.5)


def test_oh_linear_ramp_endpoints_and_midpoint() -> None:
    f = oh_linear_ramp(t0=0.0, t1=10.0, value0=1.0, value1=3.0)
    assert float(f(jnp.asarray(0.0))) == pytest.approx(1.0)
    assert float(f(jnp.asarray(10.0))) == pytest.approx(3.0)
    assert float(f(jnp.asarray(5.0))) == pytest.approx(2.0)


def test_oh_linear_ramp_clamps_outside_interval() -> None:
    f = oh_linear_ramp(t0=0.0, t1=10.0, value0=1.0, value1=3.0)
    assert float(f(jnp.asarray(-5.0))) == pytest.approx(1.0)
    assert float(f(jnp.asarray(50.0))) == pytest.approx(3.0)


def test_oh_piecewise_linear_interpolates_between_knots() -> None:
    ts = jnp.asarray([0.0, 1.0, 2.0])
    vs = jnp.asarray([0.0, 10.0, 5.0])
    f = oh_piecewise_linear(ts, vs)
    assert float(f(jnp.asarray(0.0))) == pytest.approx(0.0)
    assert float(f(jnp.asarray(0.5))) == pytest.approx(5.0)
    assert float(f(jnp.asarray(1.0))) == pytest.approx(10.0)
    assert float(f(jnp.asarray(1.5))) == pytest.approx(7.5)
    assert float(f(jnp.asarray(2.0))) == pytest.approx(5.0)


def test_oh_exponential_decay_at_t0() -> None:
    f = oh_exponential_decay(value0=2.0, decay_rate=0.5, t0=1.0)
    assert float(f(jnp.asarray(1.0))) == pytest.approx(2.0)
    # At t = t0 + 2/decay_rate = 5.0, should decay by exp(-1).
    assert float(f(jnp.asarray(5.0))) == pytest.approx(2.0 * np.exp(-2.0))


def test_as_oh_callable_passes_through_callables() -> None:
    f = oh_constant(3.0)
    assert as_oh_callable(f) is f


def test_as_oh_callable_wraps_scalars() -> None:
    f = as_oh_callable(2.5)
    assert callable(f)
    assert float(f(jnp.asarray(7.0))) == pytest.approx(2.5)


# --- simulate() integrates with a callable OH ---------------------------


def test_simulate_accepts_callable_oh(network: SOMNetwork) -> None:
    """Scalar OH=c and callable OH(t)=c must produce the same trajectory
    to solver precision."""
    bl20_idx = network.reaction_index("BL20")
    k_bl20 = float(network.k_OH[bl20_idx])
    oh = 1e-4
    t_final = 1.0 / (k_bl20 * oh)
    save_at = jnp.linspace(0.0, t_final, 20)
    y0 = build_initial(network, {"GENVOC": 1.0})

    traj_scalar = simulate(network, y0, oh=oh, t_span=(0.0, t_final), save_at=save_at, rtol=1e-8)
    traj_callable = simulate(
        network, y0, oh=oh_constant(oh), t_span=(0.0, t_final), save_at=save_at, rtol=1e-8
    )

    np.testing.assert_allclose(
        np.asarray(traj_scalar.y), np.asarray(traj_callable.y), rtol=1e-10, atol=1e-14
    )


def test_genvoc_decay_under_ramp_oh_matches_analytic(network: SOMNetwork) -> None:
    """With OH(t) = a + b*t, the analytic solution for GENVOC is
    [GENVOC](t) = exp(-k * integral(OH, 0, t)) = exp(-k * (a*t + b*t^2/2)).

    This test verifies the full time-varying machinery — diffrax must
    re-evaluate OH at every sub-step, and the trajectory has to match a
    non-trivial analytic expression rather than a pure exponential.
    """
    bl20_idx = network.reaction_index("BL20")
    k_bl20 = float(network.k_OH[bl20_idx])

    oh_start = 5e-5
    oh_end = 2e-4
    # Integrate over a time where integrated k*OH reaches ~2 (a clean decay).
    oh_mean = 0.5 * (oh_start + oh_end)
    t_final = 2.0 / (k_bl20 * oh_mean)

    save_at = jnp.linspace(0.0, t_final, 40)
    y0 = build_initial(network, {"GENVOC": 1.0})
    oh_fn = oh_linear_ramp(0.0, t_final, oh_start, oh_end)
    traj = simulate(network, y0, oh=oh_fn, t_span=(0.0, t_final), save_at=save_at, rtol=1e-8)

    # Analytic integral of OH(t) = oh_start + (oh_end - oh_start)/t_final * t.
    a = oh_start
    b = (oh_end - oh_start) / t_final
    t = np.asarray(save_at)
    integ_oh = a * t + 0.5 * b * t**2
    genvoc_analytic = np.exp(-k_bl20 * integ_oh)
    genvoc_sim = np.asarray(traj.y_of("GENVOC"))

    np.testing.assert_allclose(genvoc_sim, genvoc_analytic, rtol=1e-5, atol=1e-12)


def test_simulate_under_jit_with_callable_oh(network: SOMNetwork) -> None:
    """jit(simulate) must work with a callable OH. The closure capturing
    ``oh_callable`` becomes part of the compiled function identity."""
    y0 = build_initial(network, {"GENVOC": 1.0})
    save_at = jnp.linspace(0.0, 0.1, 5)
    oh_fn = oh_linear_ramp(0.0, 0.1, 1e-4, 2e-4)

    @jax.jit
    def run_sim() -> jnp.ndarray:
        traj = simulate(network, y0, oh=oh_fn, t_span=(0.0, 0.1), save_at=save_at)
        return traj.y_of("GENVOC")

    jit_result = np.asarray(run_sim())
    eager_result = np.asarray(
        simulate(network, y0, oh=oh_fn, t_span=(0.0, 0.1), save_at=save_at).y_of("GENVOC")
    )
    np.testing.assert_allclose(jit_result, eager_result, rtol=1e-10, atol=1e-14)


def test_grad_through_oh_parameter(network: SOMNetwork) -> None:
    """jax.grad through a parametric OH trajectory works. This is the
    smallest demonstration of the differentiability-through-OH pathway
    that later chunks will use for chamber-data fitting."""
    bl20_idx = network.reaction_index("BL20")
    k_bl20 = float(network.k_OH[bl20_idx])
    y0 = build_initial(network, {"GENVOC": 1.0})
    t_final = 0.5 / k_bl20 / 1e-4
    save_at = jnp.asarray([0.0, t_final])

    def final_genvoc(oh_peak: jnp.ndarray) -> jnp.ndarray:
        """GENVOC concentration at t_final under a ramp 0 -> oh_peak."""
        oh_fn = oh_linear_ramp(0.0, t_final, 0.0, oh_peak)
        traj = simulate(network, y0, oh=oh_fn, t_span=(0.0, t_final), save_at=save_at, rtol=1e-8)
        return traj.y_of("GENVOC")[-1]

    # Analytic check: integral of ramp 0->p over [0,t_final] = p * t_final / 2.
    # So GENVOC(t_final) = exp(-k * p * t_final / 2).
    # d/dp GENVOC(t_final) = -k * t_final / 2 * exp(-k * p * t_final / 2).
    oh_peak = jnp.asarray(1e-4)
    grad_sim = float(jax.grad(final_genvoc)(oh_peak))
    grad_analytic = -k_bl20 * t_final / 2 * np.exp(-k_bl20 * float(oh_peak) * t_final / 2)

    assert grad_sim == pytest.approx(grad_analytic, rel=1e-4)


def test_piecewise_linear_oh_trajectory_integrates(network: SOMNetwork) -> None:
    """Smoke test: simulate() runs cleanly with a piecewise-linear OH
    profile (chamber-style step-up followed by step-down)."""
    bl20_idx = network.reaction_index("BL20")
    k_bl20 = float(network.k_OH[bl20_idx])
    t_final = 1.0 / (k_bl20 * 1e-4)
    y0 = build_initial(network, {"GENVOC": 1.0})

    oh_fn = oh_piecewise_linear(
        times=jnp.asarray([0.0, t_final / 3, 2 * t_final / 3, t_final]),
        values=jnp.asarray([0.0, 1e-4, 2e-4, 5e-5]),
    )
    save_at = jnp.linspace(0.0, t_final, 30)
    traj = simulate(network, y0, oh=oh_fn, t_span=(0.0, t_final), save_at=save_at, rtol=1e-7)

    # GENVOC monotonically non-increasing (OH >= 0 always).
    genvoc = np.asarray(traj.y_of("GENVOC"))
    assert (np.diff(genvoc) <= 1e-12).all(), "GENVOC should be non-increasing under positive OH"
    # And has decayed materially.
    assert genvoc[-1] < 0.9
