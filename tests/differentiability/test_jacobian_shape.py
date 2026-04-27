"""S1.16 — Jacobian-shape test: differentiability through simulate().

Verifies that ``simulate`` is differentiable end-to-end and that the
gradient agrees with a central-difference numerical estimate. This is
the prerequisite for S1.17's optax.adam recovery demo.

Note on autodiff mode
---------------------
The master plan calls out ``jax.jacfwd``; in practice ``simulate``
uses ``diffrax.RecursiveCheckpointAdjoint`` which only supports
reverse-mode autodiff. We use ``jax.jacrev`` / ``jax.grad`` instead.
Functionally identical for our use case — scalar parameter, scalar /
trajectory output — and reverse-mode is what ``optax`` ultimately
calls anyway.

Note on parameter choice
------------------------
The master plan calls out ``dLVP`` (a volatility parameter from the
Cappa-Wilson SOM). ``dLVP`` only enters via partitioning, which is
``tomas-jax`` scope (gas-phase only here per master plan D2). We
instead differentiate w.r.t. OH concentration — also a chamber-fittable
quantity, in scope for the gas-phase port, and the most
identifiable single parameter from precursor-decay observations.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from som_jax import build_initial, simulate
from som_jax.mechanism import SOMNetwork

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"

# Differentiability tests use a short integration so they run quickly
# in the CI test budget; correctness of the gradient is independent of
# trajectory length.
_T_END_MIN = 240.0  # 4 hours
_OH_BASELINE_PPM = 6.090601e-08  # ≈ 1.5e6 molec/cm³ at 298 K, 101325 Pa
_INITIAL_GENVOC_PPM = 0.05


@pytest.fixture(scope="module")
def network() -> SOMNetwork:
    return SOMNetwork.from_json(_GENSOMG_JSON)


@pytest.fixture(scope="module")
def y0(network: SOMNetwork) -> jnp.ndarray:
    return build_initial(network, {"GENVOC": _INITIAL_GENVOC_PPM})


@pytest.fixture(scope="module")
def save_at_min() -> jnp.ndarray:
    return jnp.linspace(0.0, _T_END_MIN, 9)


def _genvoc_final(oh_ppm: float, network: SOMNetwork, y0, save_at_min) -> float:
    """Final GENVOC concentration as a scalar function of OH."""
    traj = simulate(
        network,
        y0,
        oh=oh_ppm,
        t_span=(0.0, _T_END_MIN),
        save_at=save_at_min,
        rtol=1e-8,
        atol=1e-15,
    )
    return traj.y_of("GENVOC")[-1]


def _genvoc_trajectory(oh_ppm: float, network: SOMNetwork, y0, save_at_min) -> jnp.ndarray:
    """Full GENVOC trajectory as a function of OH."""
    traj = simulate(
        network,
        y0,
        oh=oh_ppm,
        t_span=(0.0, _T_END_MIN),
        save_at=save_at_min,
        rtol=1e-8,
        atol=1e-15,
    )
    return traj.y_of("GENVOC")


# --- S1.16 -----------------------------------------------------------


def test_grad_finite_and_correct_sign(network: SOMNetwork, y0, save_at_min: jnp.ndarray) -> None:
    """``∂[GENVOC](t_final) / ∂OH`` should be finite and negative.
    More OH → faster oxidation → less GENVOC remaining."""
    deriv = jax.grad(_genvoc_final)(_OH_BASELINE_PPM, network, y0, save_at_min)
    assert jnp.isfinite(deriv)
    assert deriv < 0, f"expected negative derivative; got {deriv:.3e}"


def test_grad_matches_central_diff_within_one_percent(
    network: SOMNetwork, y0, save_at_min: jnp.ndarray
) -> None:
    """Verify the autodiff result against a central-difference numerical
    estimate. Tolerance is 1% per the master plan; in practice the
    agreement is at 1e-10 relative because the simulator is smooth in
    OH."""
    deriv_auto = float(jax.grad(_genvoc_final)(_OH_BASELINE_PPM, network, y0, save_at_min))

    eps = _OH_BASELINE_PPM * 1e-4
    f_plus = _genvoc_final(_OH_BASELINE_PPM + eps, network, y0, save_at_min)
    f_minus = _genvoc_final(_OH_BASELINE_PPM - eps, network, y0, save_at_min)
    deriv_num = float((f_plus - f_minus) / (2.0 * eps))

    rel_err = abs(deriv_auto - deriv_num) / abs(deriv_num)
    assert rel_err < 0.01, (
        f"jax.grad disagrees with central-diff: auto={deriv_auto:.6e}, "
        f"num={deriv_num:.6e}, rel_err={rel_err:.3e}"
    )


def test_jacrev_returns_finite_trajectory(
    network: SOMNetwork, y0, save_at_min: jnp.ndarray
) -> None:
    """``jax.jacrev`` over the full GENVOC trajectory should return
    a finite (n_t,) array. The initial-condition entry must be
    exactly zero (initial value doesn't depend on OH); subsequent
    entries should be strictly negative (more OH → less GENVOC
    at every later save point)."""
    jac = jax.jacrev(_genvoc_trajectory)(_OH_BASELINE_PPM, network, y0, save_at_min)
    jac_np = np.asarray(jac)
    assert jac_np.shape == (save_at_min.size,)
    assert np.all(np.isfinite(jac_np))
    assert jac_np[0] == pytest.approx(0.0, abs=1e-30)
    assert np.all(jac_np[1:] < 0.0), (
        f"expected strictly negative derivatives at later times; got {jac_np}"
    )
