"""Unit tests for :func:`som_jax.simulate.simulate`.

Exercises the diffrax wrapper end-to-end, including the S1.8 headline check
— constant-OH GENVOC decay must match the analytic first-order solution
within solver tolerance.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from som_jax import SOMTrajectory, build_initial, simulate
from som_jax.mechanism import SOMNetwork

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"


@pytest.fixture(scope="module")
def network() -> SOMNetwork:
    return SOMNetwork.from_json(_GENSOMG_JSON)


# --- basic smoke tests --------------------------------------------------


def test_zero_initial_trajectory_stays_zero(network: SOMNetwork) -> None:
    y0 = jnp.zeros(network.n_species, dtype=jnp.float64)
    save_at = jnp.linspace(0.0, 1.0, 5)
    traj = simulate(network, y0, oh=1e-4, t_span=(0.0, 1.0), save_at=save_at)
    np.testing.assert_allclose(np.asarray(traj.y), 0.0, atol=1e-14)


def test_trajectory_shape_and_species_names(network: SOMNetwork) -> None:
    y0 = build_initial(network, {"GENVOC": 1.0})
    save_at = jnp.linspace(0.0, 1.0, 10)
    traj = simulate(network, y0, oh=1e-4, t_span=(0.0, 1.0), save_at=save_at)
    assert traj.y.shape == (10, network.n_species)
    assert traj.t.shape == (10,)
    assert traj.species_names == network.species_names


def test_initial_condition_preserved(network: SOMNetwork) -> None:
    """y(t=t0) in the trajectory must match the supplied initial condition
    to numerical precision."""
    y0 = build_initial(network, {"GENVOC": 0.1, "GENSOMG_04_03": 0.05})
    save_at = jnp.asarray([0.0, 1.0])
    traj = simulate(network, y0, oh=1e-5, t_span=(0.0, 1.0), save_at=save_at)
    np.testing.assert_allclose(np.asarray(traj.y[0]), np.asarray(y0), atol=1e-14)


def test_rejects_wrong_initial_shape(network: SOMNetwork) -> None:
    y0 = jnp.zeros(network.n_species - 1)
    save_at = jnp.asarray([0.0, 1.0])
    with pytest.raises(ValueError, match="expected"):
        simulate(network, y0, oh=1e-4, t_span=(0.0, 1.0), save_at=save_at)


def test_y_of_returns_single_species_trajectory(network: SOMNetwork) -> None:
    y0 = build_initial(network, {"GENVOC": 1.0})
    save_at = jnp.linspace(0.0, 0.1, 5)
    traj = simulate(network, y0, oh=1e-4, t_span=(0.0, 0.1), save_at=save_at)
    genvoc = traj.y_of("GENVOC")
    assert genvoc.shape == (5,)
    np.testing.assert_array_equal(
        np.asarray(genvoc), np.asarray(traj.y[:, network.species_index("GENVOC")])
    )


# --- build_initial helper -----------------------------------------------


def test_build_initial_zero_default(network: SOMNetwork) -> None:
    y = build_initial(network, {})
    assert y.shape == (network.n_species,)
    assert bool(jnp.all(y == 0))


def test_build_initial_sets_named_species(network: SOMNetwork) -> None:
    y = build_initial(network, {"GENVOC": 1.0, "GENSOMG_04_03": 0.5})
    genvoc_idx = network.species_index("GENVOC")
    gensomg_idx = network.species_index("GENSOMG_04_03")
    assert float(y[genvoc_idx]) == pytest.approx(1.0)
    assert float(y[gensomg_idx]) == pytest.approx(0.5)
    mask = np.ones(network.n_species, dtype=bool)
    mask[genvoc_idx] = False
    mask[gensomg_idx] = False
    np.testing.assert_array_equal(np.asarray(y)[mask], np.zeros(network.n_species - 2))


def test_build_initial_rejects_unknown_species(network: SOMNetwork) -> None:
    with pytest.raises(KeyError):
        build_initial(network, {"NOT_A_SPECIES": 1.0})


# --- S1.8: analytic first-order decay (the headline test) ---------------


def test_genvoc_first_order_decay_under_constant_oh(network: SOMNetwork) -> None:
    """GENVOC is never produced by any reaction — it only decays via BL20.

    Under constant OH, its concentration follows the exact first-order
    solution [GENVOC](t) = [GENVOC]_0 * exp(-k_BL20 * OH * t). This is the
    definitive correctness check on the simulate wrapper plumbing: any
    stoichiometry error, unit confusion, or solver misconfiguration
    shows up here as deviation from a simple analytic curve.
    """
    bl20_idx = network.reaction_index("BL20")
    k_bl20 = float(network.k_OH[bl20_idx])

    # Choose OH and t_final so decay rate * t_final = 2, giving a clean
    # decay from 1.0 to ~0.135 — well inside the solver's dynamic range.
    oh = 1e-4
    t_final = 2.0 / (k_bl20 * oh)
    save_at = jnp.linspace(0.0, t_final, 41)

    y0 = build_initial(network, {"GENVOC": 1.0})
    traj = simulate(
        network, y0, oh=oh, t_span=(0.0, t_final), save_at=save_at, rtol=1e-8, atol=1e-14
    )

    genvoc_sim = np.asarray(traj.y_of("GENVOC"))
    genvoc_analytic = np.exp(-k_bl20 * oh * np.asarray(save_at))

    # Tolerance: rtol=1e-5 is 1000x looser than the solver's rtol=1e-8
    # and still well below the 0.1% scientific-faithfulness bar.
    np.testing.assert_allclose(genvoc_sim, genvoc_analytic, rtol=1e-5, atol=1e-12)


def test_genvoc_decay_scales_with_oh(network: SOMNetwork) -> None:
    """Doubling OH must halve the decay time-constant. Integrate to the
    same fractional decay under two OH values; the ratio of required
    times matches the ratio of rate constants."""
    bl20_idx = network.reaction_index("BL20")
    k_bl20 = float(network.k_OH[bl20_idx])

    def final_genvoc(oh: float, t_final: float) -> float:
        save_at = jnp.asarray([0.0, t_final])
        y0 = build_initial(network, {"GENVOC": 1.0})
        traj = simulate(network, y0, oh=oh, t_span=(0.0, t_final), save_at=save_at, rtol=1e-8)
        return float(traj.y_of("GENVOC")[-1])

    # Pick t_final so single-step decay to ~0.5 with oh=1e-4.
    oh_a = 1e-4
    t_final_a = np.log(2.0) / (k_bl20 * oh_a)
    # Same fractional decay with oh=2e-4 should take half the time.
    oh_b = 2e-4
    t_final_b = t_final_a / 2.0

    frac_a = final_genvoc(oh_a, t_final_a)
    frac_b = final_genvoc(oh_b, t_final_b)

    # Both should reach exp(-ln 2) = 0.5 within solver tolerance.
    assert frac_a == pytest.approx(0.5, rel=1e-5)
    assert frac_b == pytest.approx(0.5, rel=1e-5)


# --- SOMTrajectory PyTree registration -----------------------------------


def test_trajectory_is_pytree(network: SOMNetwork) -> None:
    y0 = build_initial(network, {"GENVOC": 1.0})
    save_at = jnp.linspace(0.0, 0.1, 5)
    traj = simulate(network, y0, oh=1e-4, t_span=(0.0, 0.1), save_at=save_at)

    leaves, treedef = jax.tree.flatten(traj)
    restored = jax.tree.unflatten(treedef, leaves)
    assert isinstance(restored, SOMTrajectory)
    np.testing.assert_array_equal(np.asarray(restored.t), np.asarray(traj.t))
    np.testing.assert_array_equal(np.asarray(restored.y), np.asarray(traj.y))
    assert restored.species_names == traj.species_names
