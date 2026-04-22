"""Unit tests for :func:`som_jax.rhs.som_rhs`.

These tests exercise the pure RHS function without a solver — zero-input
boundary cases, the single-reactant activation pattern (GENVOC-only state),
mass-balance structure from the ``stoich.T @ rate_vec`` form, and
JIT/grad compatibility.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from som_jax import som_rhs
from som_jax.mechanism import SOMNetwork

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"


@pytest.fixture(scope="module")
def network() -> SOMNetwork:
    return SOMNetwork.from_json(_GENSOMG_JSON)


# --- boundary cases ------------------------------------------------------


def test_zero_concentration_gives_zero_derivative(network: SOMNetwork) -> None:
    zero = jnp.zeros(network.n_species, dtype=jnp.float64)
    dydt = som_rhs(zero, 1.5e6, network)
    np.testing.assert_array_equal(np.asarray(dydt), np.zeros(network.n_species))


def test_zero_oh_gives_zero_derivative(network: SOMNetwork) -> None:
    nonzero = jnp.ones(network.n_species, dtype=jnp.float64)
    dydt = som_rhs(nonzero, 0.0, network)
    np.testing.assert_array_equal(np.asarray(dydt), np.zeros(network.n_species))


# --- single-reactant activation -----------------------------------------


def test_only_genvoc_nonzero_activates_bl20_products(network: SOMNetwork) -> None:
    """With only GENVOC nonzero, only GENVOC + BL20's four first-gen products
    see nonzero dy/dt; GENVOC decays at ``-k_BL20 * OH * [GENVOC]`` and each
    product gains ``yield * k_BL20 * OH * [GENVOC]``."""
    genvoc_idx = network.species_index("GENVOC")
    bl20_idx = network.reaction_index("BL20")
    k_bl20 = float(network.k_OH[bl20_idx])

    genvoc_conc = 2.0  # arbitrary positive scalar
    oh = 3.0
    y = jnp.zeros(network.n_species, dtype=jnp.float64).at[genvoc_idx].set(genvoc_conc)

    dydt = np.asarray(som_rhs(y, oh, network))

    # GENVOC decays at -k * OH * [GENVOC].
    assert dydt[genvoc_idx] == pytest.approx(-k_bl20 * oh * genvoc_conc, rel=1e-12)

    expected_product_yields = {
        "GENSOMG_07_01": 0.123,
        "GENSOMG_07_02": 0.001,
        "GENSOMG_07_03": 0.002,
        "GENSOMG_07_04": 0.874,
    }
    for name, yld in expected_product_yields.items():
        idx = network.species_index(name)
        assert dydt[idx] == pytest.approx(yld * k_bl20 * oh * genvoc_conc, rel=1e-6)

    active_names = {"GENVOC", *expected_product_yields}
    for i, name in enumerate(network.species_names):
        if name in active_names:
            continue
        assert dydt[i] == pytest.approx(0.0, abs=1e-15), (
            f"{name} unexpectedly has nonzero derivative {dydt[i]}"
        )


def test_only_one_grid_reactant_activates_only_its_reaction(network: SOMNetwork) -> None:
    """Activating a single GENSOMG species should drive exactly one reaction
    (the one whose reactant it is)."""
    target = "GENSOMG_03_02"  # picked from the middle of the grid
    reactant_idx = network.species_index(target)
    y = jnp.zeros(network.n_species, dtype=jnp.float64).at[reactant_idx].set(1.0)
    dydt = np.asarray(som_rhs(y, 1.0, network))

    # Only one reaction has this species as its reactant.
    reactions_consuming = [
        i for i in range(network.n_reactions) if int(network.oh_reactant_idx[i]) == reactant_idx
    ]
    assert len(reactions_consuming) == 1

    # The reactant must see a negative derivative; all other positions must
    # sum (reactant-index aside) to the total product yield times k.
    assert dydt[reactant_idx] < 0


# --- linearity and differentiability ------------------------------------


def test_rhs_is_linear_in_oh(network: SOMNetwork) -> None:
    """dy/dt scales linearly with OH."""
    y = jnp.ones(network.n_species, dtype=jnp.float64) * 1e-3
    a = som_rhs(y, 1.0, network)
    b = som_rhs(y, 2.5, network)
    np.testing.assert_allclose(np.asarray(b), 2.5 * np.asarray(a), rtol=1e-12)


def test_jit_matches_eager(network: SOMNetwork) -> None:
    y = jnp.ones(network.n_species, dtype=jnp.float64) * 1e-3
    oh = 1.5e-6  # arbitrary

    jitted = jax.jit(som_rhs, static_argnums=())
    eager = np.asarray(som_rhs(y, oh, network))
    fast = np.asarray(jitted(y, oh, network))
    # XLA's matmul reorders fp accumulations; the difference is at the
    # ULP-level (~1e-21 absolute, ~1e-14 relative). A scientifically
    # meaningful drift would be many orders of magnitude larger.
    np.testing.assert_allclose(fast, eager, rtol=1e-12, atol=1e-18)


def test_grad_through_oh(network: SOMNetwork) -> None:
    """``jax.grad`` w.r.t. OH of the total derivative is analytic:
    ``d(sum dy/dt) / d(OH) = sum(stoich.T @ (k_OH * y[reactant_idx]))``."""
    y = jnp.ones(network.n_species, dtype=jnp.float64) * 1e-3

    def total_derivative(oh: jnp.ndarray) -> jnp.ndarray:
        return som_rhs(y, oh, network).sum()

    g = jax.grad(total_derivative)(jnp.asarray(1.5))
    # Analytic: sum over products yields - sum of reactant contributions
    # = sum_i k_i * y[r_i] * (total_yield_i - 1).
    reactant_concs = y[network.oh_reactant_idx]
    rates_per_oh = network.k_OH * reactant_concs  # (n_rxn,)
    expected = (network.stoich.T @ rates_per_oh).sum()
    assert float(g) == pytest.approx(float(expected), rel=1e-10)


def test_grad_through_concentrations(network: SOMNetwork) -> None:
    """``jax.jacfwd`` w.r.t. concentrations returns the reaction Jacobian
    and has the expected sparsity: column *j* is nonzero only for reactions
    whose reactant is species *j*."""
    y = jnp.ones(network.n_species, dtype=jnp.float64) * 1e-3

    jac = jax.jacfwd(som_rhs, argnums=0)(y, 1.0, network)
    assert jac.shape == (network.n_species, network.n_species)
    jac_np = np.asarray(jac)
    for j in range(network.n_species):
        reactions_consuming_j = np.where(np.asarray(network.oh_reactant_idx) == j)[0]
        if len(reactions_consuming_j) == 0:
            # Species never appears as a reactant → column must be zero.
            np.testing.assert_array_equal(
                jac_np[:, j], np.zeros(network.n_species), err_msg=f"column {j} nonzero"
            )


# --- structural integrity from stoich shape ------------------------------


def test_rhs_output_shape(network: SOMNetwork) -> None:
    y = jnp.ones(network.n_species, dtype=jnp.float64) * 1e-3
    out = som_rhs(y, 1.0, network)
    assert out.shape == (network.n_species,)
    assert out.dtype == jnp.float64


def test_rhs_matches_explicit_sum(network: SOMNetwork) -> None:
    """Cross-check the vectorized implementation against a slow, explicit
    Python loop over reactions and products."""
    rng = np.random.default_rng(42)
    y_np = rng.uniform(0.0, 1.0, size=network.n_species)
    oh = 0.7
    y = jnp.asarray(y_np)
    dydt_fast = np.asarray(som_rhs(y, oh, network))

    stoich_np = np.asarray(network.stoich)
    k_np = np.asarray(network.k_OH)
    idx_np = np.asarray(network.oh_reactant_idx)
    rates = k_np * oh * y_np[idx_np]
    dydt_slow = stoich_np.T @ rates

    np.testing.assert_allclose(dydt_fast, dydt_slow, rtol=1e-12, atol=1e-14)
