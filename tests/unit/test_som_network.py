"""Unit tests for :class:`som_jax.mechanism.network.SOMNetwork`.

Exercises the JAX-native representation: shape, dtypes, stoichiometry
correctness, PyTree round-trip, and ``jax.jit`` compatibility.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from som_jax.mechanism import Mechanism, SOMNetwork, mechanism_from_json

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"


@pytest.fixture(scope="module")
def mechanism() -> Mechanism:
    return mechanism_from_json(_GENSOMG_JSON)


@pytest.fixture(scope="module")
def network(mechanism: Mechanism) -> SOMNetwork:
    return SOMNetwork.from_mechanism(mechanism)


# --- shapes and dtypes ---------------------------------------------------


def test_shapes_match_mechanism(mechanism: Mechanism, network: SOMNetwork) -> None:
    n_species = len(mechanism.species)
    n_reactions = len(mechanism.reactions)
    assert network.n_species == n_species
    assert network.n_reactions == n_reactions
    assert network.carbon.shape == (n_species,)
    assert network.oxygen.shape == (n_species,)
    assert network.molecular_weight.shape == (n_species,)
    assert network.is_precursor.shape == (n_species,)
    assert network.oh_reactant_idx.shape == (n_reactions,)
    assert network.stoich.shape == (n_reactions, n_species)
    assert network.k_OH.shape == (n_reactions,)


def test_dtypes_are_float64_under_x64(network: SOMNetwork) -> None:
    # conftest.py enables x64; verify it took effect.
    assert network.molecular_weight.dtype == jnp.float64
    assert network.stoich.dtype == jnp.float64
    assert network.k_OH.dtype == jnp.float64
    assert network.carbon.dtype == jnp.int32
    assert network.oxygen.dtype == jnp.int32
    assert network.is_precursor.dtype == jnp.bool_


def test_names_match_mechanism_order(mechanism: Mechanism, network: SOMNetwork) -> None:
    assert network.species_names == tuple(sp.name for sp in mechanism.species)
    assert network.reaction_labels == tuple(r.label for r in mechanism.reactions)


def test_genvoc_marked_as_precursor(network: SOMNetwork) -> None:
    i = network.species_index("GENVOC")
    assert bool(network.is_precursor[i])
    # All other species are not precursors.
    other = np.delete(np.asarray(network.is_precursor), i)
    assert not other.any()


# --- stoichiometry correctness ------------------------------------------


def test_every_reaction_has_exactly_one_reactant_at_minus_one(
    network: SOMNetwork,
) -> None:
    # For every row, the reactant column equals -1 and no other column equals -1.
    stoich = np.asarray(network.stoich)
    for i in range(network.n_reactions):
        r_idx = int(network.oh_reactant_idx[i])
        assert stoich[i, r_idx] == pytest.approx(-1.0)
        mask = np.arange(stoich.shape[1]) != r_idx
        other_cols = stoich[i, mask]
        # Product columns hold +yield; the -1 must be unique.
        assert (other_cols >= 0).all(), (
            f"reaction {network.reaction_labels[i]} has multiple negative entries"
        )


def test_bl20_stoichiometry_matches_known_yields(mechanism: Mechanism, network: SOMNetwork) -> None:
    # BL20 is GENVOC + OH -> four GENSOMG_07_* products with the Fortran
    # oxygen-yield distribution 0.123 / 0.001 / 0.002 / 0.874 and branch=1.
    rxn_idx = network.reaction_index("BL20")
    stoich_row = np.asarray(network.stoich[rxn_idx])
    genvoc_idx = network.species_index("GENVOC")
    assert stoich_row[genvoc_idx] == pytest.approx(-1.0)
    expected = {
        "GENSOMG_07_01": 0.123,
        "GENSOMG_07_02": 0.001,
        "GENSOMG_07_03": 0.002,
        "GENSOMG_07_04": 0.874,
    }
    for name, yld in expected.items():
        assert stoich_row[network.species_index(name)] == pytest.approx(yld, abs=1e-6)


def test_s1_1_fragmentation_stoichiometry(network: SOMNetwork) -> None:
    # S1.1 is GENSOMG_02_01 + OH. On-grid branch 0.969 * (0.123, 0.001, 0.876),
    # fragmentation branch 0.031 producing GENSOMG_01_01 and GENSOMG_01_02.
    rxn_idx = network.reaction_index("S1.1")
    stoich_row = np.asarray(network.stoich[rxn_idx])
    assert stoich_row[network.species_index("GENSOMG_02_01")] == pytest.approx(-1.0)
    assert stoich_row[network.species_index("GENSOMG_02_02")] == pytest.approx(
        0.969 * 0.123, abs=1e-6
    )
    assert stoich_row[network.species_index("GENSOMG_02_03")] == pytest.approx(
        0.969 * 0.001, abs=1e-6
    )
    assert stoich_row[network.species_index("GENSOMG_02_04")] == pytest.approx(
        0.969 * 0.876, abs=1e-6
    )
    assert stoich_row[network.species_index("GENSOMG_01_01")] == pytest.approx(0.031, abs=1e-6)
    assert stoich_row[network.species_index("GENSOMG_01_02")] == pytest.approx(0.031, abs=1e-6)


def test_rate_constants_match_mechanism(mechanism: Mechanism, network: SOMNetwork) -> None:
    rates_from_mech = np.array(
        [r.rate_cm3_per_mol_per_s for r in mechanism.reactions], dtype=np.float64
    )
    np.testing.assert_allclose(np.asarray(network.k_OH), rates_from_mech, rtol=0, atol=0)


# --- PyTree machinery ---------------------------------------------------


def test_flatten_unflatten_round_trip(network: SOMNetwork) -> None:
    leaves, treedef = jax.tree.flatten(network)
    restored = jax.tree.unflatten(treedef, leaves)
    assert isinstance(restored, SOMNetwork)
    assert restored.species_names == network.species_names
    assert restored.reaction_labels == network.reaction_labels
    np.testing.assert_array_equal(np.asarray(restored.stoich), np.asarray(network.stoich))
    np.testing.assert_array_equal(np.asarray(restored.k_OH), np.asarray(network.k_OH))


def test_tree_map_scales_numeric_leaves(network: SOMNetwork) -> None:
    scaled = jax.tree.map(lambda x: x * 2 if jnp.issubdtype(x.dtype, jnp.floating) else x, network)
    np.testing.assert_allclose(
        np.asarray(scaled.k_OH), 2 * np.asarray(network.k_OH), rtol=0, atol=0
    )
    # Integer fields (carbon, oxygen, oh_reactant_idx) should be unchanged by
    # the conditional in tree.map; species names (aux) are always unchanged.
    np.testing.assert_array_equal(np.asarray(scaled.carbon), np.asarray(network.carbon))
    assert scaled.species_names == network.species_names


def test_jit_through_network_rate_sum(network: SOMNetwork) -> None:
    @jax.jit
    def total_rate_coefficient(n: SOMNetwork) -> Array:  # type: ignore[name-defined]
        return n.k_OH.sum()

    expected = float(np.asarray(network.k_OH).sum())
    assert float(total_rate_coefficient(network)) == pytest.approx(expected, rel=1e-12)


# --- from_json convenience ----------------------------------------------


def test_from_json_matches_from_mechanism(network: SOMNetwork) -> None:
    via_json = SOMNetwork.from_json(_GENSOMG_JSON)
    assert via_json.species_names == network.species_names
    assert via_json.reaction_labels == network.reaction_labels
    np.testing.assert_array_equal(np.asarray(via_json.stoich), np.asarray(network.stoich))
    np.testing.assert_array_equal(np.asarray(via_json.k_OH), np.asarray(network.k_OH))


# Needed for the jit test signature annotation.
Array = jax.Array
