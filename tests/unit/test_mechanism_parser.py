"""Unit tests for :mod:`som_jax.mechanism.parser`.

These tests work entirely from the committed ``data/mechanisms/gensomg.json``
artifact so they do not require the Fortran source tree. A separate, gated
regression test (see ``tests/regression/``, not yet implemented) re-runs the
parser and diffs against the committed JSON.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from som_jax.mechanism import (
    Mechanism,
    Product,
    Reaction,
    mechanism_from_json,
    mechanism_to_json,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"


@pytest.fixture(scope="module")
def mech() -> Mechanism:
    assert _GENSOMG_JSON.exists(), (
        f"missing {_GENSOMG_JSON}; regenerate with 'python scripts/generate_mechanism_json.py ...'"
    )
    return mechanism_from_json(_GENSOMG_JSON)


# --- structural tests -----------------------------------------------------


def test_family_and_precursor(mech: Mechanism) -> None:
    assert mech.family == "GENSOMG"
    assert mech.precursor == "GENVOC"
    assert mech.grid_c_max == 7
    assert mech.grid_o_max == 7


def test_species_count_matches_grid(mech: Mechanism) -> None:
    # GENSOMG has 40 products on the grid (C=1 has 2 entries, C=2 has 4, C=3 has 6,
    # and C=4..7 have 7 each: 2 + 4 + 6 + 4*7 = 40). Plus GENVOC = 41 total.
    assert len(mech.species) == 41
    precursors = [sp for sp in mech.species if sp.is_precursor]
    assert len(precursors) == 1
    assert precursors[0].name == "GENVOC"


def test_genvoc_species_card(mech: Mechanism) -> None:
    genvoc = mech.species_by_name("GENVOC")
    assert genvoc.is_precursor
    assert genvoc.carbon == 7
    assert genvoc.oxygen == 0
    assert math.isclose(genvoc.molecular_weight, 92.14, abs_tol=1e-6)


@pytest.mark.parametrize(
    "name,carbon,oxygen,mw",
    [
        ("GENSOMG_01_01", 1, 1, 31.0),
        ("GENSOMG_04_03", 4, 3, 103.0),
        ("GENSOMG_07_07", 7, 7, 205.0),
    ],
)
def test_species_sample_cards(
    mech: Mechanism, name: str, carbon: int, oxygen: int, mw: float
) -> None:
    sp = mech.species_by_name(name)
    assert sp.carbon == carbon
    assert sp.oxygen == oxygen
    assert math.isclose(sp.molecular_weight, mw, abs_tol=1e-6)
    assert not sp.is_precursor


def test_all_reactions_are_OH_driven(mech: Mechanism) -> None:
    # Every SOM reaction in the Fortran reference has exactly two reactants:
    # the SOM species (precursor or grid point) and OH.
    for r in mech.reactions:
        assert len(r.reactants) == 2
        assert r.reactants[1] == "OH"


def test_reaction_count(mech: Mechanism) -> None:
    # 1 GENVOC+OH (BL20) + 38 GENSOMG+OH grid reactions (S1.1..S38.1).
    assert len(mech.reactions) == 39
    labels = [r.label for r in mech.reactions]
    assert labels[0] == "BL20"
    assert labels[1] == "S1.1"
    assert labels[-1] == "S38.1"


# --- content tests --------------------------------------------------------


def test_genvoc_oh_reaction(mech: Mechanism) -> None:
    r = mech.reactions[0]
    assert r.label == "BL20"
    assert r.reactants == ("GENVOC", "OH")
    # Four products with branch=1.000 and oxy_yields that sum to 1.0.
    assert len(r.products) == 4
    product_names = {p.name for p in r.products}
    assert product_names == {
        "GENSOMG_07_01",
        "GENSOMG_07_02",
        "GENSOMG_07_03",
        "GENSOMG_07_04",
    }
    for p in r.products:
        assert math.isclose(p.branch, 1.0, abs_tol=1e-6)
    assert math.isclose(r.total_yield, 1.0, abs_tol=1e-6)
    # Rate at T=300 K from saprc14_rev1.doc: 8.319E+03 cm^3 mol^-1 s^-1.
    assert math.isclose(r.rate_cm3_per_mol_per_s, 8.319e3, rel_tol=1e-6)


def test_first_grid_reaction_sum_reflects_fragmentation(mech: Mechanism) -> None:
    # S1.1 is GENSOMG_02_01 + OH. It has on-grid branch 0.969 (3 products with
    # oxy_yield summing to 1.0) plus fragmentation branch 0.031 producing two
    # C=1 fragments, so total effective yield ~= 0.969 + 2*0.031 = 1.031.
    r = next(r for r in mech.reactions if r.label == "S1.1")
    assert r.reactants[0] == "GENSOMG_02_01"
    assert math.isclose(r.total_yield, 1.031, abs_tol=1e-6)
    # Rate from .doc: 1.759E+03.
    assert math.isclose(r.rate_cm3_per_mol_per_s, 1.759e3, rel_tol=1e-6)


def test_every_product_species_is_known(mech: Mechanism) -> None:
    known = {sp.name for sp in mech.species}
    for r in mech.reactions:
        for p in r.products:
            assert p.name in known, (
                f"reaction {r.label} ({' + '.join(r.reactants)}) "
                f"references unknown product {p.name!r}"
            )


def test_rate_constants_are_finite_and_positive(mech: Mechanism) -> None:
    for r in mech.reactions:
        assert math.isfinite(r.rate_cm3_per_mol_per_s)
        assert r.rate_cm3_per_mol_per_s > 0.0, (
            f"reaction {r.label} has non-positive rate {r.rate_cm3_per_mol_per_s}"
        )


# --- serialisation tests --------------------------------------------------


def test_json_round_trip_preserves_mechanism(mech: Mechanism, tmp_path: Path) -> None:
    out = tmp_path / "gensomg_rt.json"
    mechanism_to_json(mech, out)
    rt = mechanism_from_json(out)
    assert rt.family == mech.family
    assert rt.precursor == mech.precursor
    assert rt.grid_c_max == mech.grid_c_max
    assert rt.grid_o_max == mech.grid_o_max
    assert rt.species == mech.species
    assert rt.reactions == mech.reactions


def test_metadata_records_source_shas(mech: Mechanism) -> None:
    assert mech.metadata is not None
    names = {s.path for s in mech.metadata.source_files}
    assert {"saprc14_rev1.mod", "saprc14_rev1.doc"}.issubset(names)
    for s in mech.metadata.source_files:
        assert len(s.sha256) == 64  # hex-encoded SHA-256


def test_product_yield_property_matches_branch_times_oxy() -> None:
    p = Product(name="DUMMY", branch=0.969, oxy_yield=0.123)
    assert math.isclose(p.yield_, 0.969 * 0.123, abs_tol=1e-12)


def test_reaction_total_yield_is_sum_of_product_yields() -> None:
    r = Reaction(
        label="T.1",
        reactants=("A", "OH"),
        products=(
            Product(name="B", branch=0.5, oxy_yield=0.4),
            Product(name="C", branch=0.5, oxy_yield=0.6),
        ),
        rate_cm3_per_mol_per_s=1.0e3,
        source_line_mod=0,
        source_line_doc=0,
    )
    assert math.isclose(r.total_yield, 0.5 * 0.4 + 0.5 * 0.6, abs_tol=1e-12)
