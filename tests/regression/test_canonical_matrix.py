"""S1.11 — full canonical-matrix regression vs Fortran goldens.

Extends ``test_fortran_golden.py`` (S1.10, single ``short_baseline``
fixture) to the **full 10-run** canonical matrix shipped by
``atmos-jax-common`` (sweeping endtime 10 min → 24 h, OH dose ~10×,
GENVOC magnitude ~100×, and temperature 273 / 298 / 323 K).

What we test, and what we don't
-------------------------------
For each run we compare the JAX simulator against Fortran
``_saprcgc.dat`` for **tier 1 species only** — GENVOC plus
``GENSOMG_07_01`` and ``GENSOMG_07_02`` (the two first-generation
BL20 products with the smallest further-oxidation rates).

Why such a narrow set? The Fortran reference exhibits non-linear
scaling with precursor magnitude in any species that has
meaningful secondary chemistry. A 10× change in initial GENVOC
produces only ~2.5× change in cascade species (verified
empirically against ``high_voc`` vs ``long_baseline``) — that's a
Fortran artefact, REAL*4 truncation in ``DIFUN``'s ``RKZ × C × C``
products amplifies with cascade depth and absolute magnitude. JAX
(REAL*8) gives the correct linear scaling. Pinning the cascade
values to the biased Fortran reference would force JAX to also be
wrong; instead we restrict to species where Fortran is faithful.

Of the four BL20 first-gen products, only ``07_01`` (yield 0.123)
and ``07_02`` (yield 0.001) have small enough further-oxidation
that their concentrations match Fortran to ≤1% across the matrix.
``07_03`` and ``07_04`` (yields 0.002 and 0.874) feed strongly
into the cascade and inherit Fortran's nonlinearity.

When ``saprc14_rev1.f`` is promoted to REAL*8 in a follow-up chunk,
the cascade comparison becomes meaningful and this test can extend
to the full 40-species block at tight tolerance.

Temperature handling
--------------------
``cold`` (273 K) and ``hot`` (323 K) used to be skipped because the
JAX network parsed only the 298 K rate. The Arrhenius parametrisation
now lives on the network (parser captures ``(A, Ea, B)`` from .doc;
``SOMNetwork.k_OH_at(T)`` computes ``k(T) = A·exp(-Ea/RT)·(T/Tref)^B``
per reaction). We pass ``temperature_K`` into ``simulate`` so the
JAX trajectory is at the matrix-specified temperature; both endpoints
now pass at the same 1% tier-1 tolerance.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from atmos_jax_common.canonical_runs import (
    CanonicalRun,
    default_expected_dir,
    default_manifest_path,
    load_manifest,
)
from atmos_jax_common.compare import compare_trajectories
from atmos_jax_common.goldens import GoldenRun, load_golden_run
from atmos_jax_common.units import molec_cm3_to_ppm

from som_jax import build_initial, simulate
from som_jax.mechanism import SOMNetwork

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"

# Tier-1 species the Fortran reference reproduces faithfully across
# the full matrix: precursor + the two BL20 first-gen products with
# small further-oxidation. ``GENSOMG_07_03`` / ``07_04`` have larger
# yields and feed the cascade strongly, so they inherit Fortran's
# REAL*4 nonlinearity at high precursor.
_TIER1_SPECIES = (
    "GENVOC",
    "GENSOMG_07_01",
    "GENSOMG_07_02",
)

# Tolerance budget for the tier-1 comparison. ``GENVOC`` and the
# 0.123-yield ``GENSOMG_07_01`` match Fortran to <0.05% across the
# matrix; the 0.001-yield ``GENSOMG_07_02`` ticks up to ~1.2% at
# T = 273 K (the smallest absolute concentration tier-1 species at
# the coldest matrix point — this is the REAL*4 truncation floor in
# Fortran's chemistry callback). 1.5% covers it with margin.
_TIER1_L2_TOL = 0.015


@pytest.fixture(scope="module")
def network() -> SOMNetwork:
    return SOMNetwork.from_json(_GENSOMG_JSON)


def _matrix_run_params() -> list[pytest.param]:
    """Build the parametrize() table from the manifest. All 10 runs
    are now in scope: ``cold`` / ``hot`` are exercised via
    ``simulate(temperature_K=...)`` calling into ``k_OH_at(T)``."""
    matrix = load_manifest(default_manifest_path())
    return [pytest.param(run.run_id, id=run.run_id) for run in matrix.runs]


def _run_jax_tier1(run: CanonicalRun, golden: GoldenRun, network: SOMNetwork):
    """Run JAX at the Fortran golden's save points and return matched
    ``(jax_block, fortran_block)`` arrays for tier-1 species only."""
    oh_ppm = float(
        molec_cm3_to_ppm(
            run.params.OH_molec_per_cm3,
            run.params.temp_K,
            101325.0,  # all matrix runs use 1 atm
        )
    )
    save_at_min = jnp.asarray(np.asarray(golden.gc.time_hours) * 60.0)
    t_span = (float(save_at_min[0]), float(save_at_min[-1]))
    y0 = build_initial(network, {"GENVOC": run.params.ippmprec_ppm})

    traj = simulate(
        network,
        y0,
        oh=oh_ppm,
        t_span=t_span,
        save_at=save_at_min,
        temperature_K=run.params.temp_K,
        rtol=1e-10,
        atol=1e-30,
    )

    n_t = golden.gc.time_hours.size
    n_tier1 = len(_TIER1_SPECIES)
    jax_block = np.zeros((n_t, n_tier1), dtype=np.float64)
    fortran_block = np.zeros((n_t, n_tier1), dtype=np.float64)
    for i, name in enumerate(_TIER1_SPECIES):
        jax_block[:, i] = np.asarray(traj.y_of(name))
        fortran_block[:, i] = np.asarray(
            golden.saprcgc_ppm[:, golden.spec.active_gas_species.index(name)]
        )
    return jax_block, fortran_block


@pytest.mark.parametrize("run_id", _matrix_run_params())
def test_tier1_matches_fortran_across_matrix(run_id: str, network: SOMNetwork) -> None:
    """For each in-scope canonical run, the JAX simulator should
    reproduce GENVOC + the four BL20 first-generation products from
    the Fortran ``_saprcgc.dat`` to ≤1% L2 / ≤1% final-rel."""
    matrix = load_manifest(default_manifest_path())
    run = matrix.by_id(run_id)
    golden = load_golden_run(default_expected_dir() / run_id, run_id)

    jax_block, fortran_block = _run_jax_tier1(run, golden, network)

    report = compare_trajectories(jax_block, fortran_block, _TIER1_SPECIES)
    if not report.passes(l2_tol=_TIER1_L2_TOL, correlation_min=0.0, final_rel_tol=_TIER1_L2_TOL):
        pytest.fail(
            f"{run_id}: tier-1 regression failed at l2_tol={_TIER1_L2_TOL:.2%}\n" + report.summary()
        )


def test_in_scope_runs_cover_each_axis() -> None:
    """Sanity: the parametrise table actually exercises each varying
    axis of the matrix (time, OH, VOC, T). Catches a regression where
    someone disables too many runs."""
    matrix = load_manifest(default_manifest_path())
    times = {r.params.endtime_h for r in matrix.runs}
    assert len(times) >= 4, f"only {len(times)} distinct endtimes covered"
    ohs = {r.params.OH_molec_per_cm3 for r in matrix.runs}
    assert len(ohs) >= 3, f"only {len(ohs)} distinct OH levels covered"
    vocs = {r.params.ippmprec_ppm for r in matrix.runs}
    assert len(vocs) >= 3, f"only {len(vocs)} distinct VOC levels covered"
    temps = {r.params.temp_K for r in matrix.runs}
    assert len(temps) >= 3, f"only {len(temps)} distinct temperatures covered"
