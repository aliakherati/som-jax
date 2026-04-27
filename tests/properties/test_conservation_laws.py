"""Property tests for the SOM gas-phase chemistry (S1.13/S1.14/S1.15).

The Cappa-Wilson SOM placed on the (C, O) grid has explicit mass-loss
pathways representing fragmentation products that fall off the tracked
grid (e.g., CO, CO₂, formic acid — too small to occupy a grid cell).
That makes carbon non-conserving in the strict sense: 5 reactions
(``S3.1``, ``S4.1``, ``S9.1``, ``S10.1``, ``S17.1``) are 100% off-grid
sinks per the parsed Fortran mechanism, and several others lose
fractional mass (see the per-reaction balance table in the
``S1.13``-figure script). The master plan's original property
specification (carbon ≤ 1e-6 conservation, oxygen monotonic
non-decreasing) is therefore relaxed here to what is actually
observable in the parsed mechanism:

- **S1.13**: total carbon is monotonically non-increasing — no reaction
  produces grid species with more carbon than the parent. This catches
  any phantom-carbon-creation bug (yield > 1 routing more C than the
  parent had).
- **S1.14**: total oxygen is non-negative throughout the trajectory
  and starts at zero with a pure GENVOC initial condition. It can
  decrease at later times when oxidized products leave the grid via
  fragmentation, so we *don't* assert monotonicity.
- **S1.15**: every species concentration stays at or above ``-1e-12``
  ppm — small negative excursions are allowed as solver slack but no
  meaningful negative values should appear.

A baseline 4-hour run (GENVOC = 0.05 ppm, OH = 1.5×10⁶ molec/cm³,
T = 298 K) is shared across all three tests.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from atmos_jax_common.units import molec_cm3_to_ppm

from som_jax import build_initial, simulate
from som_jax.mechanism import SOMNetwork

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"

# Run conditions match the canonical-run matrix's ``very_long`` entry —
# a 24-hour integration at baseline OH and GENVOC. We want a long
# enough integration that the cascade is meaningfully populated and
# the off-grid mass-loss pathways have visible effect; 4 h at
# OH = 1.5×10⁶ molec/cm³ only consumes ~11% of GENVOC, which doesn't
# stress the property invariants enough to be useful.
_INITIAL_GENVOC_PPM = 0.05
_OH_MOLEC_PER_CM3 = 1.5e6
_TEMP_K = 298.0
_PRES_PA = 101325.0
_END_TIME_MIN = 24.0 * 60.0  # 24 hours
_N_SAVE_POINTS = 145  # every ~10 minutes — fine enough for monotonicity checks


@pytest.fixture(scope="module")
def network() -> SOMNetwork:
    return SOMNetwork.from_json(_GENSOMG_JSON)


@pytest.fixture(scope="module")
def trajectory(network: SOMNetwork):
    """Shared 4h baseline trajectory used by all three property tests."""
    oh_ppm = float(molec_cm3_to_ppm(_OH_MOLEC_PER_CM3, _TEMP_K, _PRES_PA))
    save_at_min = jnp.linspace(0.0, _END_TIME_MIN, _N_SAVE_POINTS)
    y0 = build_initial(network, {"GENVOC": _INITIAL_GENVOC_PPM})
    return simulate(
        network,
        y0,
        oh=oh_ppm,
        t_span=(0.0, _END_TIME_MIN),
        save_at=save_at_min,
        rtol=1e-10,
        atol=1e-30,
    )


# --- S1.13 ---------------------------------------------------------------


def test_s1_13_total_carbon_is_monotonically_non_increasing(
    network: SOMNetwork, trajectory
) -> None:
    """No reaction in the GENSOMG mechanism creates carbon. Total
    carbon on the grid can only decrease (via fragmentation pathways
    that route mass off-grid) or stay flat (when all reactions
    currently active conserve carbon)."""
    carbon = np.asarray(network.carbon)  # (n_species,)
    y = np.asarray(trajectory.y)  # (n_t, n_species)
    total_C = (y * carbon[None, :]).sum(axis=1)  # (n_t,)

    # Initial total carbon is exactly the precursor's carbon-weighted ppm:
    # GENVOC has C=7, started at 0.05 ppm → 0.35 ppm·C
    assert total_C[0] == pytest.approx(0.05 * 7, rel=1e-12)

    # Solver slack: allow a tiny positive blip of size ~1e-12 ppm·C, well
    # below any meaningful chemistry signal.
    diffs = np.diff(total_C)
    assert diffs.max() < 1e-12, (
        f"total carbon increased at some step (max diff {diffs.max():.3e}); "
        f"this would indicate a phantom-carbon-creation bug in the mechanism."
    )

    # Sanity-check that the mechanism's documented off-grid sinks are
    # actually firing. Total carbon should drop by *something*
    # measurable over a 24h run with this OH.
    final_loss_frac = (total_C[0] - total_C[-1]) / total_C[0]
    assert final_loss_frac > 1e-4, (
        f"carbon loss over 24h is only {final_loss_frac:.3%}; the "
        f"off-grid mass-loss pathways should fire visibly. Either OH "
        f"is too low or the mechanism's fragmentation routes aren't "
        f"reachable from a GENVOC-only initial condition."
    )


# --- S1.14 ---------------------------------------------------------------


def test_s1_14_total_oxygen_is_non_negative_and_zero_initially(
    network: SOMNetwork, trajectory
) -> None:
    """Pure GENVOC has zero oxygen; total grid oxygen starts at 0 and
    is non-negative throughout. Strict monotonicity does NOT hold
    because oxidized products (carrying O atoms) can leave the grid
    via fragmentation; that's documented mechanism behaviour."""
    oxygen = np.asarray(network.oxygen)
    y = np.asarray(trajectory.y)
    total_O = (y * oxygen[None, :]).sum(axis=1)

    # Initial: pure GENVOC has zero oxygen → exactly 0.
    assert total_O[0] == pytest.approx(0.0, abs=1e-30)

    # Non-negativity throughout (allow solver slack at the same magnitude
    # we use for non-negativity of individual species in S1.15).
    assert total_O.min() >= -1e-12, (
        f"total oxygen went negative (min {total_O.min():.3e}); this "
        f"would indicate a wrong-sign coupling in the mechanism."
    )

    # Sanity: oxygen grows (oxidation is happening); peak should be
    # well above zero. With a 24h run the peak is ~10⁻² ppm·O range.
    assert total_O.max() > 1e-4, (
        f"total oxygen peaked at only {total_O.max():.3e} ppm·O; "
        f"oxidation is barely happening — check the OH input or rate "
        f"constants."
    )


# --- S1.15 ---------------------------------------------------------------


def test_s1_15_concentrations_stay_non_negative(trajectory) -> None:
    """No species concentration should fall below -1e-12 ppm. Small
    negative excursions are allowed as solver slack (the Kvaerno5
    integrator can briefly overshoot near zero), but anything more
    negative indicates a real overshoot that masks a chemistry bug."""
    y = np.asarray(trajectory.y)
    min_y = float(y.min())
    assert min_y >= -1e-12, (
        f"species concentration went to {min_y:.3e} ppm — beyond the "
        f"-1e-12 solver-slack tolerance. Either the integrator's atol "
        f"is too loose or a stoichiometry sign is wrong."
    )


# --- supplementary: oxidation actually progresses ------------------------


def test_supplementary_oxidation_progresses(network: SOMNetwork, trajectory) -> None:
    """Cross-check: the *mean oxidation state* (O/C ratio of grid
    species, weighted by concentration) should grow over time. This
    captures "oxidation only goes forward" without requiring the
    monotonicity-of-total-O property that off-grid fragmentation
    breaks."""
    carbon = np.asarray(network.carbon)
    oxygen = np.asarray(network.oxygen)
    y = np.asarray(trajectory.y)

    total_C = (y * carbon[None, :]).sum(axis=1)
    total_O = (y * oxygen[None, :]).sum(axis=1)
    # O/C ratio at each save point. At t=0 carbon is all in GENVOC (O=0),
    # so the ratio is 0 — same denominator at all times so the ratio is
    # well defined.
    oc_ratio = np.where(total_C > 0, total_O / total_C, 0.0)

    # Initial mean oxidation state is zero (pure GENVOC).
    assert oc_ratio[0] == pytest.approx(0.0, abs=1e-30)

    # Final mean oxidation state should be well above zero (cascade has
    # populated oxygenated grid cells). For a 24h run at OH = 1.5×10⁶
    # the cascade reaches ~0.3-0.5 mean O/C.
    assert oc_ratio[-1] > 0.1, (
        f"final O/C ratio is {oc_ratio[-1]:.3f}; oxidation should "
        f"have moved the centre of mass off C=7,O=0 by 24h."
    )
