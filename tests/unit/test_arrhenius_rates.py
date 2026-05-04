"""Unit tests for the Arrhenius temperature dependence pipeline.

Covers four pieces wired up for T-dependent rates:

- Parser captures the optional ``(A, Ea, B)`` triplet from ``.doc``.
- JSON round-trip preserves the triplet on reactions that have one,
  and leaves the field absent on reactions that don't.
- ``SOMNetwork.k_OH_at(T)`` evaluates the SAPRC formula
  ``k(T) = A·exp(-Ea/(R·T))·(T/T_ref)^B`` per reaction.
- ``simulate(temperature_K=...)`` propagates the T-dependent rates
  end-to-end and is differentiable through ``jax.grad``.
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


@pytest.fixture(scope="module")
def network() -> SOMNetwork:
    return SOMNetwork.from_json(_GENSOMG_JSON)


def test_bl20_carries_arrhenius_triplet(network: SOMNetwork) -> None:
    """BL20 (the precursor + OH) is the only reaction with an
    explicit Arrhenius triplet in the .doc; the parser should
    capture it. Reference values from saprc14_rev1.doc:
    A = 8264, Ea = 0, B = -1.000."""
    bl20_idx = network.reaction_index("BL20")
    assert float(network.arrhenius_a[bl20_idx]) == pytest.approx(8264.0, rel=1e-4)
    assert float(network.arrhenius_ea_kcal_per_mol[bl20_idx]) == pytest.approx(0.0)
    assert float(network.arrhenius_b[bl20_idx]) == pytest.approx(-1.0)


def test_cascade_reactions_default_to_t_independent(network: SOMNetwork) -> None:
    """For SOM cascade reactions (S1.1..S38.1) the .doc lists no
    Arrhenius triplet, so the network stores ``A = k_OH``, ``Ea = 0``,
    ``B = 0`` — making ``k_OH_at(T)`` T-independent and equal to the
    listed rate at any T."""
    s11 = network.reaction_index("S1.1")
    assert float(network.arrhenius_b[s11]) == pytest.approx(0.0)
    assert float(network.arrhenius_ea_kcal_per_mol[s11]) == pytest.approx(0.0)
    assert float(network.arrhenius_a[s11]) == pytest.approx(float(network.k_OH[s11]))


def test_k_oh_at_298_matches_listed_rates(network: SOMNetwork) -> None:
    """At T=298K the .doc-listed rate (stored as ``k_OH``) and the
    Arrhenius-evaluated rate must agree to within parser rounding —
    BL20 has B=-1 with Tref=300, so the conversion picks up a tiny
    298/300 factor that the .doc rounded to 4 sig figs."""
    k298 = network.k_OH_at(298.0)
    bl20_idx = network.reaction_index("BL20")
    # BL20 is rounded to 4 sig figs in the .doc; allow ULP for the
    # (298/300)^(-1) recomputation.
    assert float(k298[bl20_idx]) == pytest.approx(float(network.k_OH[bl20_idx]), rel=1e-4)
    # All cascade rates are T-independent so equality is exact.
    for label in ("S1.1", "S5.1", "S38.1"):
        i = network.reaction_index(label)
        assert float(k298[i]) == pytest.approx(float(network.k_OH[i]), rel=1e-12)


def test_bl20_at_273_is_about_9pct_higher(network: SOMNetwork) -> None:
    """BL20 has B = -1, so k(T) ∝ 1/T. At T=273 K the rate is
    300/273 ≈ 1.099× the value at T=300; relative to the 298-K
    listed value the bump is ~9%."""
    bl20_idx = network.reaction_index("BL20")
    k298 = float(network.k_OH_at(298.0)[bl20_idx])
    k273 = float(network.k_OH_at(273.0)[bl20_idx])
    ratio = k273 / k298
    expected = 298.0 / 273.0  # k(T)/k(298) = (298/T) for B = -1
    assert ratio == pytest.approx(expected, rel=1e-4)
    assert ratio > 1.08  # at least 8% higher at chamber-cold


def test_bl20_at_323_is_about_7pct_lower(network: SOMNetwork) -> None:
    """Companion to the cold case: at T=323 K the BL20 rate is
    298/323 ≈ 0.923× the 298-K value (about 7-8% lower)."""
    bl20_idx = network.reaction_index("BL20")
    k298 = float(network.k_OH_at(298.0)[bl20_idx])
    k323 = float(network.k_OH_at(323.0)[bl20_idx])
    ratio = k323 / k298
    expected = 298.0 / 323.0
    assert ratio == pytest.approx(expected, rel=1e-4)
    assert 0.91 < ratio < 0.93


def test_cascade_rates_invariant_under_temperature(network: SOMNetwork) -> None:
    """SOM cascade rates have no Arrhenius parameters in the .doc,
    so ``k_OH_at(T)`` should return the same value at any T."""
    s11 = network.reaction_index("S1.1")
    s38 = network.reaction_index("S38.1")
    for T in (250.0, 273.0, 298.0, 323.0, 350.0):
        kT = network.k_OH_at(T)
        assert float(kT[s11]) == pytest.approx(float(network.k_OH[s11]), rel=1e-12)
        assert float(kT[s38]) == pytest.approx(float(network.k_OH[s38]), rel=1e-12)


def test_simulate_with_temperature_reproduces_default_at_298(
    network: SOMNetwork,
) -> None:
    """``simulate(temperature_K=298)`` must produce the same trajectory
    as the default-temperature call, modulo a tiny BL20 rate
    rounding (the .doc lists k(298) at 4 sig figs but the Arrhenius
    parametrisation reproduces it via ``A * (298/300)^-1``)."""
    y0 = build_initial(network, {"GENVOC": 0.05})
    save_at = jnp.linspace(0.0, 60.0, 7)

    traj_default = simulate(
        network,
        y0,
        oh=6.090601e-08,
        t_span=(0.0, 60.0),
        save_at=save_at,
        rtol=1e-10,
        atol=1e-30,
    )
    traj_298 = simulate(
        network,
        y0,
        oh=6.090601e-08,
        t_span=(0.0, 60.0),
        save_at=save_at,
        temperature_K=298.0,
        rtol=1e-10,
        atol=1e-30,
    )
    # Same .doc rounding ULP that test_k_oh_at_298_matches_listed_rates
    # already covers.
    np.testing.assert_allclose(
        np.asarray(traj_default.y_of("GENVOC")),
        np.asarray(traj_298.y_of("GENVOC")),
        rtol=1e-4,
    )


def test_simulate_at_273_produces_more_decay(
    network: SOMNetwork,
) -> None:
    """At T=273 K BL20 is 9% faster, so over a 60-min run starting
    from pure GENVOC the precursor concentration at the final time
    should be measurably lower than at 298 K (more decay)."""
    y0 = build_initial(network, {"GENVOC": 0.05})
    save_at = jnp.linspace(0.0, 60.0, 7)

    traj_298 = simulate(
        network,
        y0,
        oh=6.090601e-08,
        t_span=(0.0, 60.0),
        save_at=save_at,
        temperature_K=298.0,
        rtol=1e-10,
        atol=1e-30,
    )
    traj_273 = simulate(
        network,
        y0,
        oh=6.090601e-08,
        t_span=(0.0, 60.0),
        save_at=save_at,
        temperature_K=273.0,
        rtol=1e-10,
        atol=1e-30,
    )
    g_298 = float(traj_298.y_of("GENVOC")[-1])
    g_273 = float(traj_273.y_of("GENVOC")[-1])
    # Extra decay at cold should be roughly proportional to the rate
    # bump. With ~9% faster rate over 60 min, the precursor sees
    # slightly more loss; the difference is small in absolute terms
    # because total decay over 1 h is only a few percent.
    assert g_273 < g_298
    assert (g_298 - g_273) / g_298 > 1e-4


def test_simulate_is_differentiable_through_temperature(
    network: SOMNetwork,
) -> None:
    """The whole point: ``temperature_K`` is differentiable. The
    derivative of [GENVOC](t_final) w.r.t. T should be positive for
    BL20 (k ∝ 1/T means cooler ⇒ faster ⇒ less GENVOC remaining;
    therefore d[GENVOC]/dT > 0 — warmer keeps more precursor around).
    """
    y0 = build_initial(network, {"GENVOC": 0.05})
    save_at = jnp.linspace(0.0, 60.0, 7)

    def genvoc_final_at_T(T):
        return simulate(
            network,
            y0,
            oh=6.090601e-08,
            t_span=(0.0, 60.0),
            save_at=save_at,
            temperature_K=T,
            rtol=1e-8,
            atol=1e-15,
        ).y_of("GENVOC")[-1]

    deriv = float(jax.grad(genvoc_final_at_T)(298.0))
    assert jnp.isfinite(deriv)
    assert deriv > 0, f"expected d[GENVOC]/dT > 0 (warmer ⇒ slower decay); got {deriv:.3e}"

    # Cross-check vs central difference within 1%.
    eps = 0.5
    fp = float(genvoc_final_at_T(298.0 + eps))
    fm = float(genvoc_final_at_T(298.0 - eps))
    deriv_num = (fp - fm) / (2.0 * eps)
    rel_err = abs(deriv - deriv_num) / abs(deriv_num)
    assert rel_err < 0.01, (
        f"jax.grad disagrees with central-diff: auto={deriv:.6e}, "
        f"num={deriv_num:.6e}, rel_err={rel_err:.3e}"
    )
