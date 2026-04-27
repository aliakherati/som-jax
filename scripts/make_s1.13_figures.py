"""Render visual diagnostics for the S1.13/S1.14/S1.15 property tests.

Three-panel figure under ``docs/figures/s1.13/``. The same 24-hour
baseline trajectory used by the property tests is shown here, with one
panel per property:

1. Top: total grid carbon C(t) over time — monotonically non-increasing
   per S1.13. Annotated with the total carbon lost to off-grid
   fragmentation.
2. Middle: total grid oxygen O(t) — starts at zero with pure GENVOC,
   grows as oxidation populates O>0 grid cells, then can decrease as
   oxidized products fragment off-grid (S1.14 — non-monotonic, but
   non-negative throughout).
3. Bottom: min(y(t)) across all 41 species — sits at exactly zero at
   t=0 and approaches the diffrax atol throughout (S1.15 — solver
   slack is well below the -1e-12 ppm tolerance).

The figure is the user-facing rendering of "what does conservation
look like in this mechanism?". When the master plan's Q5 lands and
the matrix is revised, this script just re-runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

config.update("jax_enable_x64", True)

from atmos_jax_common.units import molec_cm3_to_ppm  # noqa: E402

from som_jax import build_initial, simulate  # noqa: E402
from som_jax.mechanism import SOMNetwork  # noqa: E402

_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.13"

# Same baseline run as the property tests
_INITIAL_GENVOC_PPM = 0.05
_OH_MOLEC_PER_CM3 = 1.5e6
_TEMP_K = 298.0
_PRES_PA = 101325.0
_END_TIME_MIN = 24.0 * 60.0
_N_SAVE_POINTS = 145


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    network = SOMNetwork.from_json(_GENSOMG_JSON)
    oh_ppm = float(molec_cm3_to_ppm(_OH_MOLEC_PER_CM3, _TEMP_K, _PRES_PA))
    save_at = jnp.linspace(0.0, _END_TIME_MIN, _N_SAVE_POINTS)
    y0 = build_initial(network, {"GENVOC": _INITIAL_GENVOC_PPM})
    traj = simulate(
        network,
        y0,
        oh=oh_ppm,
        t_span=(0.0, _END_TIME_MIN),
        save_at=save_at,
        rtol=1e-10,
        atol=1e-30,
    )

    carbon = np.asarray(network.carbon)
    oxygen = np.asarray(network.oxygen)
    y = np.asarray(traj.y)
    t_h = np.asarray(traj.t) / 60.0  # minutes -> hours
    total_C = (y * carbon[None, :]).sum(axis=1)
    total_O = (y * oxygen[None, :]).sum(axis=1)
    min_y_per_t = y.min(axis=1)

    fig, (ax_c, ax_o, ax_neg) = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True)

    # --- S1.13 carbon ---------------------------------------------------
    ax_c.plot(t_h, total_C, color="#d62728", linewidth=1.5)
    ax_c.axhline(total_C[0], color="gray", linewidth=0.6, linestyle=":", label="initial total C")
    ax_c.fill_between(t_h, total_C, total_C[0], alpha=0.15, color="#d62728")
    final_loss = (total_C[0] - total_C[-1]) / total_C[0]
    ax_c.set_title(
        f"S1.13 — total grid carbon (t=0: {total_C[0]:.3f} ppm·C; "
        f"24h loss to off-grid fragmentation: {final_loss:.2%})",
        fontsize=10,
    )
    ax_c.set_ylabel("Σ Cᵢ · yᵢ(t)  (ppm·C)")
    ax_c.legend(loc="lower left", fontsize=8)
    ax_c.grid(True, alpha=0.3, linewidth=0.4)

    # --- S1.14 oxygen ---------------------------------------------------
    ax_o.plot(t_h, total_O, color="#1f77b4", linewidth=1.5)
    ax_o.axhline(0.0, color="gray", linewidth=0.6, linestyle=":")
    # In this 24h baseline run the cascade hasn't reached the deep
    # off-grid sinks (e.g., GENSOMG_04_07) hard enough to make total O
    # non-monotonic, so the curve happens to grow throughout. The
    # property test asserts only non-negativity (S1.14) — strict
    # monotonicity wouldn't hold under higher-OH or longer integrations.
    diffs_o = np.diff(total_O)
    max_drop = -float(diffs_o.min()) if diffs_o.min() < 0 else 0.0
    ax_o.set_title(
        f"S1.14 — total grid oxygen (t=0: {total_O[0]:.3f} ppm·O, peak: {total_O.max():.4f} ppm·O; "
        f"max single-step drop: {max_drop:.2e} ppm·O — "
        f"non-negative throughout)",
        fontsize=10,
    )
    ax_o.set_ylabel("Σ Oᵢ · yᵢ(t)  (ppm·O)")
    ax_o.grid(True, alpha=0.3, linewidth=0.4)

    # --- S1.15 non-negativity -------------------------------------------
    # Plot min(y) per time. With float64 + atol=1e-30 we expect this to
    # sit at exactly 0 (early) or extremely small positive (later, as
    # tail species populate). Use a symlog scale so we see both regimes.
    # Replace exact zeros with a small floor for log readability.
    floor = 1e-40
    plot_min = np.where(min_y_per_t == 0, floor, min_y_per_t)
    ax_neg.plot(t_h, plot_min, color="#2ca02c", linewidth=1.5)
    ax_neg.axhline(
        -1e-12,
        color="black",
        linewidth=0.8,
        linestyle="--",
        label="non-negativity tolerance (-1e-12 ppm)",
    )
    ax_neg.axhline(floor, color="gray", linewidth=0.4, linestyle=":", label="display floor (1e-40)")
    ax_neg.set_yscale("symlog", linthresh=1e-30)
    ax_neg.set_title(
        f"S1.15 — minimum species concentration over time (overall min = {min_y_per_t.min():.3e})",
        fontsize=10,
    )
    ax_neg.set_ylabel("min y(t) over species  (ppm)")
    ax_neg.set_xlabel("time (h)")
    ax_neg.legend(loc="lower right", fontsize=8)
    ax_neg.grid(True, alpha=0.3, linewidth=0.4)

    fig.suptitle(
        "S1.13/S1.14/S1.15 — conservation properties on a 24-hour baseline run "
        "(GENVOC=0.05 ppm, OH=1.5e6 molec/cm³, T=298 K)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    _save(fig, "conservation_overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
