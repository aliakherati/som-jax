"""Render the S1.10 Fortran-vs-JAX regression visually.

Two-panel figure under ``docs/figures/s1.10/``:

1. ``regression_overview.png``
   - Top: per-species relative L2 bar chart, log-y, with the master-plan
     scientific-faithfulness line (1e-3) and the regression-test
     tolerance line (3e-2) drawn for context.
   - Bottom: candidate-vs-reference scatter for all 40 SOM species at
     ``t = t_final``, log-log, with the y=x diagonal and ±3% bands.

The figure is the user-facing rendering of the regression. The Fortran
reference now runs ``INTEGR2`` with rtol=1e-10 in REAL*8 (per
``som-tomas-fortran#2``); the residual gap to JAX/diffrax is dominated
by REAL*4 noise inside the auto-generated SAPRC mechanism callback and
is well under 3% across the well-resolved cascade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config
from matplotlib.patches import Patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

config.update("jax_enable_x64", True)

from atmos_jax_common.compare import relative_l2  # noqa: E402
from atmos_jax_common.goldens import load_golden_run  # noqa: E402
from atmos_jax_common.units import molec_cm3_to_ppm  # noqa: E402

from som_jax import build_initial, simulate  # noqa: E402
from som_jax.mechanism import SOMNetwork  # noqa: E402

_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "sample_run"
_RUN_NAME = "sample_for_loader_dev"
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.10"

# Match the regression-test inputs.
_INITIAL_GENVOC_PPM = 0.05
_OH_MOLEC_PER_CM3 = 1.5e6
_TEMP_K = 298.0
_PRES_PA = 101325.0


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    network = SOMNetwork.from_json(_GENSOMG_JSON)
    golden = load_golden_run(_FIXTURE_DIR, _RUN_NAME)

    # JAX simulation matching the fixture's input conditions.
    oh_ppm = float(molec_cm3_to_ppm(_OH_MOLEC_PER_CM3, _TEMP_K, _PRES_PA))
    save_at_min = jnp.asarray(np.asarray(golden.gc.time_hours) * 60.0)
    t_span = (float(save_at_min[0]), float(save_at_min[-1]))
    y0 = build_initial(network, {"GENVOC": _INITIAL_GENVOC_PPM})
    traj = simulate(
        network,
        y0,
        oh=oh_ppm,
        t_span=t_span,
        save_at=save_at_min,
        rtol=1e-10,
        atol=1e-40,
    )

    # Build matched (n_t, n_som) blocks.
    som_names = list(golden.spec.som_species)
    n_t = golden.gc.time_hours.size
    n_som = len(som_names)

    jax_block = np.zeros((n_t, n_som), dtype=np.float64)
    fortran_block = np.zeros_like(jax_block)
    for i, name in enumerate(som_names):
        jax_block[:, i] = np.asarray(traj.y_of(name))
        fortran_block[:, i] = np.asarray(
            golden.saprcgc_ppm[:, golden.spec.active_gas_species.index(name)]
        )

    l2 = np.asarray(relative_l2(jax_block, fortran_block))
    final_jax = jax_block[-1]
    final_fortran = fortran_block[-1]

    # Match the regression test: only species with peak magnitude above the
    # 1e-10 ppm floor are tested with relative-error tolerances. Below the
    # floor the Fortran integrator's atol and REAL*4 noise inside DIFUN
    # dominate, so a relative-error metric is meaningless.
    magnitude_floor_ppm = 1e-10
    peak = np.maximum(np.abs(jax_block).max(axis=0), np.abs(fortran_block).max(axis=0))
    above_floor = peak > magnitude_floor_ppm
    n_above = int(above_floor.sum())
    n_below = n_som - n_above
    l2_above = l2[above_floor]

    fig, (ax_l2, ax_scatter) = plt.subplots(
        2, 1, figsize=(11.0, 8.5), gridspec_kw={"height_ratios": [1.1, 1]}
    )

    # --- Top: per-species relative L2 ----------------------------------
    # Bars are coloured by floor membership (light gray = below floor, not
    # part of the test) and by tolerance pass (red = above floor + above 3%).
    x = np.arange(n_som)
    bar_colours = []
    for i, v in enumerate(l2):
        if not above_floor[i]:
            bar_colours.append("#cccccc")  # below floor — excluded from test
        elif v > 0.03:
            bar_colours.append("#d62728")  # above floor + above tolerance
        else:
            bar_colours.append("#1f77b4")  # passes
    ax_l2.bar(x, np.maximum(l2, 1e-12), color=bar_colours, edgecolor="black", linewidth=0.3)
    ax_l2.set_yscale("log")
    ax_l2.set_xticks(x)
    ax_l2.set_xticklabels(som_names, rotation=90, fontsize=6)
    ax_l2.set_ylabel("relative L2 (log scale)")
    ax_l2.axhline(3e-2, color="black", linewidth=0.8, linestyle="--", label="regression tol (3%)")
    ax_l2.axhline(
        1e-3,
        color="gray",
        linewidth=0.6,
        linestyle=":",
        label="master-plan faithfulness target (0.1%)",
    )
    # Custom legend entries describing the bar colour scheme.
    handles, _ = ax_l2.get_legend_handles_labels()
    handles += [
        Patch(facecolor="#1f77b4", edgecolor="black", label="passes (above floor)"),
        Patch(facecolor="#d62728", edgecolor="black", label="fails (above floor)"),
        Patch(
            facecolor="#cccccc",
            edgecolor="black",
            label=f"below 1e-10 ppm floor (excluded; {n_below}/{n_som})",
        ),
    ]
    ax_l2.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)
    ax_l2.set_title(
        f"S1.10 regression: per-species relative L2 (JAX vs Fortran ``_saprcgc.dat``) "
        f"— above-floor max = {l2_above.max():.2%}, "
        f"median = {np.median(l2_above):.2%} "
        f"({n_above}/{n_som} species)"
    )
    ax_l2.set_ylim(1e-7, 2.0)
    ax_l2.grid(True, which="both", alpha=0.3, linewidth=0.4, axis="y")

    # --- Bottom: candidate-vs-reference scatter at t_final --------------
    floor = 1e-15
    f_x = np.maximum(final_fortran, floor)
    j_y = np.maximum(final_jax, floor)
    sizes = 25 + 50 * np.log10(f_x / floor)  # bigger dots for higher-magnitude species
    ax_scatter.scatter(
        f_x,
        j_y,
        s=sizes,
        c=l2,
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.3,
        norm=plt.matplotlib.colors.LogNorm(vmin=max(1e-4, l2.min() + 1e-12), vmax=l2.max()),
    )
    lo = max(floor, min(f_x.min(), j_y.min()))
    hi = max(f_x.max(), j_y.max()) * 2
    ax_scatter.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    # ±3% bands
    ax_scatter.plot([lo, hi], [lo * 0.97, hi * 0.97], "gray", linewidth=0.5, linestyle=":")
    ax_scatter.plot(
        [lo, hi], [lo * 1.03, hi * 1.03], "gray", linewidth=0.5, linestyle=":", label="±3% band"
    )
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_xlabel("Fortran final concentration  (ppm, log)")
    ax_scatter.set_ylabel("JAX final concentration  (ppm, log)")
    t_final_h = float(golden.gc.time_hours[-1])
    ax_scatter.set_title(f"Final-time concentrations (t = {t_final_h:.4f} h, all 40 SOM species)")
    ax_scatter.legend(loc="upper left", fontsize=8)
    ax_scatter.grid(True, which="both", alpha=0.3, linewidth=0.4)

    fig.tight_layout()
    _save(fig, "regression_overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
