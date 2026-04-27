"""Render visual diagnostics for the S1.11 canonical-matrix regression.

Two figures under ``docs/figures/s1.11/``:

1. ``tier1_overview.png`` — for each in-scope run, plot the GENVOC
   trajectory (the cleanest tier-1 species) from JAX and Fortran
   overlaid. Visual confirmation that JAX matches Fortran across
   the matrix at the precursor level.

2. ``cascade_nonlinearity.png`` — illustrates the Fortran nonlinearity
   that motivates restricting the regression to tier-1 species. Plots
   the high_voc / long_baseline ratio for each species at t_final;
   linear chemistry would give exactly 10×; Fortran shows ~2.5× on
   deep cascade species while JAX correctly gives 10× throughout.
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

from atmos_jax_common.canonical_runs import (  # noqa: E402
    default_expected_dir,
    default_manifest_path,
    load_manifest,
)
from atmos_jax_common.compare import relative_l2  # noqa: E402
from atmos_jax_common.goldens import load_golden_run  # noqa: E402
from atmos_jax_common.units import molec_cm3_to_ppm  # noqa: E402

from som_jax import build_initial, simulate  # noqa: E402
from som_jax.mechanism import SOMNetwork  # noqa: E402

_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.11"

_IN_SCOPE = (
    "short_baseline",
    "medium_baseline",
    "long_baseline",
    "very_long",
    "low_oh",
    "high_oh",
    "low_voc",
    "high_voc",
)

_FAMILY_COLOUR = {
    "short_baseline": "#1f77b4",
    "medium_baseline": "#1f77b4",
    "long_baseline": "#1f77b4",
    "very_long": "#1f77b4",
    "low_oh": "#d62728",
    "high_oh": "#d62728",
    "low_voc": "#2ca02c",
    "high_voc": "#2ca02c",
}


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def _run_jax(network: SOMNetwork, run, save_at_min):
    oh_ppm = float(molec_cm3_to_ppm(run.params.OH_molec_per_cm3, run.params.temp_K, 101325.0))
    y0 = build_initial(network, {"GENVOC": run.params.ippmprec_ppm})
    return simulate(
        network,
        y0,
        oh=oh_ppm,
        t_span=(float(save_at_min[0]), float(save_at_min[-1])),
        save_at=save_at_min,
        rtol=1e-10,
        atol=1e-30,
    )


def make_tier1_overview(network: SOMNetwork, matrix) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.0), sharey=False)
    axes_flat = axes.flatten()

    for ax, run_id in zip(axes_flat, _IN_SCOPE, strict=False):
        run = matrix.by_id(run_id)
        golden = load_golden_run(default_expected_dir() / run_id, run_id)
        save_at_min = jnp.asarray(np.asarray(golden.gc.time_hours) * 60.0)
        traj = _run_jax(network, run, save_at_min)

        t_h = np.asarray(golden.gc.time_hours)
        genvoc_idx = golden.spec.active_gas_species.index("GENVOC")
        ax.plot(
            t_h,
            np.asarray(golden.saprcgc_ppm[:, genvoc_idx]),
            color="black",
            linewidth=1.6,
            label="Fortran",
        )
        ax.plot(
            t_h,
            np.asarray(traj.y_of("GENVOC")),
            color=_FAMILY_COLOUR[run_id],
            linestyle="--",
            linewidth=1.4,
            label="JAX",
        )

        # Compute per-species relative L2 on tier-1 species
        tier1 = ("GENVOC", "GENSOMG_07_01", "GENSOMG_07_02")
        n_t = t_h.size
        jax_block = np.zeros((n_t, len(tier1)))
        ftn_block = np.zeros_like(jax_block)
        for j, name in enumerate(tier1):
            jax_block[:, j] = np.asarray(traj.y_of(name))
            ftn_block[:, j] = np.asarray(
                golden.saprcgc_ppm[:, golden.spec.active_gas_species.index(name)]
            )
        l2 = np.asarray(relative_l2(jax_block, ftn_block))

        ax.set_title(
            f"{run_id}\nmax tier-1 L2 = {l2.max():.2e}",
            fontsize=9,
        )
        ax.set_xlabel("t (h)", fontsize=8)
        ax.set_ylabel("[GENVOC] (ppm)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(axis="both", labelsize=7)

    fig.suptitle(
        "S1.11 — GENVOC trajectory: JAX vs Fortran across the canonical matrix "
        "(8 in-scope runs; cold/hot skipped pending T-dependent rates)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "tier1_overview.png")


def make_cascade_nonlinearity(network: SOMNetwork, matrix) -> None:
    """Show why the test is restricted to tier-1: Fortran's cascade
    is non-linear in precursor magnitude (a 10× GENVOC change should
    give 10× cascade for linear chemistry); Fortran gives ~2.5×."""
    long_run = matrix.by_id("long_baseline")
    high_run = matrix.by_id("high_voc")
    g_long = load_golden_run(default_expected_dir() / "long_baseline", "long_baseline")
    g_high = load_golden_run(default_expected_dir() / "high_voc", "high_voc")

    save_long = jnp.asarray(np.asarray(g_long.gc.time_hours) * 60.0)
    save_high = jnp.asarray(np.asarray(g_high.gc.time_hours) * 60.0)
    j_long = _run_jax(network, long_run, save_long)
    j_high = _run_jax(network, high_run, save_high)

    species = list(g_long.spec.som_species)
    fortran_ratio = np.zeros(len(species))
    jax_ratio = np.zeros(len(species))
    long_max = np.zeros(len(species))
    for i, name in enumerate(species):
        l_idx = g_long.spec.active_gas_species.index(name)
        h_idx = g_high.spec.active_gas_species.index(name)
        l_v = float(g_long.saprcgc_ppm[-1, l_idx])
        h_v = float(g_high.saprcgc_ppm[-1, h_idx])
        long_max[i] = l_v
        fortran_ratio[i] = h_v / l_v if l_v > 1e-15 else np.nan
        l_jax = float(j_long.y_of(name)[-1])
        h_jax = float(j_high.y_of(name)[-1])
        jax_ratio[i] = h_jax / l_jax if l_jax > 1e-15 else np.nan

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    x = np.arange(len(species))
    ax.bar(
        x - 0.2,
        fortran_ratio,
        width=0.4,
        color="black",
        edgecolor="white",
        label="Fortran (high_voc / long_baseline)",
    )
    ax.bar(
        x + 0.2,
        jax_ratio,
        width=0.4,
        color="#2ca02c",
        edgecolor="white",
        label="JAX (high_voc / long_baseline)",
    )
    ax.axhline(
        10.0, color="red", linewidth=1.0, linestyle="--", label="linear-chemistry expected (10x)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=90, fontsize=6)
    ax.set_ylabel("ratio of final concentration (high_voc / long_baseline)")
    ax.set_title(
        "S1.11 motivation - Fortran is non-linear in precursor magnitude. "
        "JAX (REAL*8) gives the correct linear 10x across the cascade; "
        "Fortran (REAL*4 DIFUN) drops to ~2.5x on deep tail."
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax.set_ylim(0, 13)

    fig.tight_layout()
    _save(fig, "cascade_nonlinearity.png")


def main() -> int:
    network = SOMNetwork.from_json(_GENSOMG_JSON)
    matrix = load_manifest(default_manifest_path())
    make_tier1_overview(network, matrix)
    make_cascade_nonlinearity(network, matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
