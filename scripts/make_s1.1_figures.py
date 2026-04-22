"""Generate figures illustrating the parsed S1.1 mechanism.

Three panels, written as three separate PNGs under ``docs/figures/s1.1/``:

1. ``cover_grid.png`` — which (carbon, oxygen) cells the mechanism actually
   populates. Makes off-by-one or missing-species bugs visible at a glance.
2. ``yield_sum_per_reaction.png`` — sum of effective yields
   ``branch * oxy_yield`` per reaction, coloured by regime. Confirms the
   sum-to-one vs fragmentation vs zero-yield story from the CHANGELOG.
3. ``rate_constants.png`` — distribution of ``k_OH`` (log-scale histogram)
   across the 39 reactions.

Dev-time script; re-run after any mechanism JSON regeneration. Imports
matplotlib, which ships only under the ``dev`` extras.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from som_jax.mechanism import mechanism_from_json  # noqa: E402

_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.1"


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    mech = mechanism_from_json(_JSON)

    # --- figure 1: (C, O) grid coverage -------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    # Full grid footprint, including unpopulated cells.
    ax.set_xticks(np.arange(0, mech.grid_c_max + 1))
    ax.set_yticks(np.arange(0, mech.grid_o_max + 1))
    ax.set_xticks(np.arange(-0.5, mech.grid_c_max + 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mech.grid_o_max + 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5, zorder=0)
    ax.tick_params(which="minor", length=0)
    ax.set_xlim(-0.5, mech.grid_c_max + 0.5)
    ax.set_ylim(-0.5, mech.grid_o_max + 0.5)
    for sp in mech.species:
        colour = "#d62728" if sp.is_precursor else "#1f77b4"
        marker = "s" if sp.is_precursor else "o"
        ax.scatter(sp.carbon, sp.oxygen, s=120, c=colour, marker=marker, zorder=2)
    ax.annotate(
        "GENVOC\n(precursor)",
        xy=(7, 0),
        xytext=(5.5, 0.6),
        fontsize=9,
        ha="center",
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
    )
    ax.set_xlabel("Carbon atoms")
    ax.set_ylabel("Oxygen atoms")
    ax.set_title(
        f"GENSOMG (C, O) grid — {len(mech.species)} species present "
        f"({sum(1 for s in mech.species if not s.is_precursor)} products + "
        f"{sum(1 for s in mech.species if s.is_precursor)} precursor)"
    )
    ax.set_aspect("equal")
    _save(fig, "cover_grid.png")

    # --- figure 2: sum of yields per reaction -------------------------
    sums = np.array([r.total_yield for r in mech.reactions], dtype=float)
    labels = [r.label for r in mech.reactions]
    x = np.arange(len(sums))

    def _classify(s: float) -> str:
        if np.isclose(s, 0.0, atol=1e-6):
            return "mass loss (sum ~ 0)"
        if s > 1.05:
            return "fragmentation (sum > 1)"
        return "on-grid (sum ~ 1)"

    colours = {
        "on-grid (sum ~ 1)": "#2ca02c",
        "fragmentation (sum > 1)": "#ff7f0e",
        "mass loss (sum ~ 0)": "#7f7f7f",
    }
    classes = [_classify(s) for s in sums]
    bar_colours = [colours[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.bar(x, sums, color=bar_colours, edgecolor="black", linewidth=0.3)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")

    # Zero-yield reactions have height 0 so no bar is visible. Mark them
    # with a gray hollow circle on the x-axis so they still catch the eye.
    zero_mask = np.isclose(sums, 0.0, atol=1e-6)
    ax.scatter(
        x[zero_mask],
        np.zeros(int(zero_mask.sum())),
        s=40,
        facecolors="none",
        edgecolors=colours["mass loss (sum ~ 0)"],
        linewidths=1.5,
        zorder=3,
        label=None,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("sum of branch x oxy_yield over products")
    ax.set_title(
        "Effective product-yield sum per reaction - BL20 = GENVOC+OH, S1.1..S38.1 = GENSOMG+OH"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in colours]
    ax.legend(handles, list(colours.keys()), loc="upper right", fontsize=9)
    ax.set_ylim(-0.05, max(2.2, sums.max() * 1.1))
    _save(fig, "yield_sum_per_reaction.png")

    # --- figure 3: rate-constant distribution -------------------------
    rates = np.array([r.rate_cm3_per_mol_per_s for r in mech.reactions], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bins = np.logspace(np.log10(rates.min()) - 0.1, np.log10(rates.max()) + 0.1, 20)
    ax.hist(rates, bins=bins, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("k_OH at 300 K  (cm^3 mol^-1 s^-1, Fortran convention)")
    ax.set_ylabel("number of reactions")
    ax.set_title(
        f"Rate-constant distribution across {len(mech.reactions)} SOM reactions "
        f"(min {rates.min():.2e}, max {rates.max():.2e})"
    )
    # Annotate BL20 specifically - the entry-point precursor reaction.
    bl20_rate = next(r.rate_cm3_per_mol_per_s for r in mech.reactions if r.label == "BL20")
    ax.axvline(bl20_rate, color="#d62728", linewidth=1.2, linestyle="--")
    # Place the label in the top-right corner rather than on the line so it
    # does not overlap the histogram bars.
    ax.text(
        0.98,
        0.95,
        f"BL20 (GENVOC+OH):\n{bl20_rate:.2e} cm^3 mol^-1 s^-1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", lw=0.6),
    )
    _save(fig, "rate_constants.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
