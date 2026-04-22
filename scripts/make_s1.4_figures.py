"""Generate figures illustrating the JAX-native SOMNetwork (S1.4).

Three panels, written under ``docs/figures/s1.4/``:

1. ``stoich_heatmap.png`` — the (n_reactions, n_species) signed stoichiometry
   matrix. Reactant columns (-1) in red, product columns (+yield) in blue.
2. ``reactant_degree.png`` — how many reactions each species drives as a
   reactant. For SOM the answer is 0 or 1 per species; this figure mostly
   serves as a "is every grid species reachable" sanity check.
3. ``product_degree.png`` — how many reactions each species appears in as a
   product. Centrally-placed grid species (e.g. GENSOMG_01_01) show up as
   products in many reactions due to fragmentation routing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# The x64 flag is required for the float64 stoich matrix to match the
# committed JSON's numeric precision. We import jax first and set it
# before any SOMNetwork allocation happens.
from jax import config  # noqa: E402

config.update("jax_enable_x64", True)

from som_jax.mechanism import SOMNetwork  # noqa: E402

_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.4"


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    network = SOMNetwork.from_json(_JSON)
    stoich = np.asarray(network.stoich)
    species_names = list(network.species_names)
    reaction_labels = list(network.reaction_labels)
    n_rxn, n_sp = stoich.shape

    # --- figure 1: stoichiometry heatmap ------------------------------
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    # TwoSlopeNorm centers 0 on white regardless of the asymmetric bounds.
    # With cmap="RdBu_r", low values (-1 reactants) render blue and high
    # values (positive yields) render red.
    vmax = max(0.001, float(np.max(stoich)))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=vmax)
    im = ax.imshow(stoich, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(n_sp))
    ax.set_xticklabels(species_names, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n_rxn))
    ax.set_yticklabels(reaction_labels, fontsize=7)
    ax.set_xlabel("species (column index)")
    ax.set_ylabel("reaction (row index)")
    ax.set_title(
        f"SOMNetwork stoichiometry  ({n_rxn} reactions x {n_sp} species) - "
        "blue = reactant (-1), red = product yield"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("branch x oxy_yield  (0 = species not in reaction)")
    _save(fig, "stoich_heatmap.png")

    # --- figure 2: reactant degree ------------------------------------
    reactant_idx = np.asarray(network.oh_reactant_idx)
    reactant_counts = np.bincount(reactant_idx, minlength=n_sp)
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    colours = ["#d62728" if species_names[i] == "GENVOC" else "#1f77b4" for i in range(n_sp)]
    ax.bar(np.arange(n_sp), reactant_counts, color=colours, edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(n_sp))
    ax.set_xticklabels(species_names, rotation=90, fontsize=7)
    ax.set_ylabel("reactions where this species is the reactant")
    ax.set_title(
        "Reactant degree per species  (every grid species should drive exactly one reaction)"
    )
    never_reactant = [species_names[i] for i in range(n_sp) if reactant_counts[i] == 0]
    if never_reactant:
        # Bottom-left so the annotation does not overlap the bars near y=1.
        ax.text(
            0.01,
            -0.38,
            "Never appears as reactant: " + ", ".join(never_reactant),
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            color="gray",
        )
    _save(fig, "reactant_degree.png")

    # --- figure 3: product degree -------------------------------------
    # A species is "a product of reaction i" if stoich[i, j] > 0.
    product_counts = (stoich > 0).sum(axis=0)
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.bar(np.arange(n_sp), product_counts, color="#2ca02c", edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(n_sp))
    ax.set_xticklabels(species_names, rotation=90, fontsize=7)
    ax.set_ylabel("reactions where this species is a product")
    ax.set_title(
        "Product degree per species  (how centrally a species sits in the oxidation graph)"
    )
    # Highlight the precursor in place: its tick label goes red and the
    # subtitle notes why its bar is missing. No arrow crossing the bars.
    genvoc_idx = species_names.index("GENVOC")
    ax.get_xticklabels()[genvoc_idx].set_color("#d62728")
    ax.text(
        0.99,
        0.97,
        "GENVOC bar absent: the precursor is never a product.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", lw=0.6),
    )
    _save(fig, "product_degree.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
