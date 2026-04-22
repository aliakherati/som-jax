"""Generate figures illustrating the S1.7 simulate wrapper.

One panel under ``docs/figures/s1.7/``:

1. ``all_species_trajectories.png`` — every species' concentration over a
   4 e-fold integration starting from only GENVOC nonzero. Species are
   coloured by carbon number to make the oxidation cascade visible:
   GENVOC (C=7) decays; first-gen products (C=7, oxygen growing)
   accumulate then decline as secondary chemistry picks up; the
   fragmentation cascade populates lower-C species over time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jax import config
from matplotlib import cm
from matplotlib.colors import Normalize

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

config.update("jax_enable_x64", True)

from som_jax import build_initial, simulate  # noqa: E402
from som_jax.mechanism import SOMNetwork  # noqa: E402

_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.7"


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    network = SOMNetwork.from_json(_JSON)

    # Choose OH + t_final so GENVOC decays by ~4 e-folds (1.0 -> 0.018).
    # This drives the cascade far enough to expose later-generation species
    # but not so far that everything has reached fragmentation equilibrium.
    bl20_idx = network.reaction_index("BL20")
    k_bl20 = float(network.k_OH[bl20_idx])
    oh = 1e-4
    t_final = 4.0 / (k_bl20 * oh)

    save_at = jnp.linspace(0.0, t_final, 200)
    y0 = build_initial(network, {"GENVOC": 1.0})
    traj = simulate(network, y0, oh=oh, t_span=(0.0, t_final), save_at=save_at, rtol=1e-8)
    y = np.asarray(traj.y)
    t = np.asarray(traj.t)

    carbons = np.asarray(network.carbon)
    norm = Normalize(vmin=1, vmax=int(carbons.max()))
    cmap = mpl.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    for i, name in enumerate(network.species_names):
        c = int(carbons[i])
        colour = cmap(norm(c))
        linewidth = 2.2 if name == "GENVOC" else 1.0
        alpha = 1.0 if name == "GENVOC" else 0.75
        ax.plot(
            t,
            np.clip(y[:, i], 1e-10, None),  # clip for log scale
            color=colour,
            linewidth=linewidth,
            alpha=alpha,
            label=name if name == "GENVOC" else None,
        )

    ax.set_yscale("log")
    ax.set_xlabel(f"time (units of 1/(k_BL20 * OH) with k_BL20 = {k_bl20:.2e}, OH = {oh:.0e})")
    ax.set_ylabel("concentration  (log scale)")
    ax.set_title(
        "All 41 species trajectories from a GENVOC-only initial state (4 e-fold integration)"
    )
    ax.set_ylim(1e-6, 2.0)
    ax.axhline(1.0 / np.e, color="gray", linewidth=0.5, linestyle=":")
    ax.text(
        t[-1] * 0.02,
        1.0 / np.e * 1.1,
        "1/e",
        fontsize=8,
        color="gray",
        va="bottom",
    )

    # Colourbar for carbon number.
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("carbon number of species")
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "all_species_trajectories.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
