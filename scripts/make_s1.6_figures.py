"""Generate figures illustrating the S1.6 SOM right-hand side.

Three panels written under ``docs/figures/s1.6/``:

1. ``genvoc_only_dydt.png`` — dy/dt at t=0 when only GENVOC is nonzero.
   Demonstrates the first-generation activation pattern: exactly five
   species have nonzero derivatives (GENVOC decaying, four GENSOMG_07_*
   products accumulating at the BL20 yields).
2. ``uniform_state_dydt.png`` — dy/dt at a uniform concentration. Shows
   the steady-state-ish structure: grid species in the middle of the
   oxidation cascade are net produced while the end-points are net
   consumed. A useful sanity check on coupling direction.
3. ``jacobian_structure.png`` — ``df/dy`` at a uniform state, as a
   signed heatmap. Exposes the reaction-graph's adjacency structure:
   row i, column j is nonzero iff reaction *i*'s reactant is species *j*
   (self-reactant diagonal) or species *i* appears as a product of a
   reaction whose reactant is *j*.

Dev-time script. Requires the ``dev`` extras (matplotlib).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Float64 needed for the scientific derivative magnitudes to round-trip
# cleanly. Matches the test-suite convention (tests/conftest.py).
from jax import config  # noqa: E402

config.update("jax_enable_x64", True)

from som_jax import som_rhs  # noqa: E402
from som_jax.mechanism import SOMNetwork  # noqa: E402

_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.6"


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    network = SOMNetwork.from_json(_JSON)
    species_names = list(network.species_names)
    n_sp = network.n_species

    # --- figure 1: GENVOC-only initial dy/dt --------------------------
    genvoc_idx = species_names.index("GENVOC")
    y0 = jnp.zeros(n_sp, dtype=jnp.float64).at[genvoc_idx].set(1.0)
    dydt = np.asarray(som_rhs(y0, 1.0, network))

    # Colours: reactant (GENVOC) in red, first-gen products in blue,
    # inert (zero derivative) in gray.
    active_mask = np.abs(dydt) > 1e-15
    colours = []
    for i in range(n_sp):
        if not active_mask[i]:
            colours.append("#cfcfcf")
        elif dydt[i] < 0:
            colours.append("#d62728")
        else:
            colours.append("#1f77b4")

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.bar(np.arange(n_sp), dydt, color=colours, edgecolor="black", linewidth=0.3, zorder=2)
    ax.axhline(0.0, color="black", linewidth=0.5, zorder=1)
    ax.set_xticks(np.arange(n_sp))
    ax.set_xticklabels(species_names, rotation=90, fontsize=7)
    ax.set_ylabel("dy/dt  (concentration / time)")
    ax.set_title("dy/dt at t=0 with only GENVOC=1.0, OH=1.0 — BL20 activates exactly five species")
    # Annotate the five active entries.
    for i in np.where(active_mask)[0]:
        ax.text(
            i,
            dydt[i] + (0.03 if dydt[i] > 0 else -0.05) * max(abs(dydt.max()), abs(dydt.min())),
            f"{dydt[i]:+.1f}",
            ha="center",
            va="bottom" if dydt[i] > 0 else "top",
            fontsize=8,
            color="black",
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d62728"),
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        plt.Rectangle((0, 0), 1, 1, color="#cfcfcf"),
    ]
    ax.legend(
        handles,
        ["reactant (consumed)", "product (formed)", "inactive (dy/dt = 0)"],
        loc="lower right",
        fontsize=9,
    )
    _save(fig, "genvoc_only_dydt.png")

    # --- figure 2: uniform state dy/dt --------------------------------
    y_uniform = jnp.ones(n_sp, dtype=jnp.float64) * 0.01
    dydt_u = np.asarray(som_rhs(y_uniform, 1.0, network))

    # Colour bars by sign of dy/dt.
    signs = ["#d62728" if v < 0 else "#2ca02c" for v in dydt_u]
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.bar(np.arange(n_sp), dydt_u, color=signs, edgecolor="black", linewidth=0.3, zorder=2)
    ax.axhline(0.0, color="black", linewidth=0.5, zorder=1)
    ax.set_xticks(np.arange(n_sp))
    ax.set_xticklabels(species_names, rotation=90, fontsize=7)
    ax.set_ylabel("dy/dt  (concentration / time)")
    ax.set_title(
        "dy/dt at uniform state (y = 0.01 everywhere, OH = 1.0) — "
        "shows net-production vs net-consumption structure"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d62728"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c"),
    ]
    ax.legend(
        handles,
        ["net consumed", "net produced"],
        loc="upper right",
        fontsize=9,
    )
    _save(fig, "uniform_state_dydt.png")

    # --- figure 3: Jacobian structure ---------------------------------
    jac_fn = jax.jacfwd(som_rhs, argnums=0)
    jac = np.asarray(jac_fn(y_uniform, 1.0, network))
    bound = max(abs(jac.min()), abs(jac.max()), 1e-12)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    fig, ax = plt.subplots(figsize=(11.0, 9.5))
    im = ax.imshow(jac, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(n_sp))
    ax.set_xticklabels(species_names, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n_sp))
    ax.set_yticklabels(species_names, fontsize=6)
    ax.set_xlabel("species j  (d/dy_j)")
    ax.set_ylabel("species i  (dy_i/dt)")
    ax.set_title(
        "Jacobian  df/dy  at uniform state  "
        "(red = positive coupling, blue = negative; diagonal = self-consumption)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("df_i / dy_j  (concentration / time per concentration)")
    _save(fig, "jacobian_structure.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
