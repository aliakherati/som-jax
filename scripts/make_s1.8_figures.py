"""Generate figures for the S1.8 analytic-decay correctness check.

One panel under ``docs/figures/s1.8/``:

1. ``genvoc_decay_vs_analytic.png`` — two-panel figure. Top: simulated
   GENVOC(t) under constant OH, overlaid on the exact analytic solution
   ``exp(-k_BL20 * OH * t)``. Bottom: relative error vs time.

This is the definitive sim-vs-truth plot: any stoichiometry error, unit
confusion, or solver misconfiguration would show up as deviation from the
analytic curve. It is the visual counterpart to
``test_genvoc_first_order_decay_under_constant_oh``.
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

from som_jax import build_initial, simulate  # noqa: E402
from som_jax.mechanism import SOMNetwork  # noqa: E402

_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.8"


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def main() -> int:
    network = SOMNetwork.from_json(_JSON)
    bl20_idx = network.reaction_index("BL20")
    k_bl20 = float(network.k_OH[bl20_idx])

    # Integrate across 3 e-folds; gives a wide dynamic range for the error.
    oh = 1e-4
    t_final = 3.0 / (k_bl20 * oh)
    save_at = jnp.linspace(0.0, t_final, 200)

    y0 = build_initial(network, {"GENVOC": 1.0})
    traj = simulate(
        network,
        y0,
        oh=oh,
        t_span=(0.0, t_final),
        save_at=save_at,
        rtol=1e-8,
        atol=1e-14,
    )

    t = np.asarray(save_at)
    genvoc_sim = np.asarray(traj.y_of("GENVOC"))
    genvoc_analytic = np.exp(-k_bl20 * oh * t)
    rel_err = np.abs(genvoc_sim - genvoc_analytic) / np.maximum(genvoc_analytic, 1e-20)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # Top panel: overlay
    ax_top.plot(
        t,
        genvoc_sim,
        color="#1f77b4",
        linewidth=2.0,
        label="diffrax Kvaerno5 (rtol=1e-8)",
    )
    ax_top.plot(
        t,
        genvoc_analytic,
        color="#d62728",
        linewidth=1.5,
        linestyle="--",
        label=r"analytic  exp(-k$_{\rm BL20}$ $\cdot$ OH $\cdot$ t)",
    )
    ax_top.set_yscale("log")
    ax_top.set_ylabel("[GENVOC](t)")
    ax_top.set_title(
        "GENVOC first-order decay under constant OH - sim vs analytic "
        f"(k_BL20 = {k_bl20:.3e}, OH = {oh:.0e})"
    )
    ax_top.legend(loc="lower left", fontsize=10)
    ax_top.grid(True, which="both", linewidth=0.3, alpha=0.5)

    # Bottom panel: relative error
    ax_bot.plot(t, rel_err, color="#2ca02c", linewidth=1.5)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(f"time (units of 1/(k_BL20 * OH) = {1 / (k_bl20 * oh):.1f})")
    ax_bot.set_ylabel("|sim - analytic| / analytic")
    ax_bot.axhline(1e-5, color="black", linewidth=0.6, linestyle=":", label="test tolerance (1e-5)")
    ax_bot.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax_bot.legend(loc="upper left", fontsize=9)

    _save(fig, "genvoc_decay_vs_analytic.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
