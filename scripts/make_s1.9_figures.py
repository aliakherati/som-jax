"""Generate figures for S1.9 time-varying-OH support.

One panel under ``docs/figures/s1.9/``:

1. ``genvoc_decay_ramp_oh.png`` — three-panel figure. Top: OH(t) ramp
   profile. Middle: simulated GENVOC(t) overlaid on the analytic solution
   ``exp(-k * integral(OH, 0, t))``. Bottom: relative error.

This is the counterpart of ``docs/figures/s1.8/genvoc_decay_vs_analytic.png``
but with a non-constant OH, exercising the callable-OH pathway end-to-end.
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

from som_jax import build_initial, oh_linear_ramp, simulate  # noqa: E402
from som_jax.mechanism import SOMNetwork  # noqa: E402

_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.9"


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

    oh_start = 5e-5
    oh_end = 2e-4
    oh_mean = 0.5 * (oh_start + oh_end)
    t_final = 2.0 / (k_bl20 * oh_mean)

    save_at = jnp.linspace(0.0, t_final, 200)
    y0 = build_initial(network, {"GENVOC": 1.0})
    oh_fn = oh_linear_ramp(0.0, t_final, oh_start, oh_end)
    traj = simulate(network, y0, oh=oh_fn, t_span=(0.0, t_final), save_at=save_at, rtol=1e-8)

    t = np.asarray(save_at)
    genvoc_sim = np.asarray(traj.y_of("GENVOC"))
    a = oh_start
    b = (oh_end - oh_start) / t_final
    integ_oh = a * t + 0.5 * b * t**2
    genvoc_analytic = np.exp(-k_bl20 * integ_oh)
    rel_err = np.abs(genvoc_sim - genvoc_analytic) / np.maximum(genvoc_analytic, 1e-20)

    fig, (ax_oh, ax_mid, ax_bot) = plt.subplots(
        3,
        1,
        figsize=(8.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3, 1], "hspace": 0.08},
    )

    # Top: OH(t).
    ax_oh.plot(t, a + b * t, color="#ff7f0e", linewidth=2.0)
    ax_oh.set_ylabel("OH(t)")
    ax_oh.set_title(
        "Time-varying OH: linear ramp from 5e-5 to 2e-4 - sim vs analytic integral solution"
    )
    ax_oh.grid(True, linewidth=0.3, alpha=0.5)

    # Middle: sim vs analytic.
    ax_mid.plot(
        t,
        genvoc_sim,
        color="#1f77b4",
        linewidth=2.0,
        label="diffrax Kvaerno5 (rtol=1e-8)",
    )
    ax_mid.plot(
        t,
        genvoc_analytic,
        color="#d62728",
        linewidth=1.5,
        linestyle="--",
        label=r"analytic  exp($-k_{\rm BL20} \int_0^t {\rm OH}(s)\,ds$)",
    )
    ax_mid.set_yscale("log")
    ax_mid.set_ylabel("[GENVOC](t)")
    ax_mid.legend(loc="lower left", fontsize=9)
    ax_mid.grid(True, which="both", linewidth=0.3, alpha=0.5)

    # Bottom: relative error.
    ax_bot.plot(t, rel_err, color="#2ca02c", linewidth=1.5)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(f"time (units chosen so integrated k*OH ~ 2 at t={t_final:.1f})")
    ax_bot.set_ylabel("|sim - analytic| / analytic")
    ax_bot.axhline(1e-5, color="black", linewidth=0.6, linestyle=":", label="test tolerance (1e-5)")
    ax_bot.legend(loc="upper left", fontsize=9)
    ax_bot.grid(True, which="both", linewidth=0.3, alpha=0.5)

    _save(fig, "genvoc_decay_ramp_oh.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
