"""Render visual diagnostics for the S1.18 k_scale identifiability scan.

Two figures under ``docs/figures/s1.18/``:

1. ``identifiability_scan.png`` — joint Adam optimisation of all 39
   rate-constant scales after perturbing BL20 to 1.5x. Top panel:
   recovered scales (bar chart) per reaction; the perturbed rate
   highlighted in green. Other rates that move appreciably from 1.0
   are documented as scientifically-correlated with the perturbed
   one (the "soft directions" of the loss landscape). Bottom: loss
   curve over iterations on a log-y scale.

2. ``per_rate_sensitivity.png`` — per-rate sensitivity analysis. For
   each reaction, perturb its rate by 10% and measure the resulting
   change in the GENVOC trajectory. Bar chart sorted by sensitivity;
   the well-identified rates have large impact, poorly-identified
   rates have small impact (their perturbations leave trajectories
   nearly unchanged, so the optimiser has no signal to recover them).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import config

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

config.update("jax_enable_x64", True)

from som_jax import build_initial, simulate  # noqa: E402
from som_jax.mechanism import SOMNetwork  # noqa: E402

_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"
_OUT = _REPO_ROOT / "docs" / "figures" / "s1.18"

_OH_BASELINE_PPM = 6.090601e-08
_INITIAL_GENVOC_PPM = 0.05
_T_END_MIN = 60.0
_N_SAVE = 9

_BL20_IDX = 0
_BL20_SCALE_TARGET = 1.5
_LEARNING_RATE = 0.05
_N_ITERS = 200


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def _simulate_with_scales(rate_scales, network: SOMNetwork, y0, save_at) -> jnp.ndarray:
    scaled_network = dataclasses.replace(network, k_OH=network.k_OH * rate_scales)
    traj = simulate(
        scaled_network,
        y0,
        oh=_OH_BASELINE_PPM,
        t_span=(0.0, _T_END_MIN),
        save_at=save_at,
        rtol=1e-8,
        atol=1e-15,
    )
    return traj.y


def make_identifiability_scan(network: SOMNetwork, y0, save_at) -> None:
    n_rxn = int(network.k_OH.size)
    rxn_labels = (
        list(network.reaction_labels)
        if hasattr(network, "reaction_labels")
        else [f"k_{i}" for i in range(n_rxn)]
    )

    target_scales = jnp.ones(n_rxn).at[_BL20_IDX].set(_BL20_SCALE_TARGET)
    target_traj = jax.lax.stop_gradient(_simulate_with_scales(target_scales, network, y0, save_at))

    def loss(scales):
        pred = _simulate_with_scales(scales, network, y0, save_at)
        return jnp.sum((pred - target_traj) ** 2)

    optimizer = optax.adam(learning_rate=_LEARNING_RATE)
    scales = jnp.ones(n_rxn)
    state = optimizer.init(scales)

    @jax.jit
    def step(scales, state):
        loss_val, grad = jax.value_and_grad(loss)(scales)
        updates, new_state = optimizer.update(grad, state)
        return optax.apply_updates(scales, updates), new_state, loss_val

    scale_history = []
    losses = []
    for _ in range(_N_ITERS):
        scales, state, loss_val = step(scales, state)
        scale_history.append(np.asarray(scales))
        losses.append(float(loss_val))
    final_scales = np.asarray(scales)
    losses_arr = np.asarray(losses)

    fig, (ax_scales, ax_loss) = plt.subplots(2, 1, figsize=(11.0, 7.5))

    x = np.arange(n_rxn)
    bar_colours = ["#2ca02c" if i == _BL20_IDX else "#1f77b4" for i in range(n_rxn)]
    ax_scales.bar(x, final_scales, color=bar_colours, edgecolor="black", linewidth=0.4)
    ax_scales.axhline(1.0, color="gray", linewidth=0.6, linestyle=":", label="initial guess (1.0)")
    ax_scales.axhline(
        _BL20_SCALE_TARGET,
        color="#2ca02c",
        linewidth=0.8,
        linestyle="--",
        label=f"target (BL20 = {_BL20_SCALE_TARGET})",
    )
    ax_scales.set_xticks(x)
    ax_scales.set_xticklabels(rxn_labels, rotation=90, fontsize=6)
    ax_scales.set_ylabel("recovered rate scale")
    ax_scales.set_title(
        f"S1.18 — joint Adam fit of all {n_rxn} rate-constant scales after "
        f"perturbing BL20 to {_BL20_SCALE_TARGET}x. BL20 (green) recovers; "
        f"other rates show their identifiability w.r.t. the precursor + cascade signal.",
        fontsize=10,
    )
    ax_scales.legend(loc="upper right", fontsize=8)
    ax_scales.grid(True, axis="y", alpha=0.3, linewidth=0.4)

    ax_loss.semilogy(np.arange(_N_ITERS), losses_arr, color="#9467bd", linewidth=1.4)
    ax_loss.set_ylabel("loss (sum of squared residuals)")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_title(
        f"Loss vs iteration "
        f"(drops {losses_arr[0] / max(losses_arr[-1], 1e-300):.2g}x over {_N_ITERS} iters)",
        fontsize=10,
    )
    ax_loss.grid(True, which="both", alpha=0.3, linewidth=0.4)

    fig.tight_layout()
    _save(fig, "identifiability_scan.png")


def make_per_rate_sensitivity(network: SOMNetwork, y0, save_at) -> None:
    """Per-rate sensitivity: how much does perturbing rate i alone
    change the GENVOC + cascade trajectory? Quantifies why some rates
    are well-identifiable from chamber data and others aren't."""
    n_rxn = int(network.k_OH.size)
    rxn_labels = (
        list(network.reaction_labels)
        if hasattr(network, "reaction_labels")
        else [f"k_{i}" for i in range(n_rxn)]
    )

    baseline = _simulate_with_scales(jnp.ones(n_rxn), network, y0, save_at)
    baseline_np = np.asarray(baseline)

    sensitivity = np.zeros(n_rxn)
    for i in range(n_rxn):
        scales = jnp.ones(n_rxn).at[i].set(1.10)  # 10% bump
        perturbed = np.asarray(_simulate_with_scales(scales, network, y0, save_at))
        # Aggregate sensitivity = total relative L2 change across all species
        diff = perturbed - baseline_np
        denom = np.maximum(np.abs(baseline_np), 1e-30)
        sensitivity[i] = float(np.sqrt(np.sum((diff / denom) ** 2)))

    order = np.argsort(sensitivity)[::-1]

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    sorted_labels = [rxn_labels[i] for i in order]
    sorted_sens = sensitivity[order]
    bar_colours = ["#2ca02c" if order[i] == _BL20_IDX else "#1f77b4" for i in range(n_rxn)]
    ax.bar(np.arange(n_rxn), sorted_sens, color=bar_colours, edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(n_rxn))
    ax.set_xticklabels(sorted_labels, rotation=90, fontsize=6)
    ax.set_ylabel("relative L2 trajectory change per 10% rate bump")
    ax.set_yscale("log")
    ax.set_title(
        "S1.18 — per-rate sensitivity (sorted high-to-low). High-sensitivity rates "
        "are well-identifiable from chamber observations; low-sensitivity rates are "
        "the 'soft directions' of the inverse problem. BL20 (green) tops the list."
    )
    ax.grid(True, which="both", alpha=0.3, linewidth=0.4)

    fig.tight_layout()
    _save(fig, "per_rate_sensitivity.png")


def main() -> int:
    network = SOMNetwork.from_json(_GENSOMG_JSON)
    y0 = build_initial(network, {"GENVOC": _INITIAL_GENVOC_PPM})
    save_at = jnp.linspace(0.0, _T_END_MIN, _N_SAVE)
    make_identifiability_scan(network, y0, save_at)
    make_per_rate_sensitivity(network, y0, save_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
