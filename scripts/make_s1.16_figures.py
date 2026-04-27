"""Render visual diagnostics for the S1.16/S1.17 differentiability tests.

Two figures under ``docs/figures/s1.16/`` and ``docs/figures/s1.17/``:

1. ``s1.16/jacobian_overview.png`` (S1.16) — top: GENVOC(t) at three
   OH levels showing the smooth dependence on the parameter. Bottom:
   ``∂[GENVOC](t) / ∂OH`` from ``jax.jacrev`` overlaid on a central-
   difference reference; the two curves are visually identical, the
   relative-error subplot lives below ~1e-10 across the trajectory.

2. ``s1.17/optax_recovery.png`` (S1.17) — three-panel. Top: GENVOC(t)
   target trajectory (green) with the initial-guess trajectory (red
   dashed) and the recovered trajectory (blue) after 200 Adam
   iterations. Middle: the parameter ``oh_scale`` vs iteration,
   converging to the target value. Bottom: log loss vs iteration.
"""

from __future__ import annotations

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
_OUT_S1_16 = _REPO_ROOT / "docs" / "figures" / "s1.16"
_OUT_S1_17 = _REPO_ROOT / "docs" / "figures" / "s1.17"

_OH_BASELINE_PPM = 6.090601e-08
_INITIAL_GENVOC_PPM = 0.05
_T_END_MIN = 240.0
_OH_SCALE_TARGET = 2.0
_OH_SCALE_INIT = 0.6
_LEARNING_RATE = 0.05
_N_ITERS = 200


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(_REPO_ROOT)}  ({size_kb:.1f} KB)")


def _genvoc_traj(oh_ppm, network, y0, save_at):
    traj = simulate(
        network,
        y0,
        oh=oh_ppm,
        t_span=(0.0, _T_END_MIN),
        save_at=save_at,
        rtol=1e-8,
        atol=1e-15,
    )
    return traj.y_of("GENVOC")


def make_s1_16_figure(network: SOMNetwork, y0, save_at_dense) -> None:
    """Visualise smoothness of the trajectory in OH and the gradient
    agreement with central-difference."""
    # Top: GENVOC(t) at 3 OH levels (baseline, 1.5×, 2×).
    fig, (ax_top, ax_grad, ax_err) = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True)

    for scale, colour in zip([1.0, 1.5, 2.0], ["#1f77b4", "#2ca02c", "#d62728"], strict=False):
        traj = _genvoc_traj(scale * _OH_BASELINE_PPM, network, y0, save_at_dense)
        ax_top.plot(
            np.asarray(save_at_dense) / 60.0,
            np.asarray(traj),
            color=colour,
            linewidth=1.4,
            label=f"oh_scale = {scale:.1f}",
        )
    ax_top.set_ylabel("[GENVOC] (ppm)")
    ax_top.set_title(
        "S1.16 — GENVOC(t) is smooth in the OH parameter (higher OH ⇒ faster decay)",
        fontsize=10,
    )
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(True, alpha=0.3, linewidth=0.4)

    # Middle: ∂[GENVOC](t) / ∂OH from jax.jacrev (autodiff) and
    # central-difference. Use the baseline OH point.
    def traj_fn(oh):
        return _genvoc_traj(oh, network, y0, save_at_dense)

    jac_auto = np.asarray(jax.jacrev(traj_fn)(_OH_BASELINE_PPM))
    eps = _OH_BASELINE_PPM * 1e-4
    jac_num = np.asarray(
        (
            _genvoc_traj(_OH_BASELINE_PPM + eps, network, y0, save_at_dense)
            - _genvoc_traj(_OH_BASELINE_PPM - eps, network, y0, save_at_dense)
        )
        / (2.0 * eps)
    )
    t_h = np.asarray(save_at_dense) / 60.0
    ax_grad.plot(t_h, jac_auto, color="#1f77b4", linewidth=1.6, label="jax.jacrev")
    ax_grad.plot(t_h, jac_num, color="black", linestyle="--", linewidth=1.0, label="central-diff")
    ax_grad.set_ylabel("∂[GENVOC](t) / ∂OH (ppm / ppm)")
    ax_grad.set_title("Autodiff gradient overlaid on central-difference reference", fontsize=10)
    ax_grad.legend(loc="upper right", fontsize=8)
    ax_grad.grid(True, alpha=0.3, linewidth=0.4)

    # Bottom: relative error between autodiff and central-diff.
    rel_err = np.abs(jac_auto - jac_num) / np.maximum(np.abs(jac_num), 1e-30)
    # Mask the t=0 point where both are exactly zero.
    rel_err_plot = np.where(np.abs(jac_num) < 1e-20, np.nan, rel_err)
    ax_err.plot(t_h, rel_err_plot, color="#9467bd", linewidth=1.4)
    ax_err.set_yscale("log")
    ax_err.set_ylabel("|autodiff - num| / |num|")
    ax_err.set_xlabel("time (h)")
    ax_err.set_title(
        f"Relative error between autodiff and central-diff (max {np.nanmax(rel_err_plot):.1e})",
        fontsize=10,
    )
    ax_err.grid(True, which="both", alpha=0.3, linewidth=0.4)
    ax_err.axhline(0.01, color="black", linewidth=0.6, linestyle=":", label="1% tol")
    ax_err.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _save(fig, _OUT_S1_16, "jacobian_overview.png")


def make_s1_17_figure(network: SOMNetwork, y0, save_at) -> None:
    """Run the optax recovery and render trajectory + parameter +
    loss curves."""
    target_traj = jax.lax.stop_gradient(
        _genvoc_traj(_OH_SCALE_TARGET * _OH_BASELINE_PPM, network, y0, save_at)
    )

    def loss_fn(oh_scale):
        pred = _genvoc_traj(oh_scale * _OH_BASELINE_PPM, network, y0, save_at)
        return jnp.sum((pred - target_traj) ** 2)

    optimizer = optax.adam(learning_rate=_LEARNING_RATE)
    state = optimizer.init(jnp.asarray(_OH_SCALE_INIT))

    @jax.jit
    def step(oh_scale, state):
        loss_val, grad = jax.value_and_grad(loss_fn)(oh_scale)
        updates, new_state = optimizer.update(grad, state)
        return optax.apply_updates(oh_scale, updates), new_state, loss_val

    oh_scale = jnp.asarray(_OH_SCALE_INIT)
    scales = []
    losses = []
    for _ in range(_N_ITERS):
        oh_scale, state, loss_val = step(oh_scale, state)
        scales.append(float(oh_scale))
        losses.append(float(loss_val))
    scales_arr = np.asarray(scales)
    losses_arr = np.asarray(losses)

    # Trajectories at start, midway, and end.
    init_traj = _genvoc_traj(_OH_SCALE_INIT * _OH_BASELINE_PPM, network, y0, save_at)
    mid_idx = _N_ITERS // 4
    mid_traj = _genvoc_traj(scales_arr[mid_idx] * _OH_BASELINE_PPM, network, y0, save_at)
    final_traj = _genvoc_traj(scales_arr[-1] * _OH_BASELINE_PPM, network, y0, save_at)

    fig, (ax_traj, ax_param, ax_loss) = plt.subplots(3, 1, figsize=(10.0, 9.5))

    t_h = np.asarray(save_at) / 60.0
    ax_traj.plot(
        t_h,
        np.asarray(target_traj),
        color="#2ca02c",
        linewidth=2.2,
        label=f"target (oh_scale = {_OH_SCALE_TARGET})",
    )
    ax_traj.plot(
        t_h,
        np.asarray(init_traj),
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"initial guess (oh_scale = {_OH_SCALE_INIT})",
    )
    ax_traj.plot(
        t_h,
        np.asarray(mid_traj),
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label=f"iter {mid_idx} (oh_scale = {scales_arr[mid_idx]:.3f})",
    )
    ax_traj.plot(
        t_h,
        np.asarray(final_traj),
        color="#1f77b4",
        linewidth=1.4,
        label=f"recovered (oh_scale = {scales_arr[-1]:.3f}, iter {_N_ITERS})",
    )
    ax_traj.set_xlabel("time (h)")
    ax_traj.set_ylabel("[GENVOC] (ppm)")
    ax_traj.set_title(
        "S1.17 — GENVOC trajectory: target vs initial guess vs recovered",
        fontsize=10,
    )
    ax_traj.legend(loc="upper right", fontsize=8)
    ax_traj.grid(True, alpha=0.3, linewidth=0.4)

    ax_param.plot(np.arange(_N_ITERS), scales_arr, color="#1f77b4", linewidth=1.4)
    ax_param.axhline(
        _OH_SCALE_TARGET,
        color="#2ca02c",
        linewidth=0.8,
        linestyle=":",
        label=f"target = {_OH_SCALE_TARGET}",
    )
    ax_param.set_ylabel("oh_scale")
    final_rel_err = abs(scales_arr[-1] - _OH_SCALE_TARGET) / _OH_SCALE_TARGET
    ax_param.set_title(
        f"Adam parameter trajectory — recovered to within {final_rel_err:.2%} of target",
        fontsize=10,
    )
    ax_param.legend(loc="lower right", fontsize=8)
    ax_param.grid(True, alpha=0.3, linewidth=0.4)

    ax_loss.semilogy(np.arange(_N_ITERS), losses_arr, color="#9467bd", linewidth=1.4)
    ax_loss.set_ylabel("loss (sum of squared residuals)")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_title(
        f"Loss vs iteration (drops {losses_arr[0] / losses_arr[-1]:.2g}x over {_N_ITERS} iters)",
        fontsize=10,
    )
    ax_loss.grid(True, which="both", alpha=0.3, linewidth=0.4)

    fig.tight_layout()
    _save(fig, _OUT_S1_17, "optax_recovery.png")


def main() -> int:
    network = SOMNetwork.from_json(_GENSOMG_JSON)
    y0 = build_initial(network, {"GENVOC": _INITIAL_GENVOC_PPM})
    # Dense save points for the S1.16 figure; the S1.17 recovery uses
    # the same setup as the test (9 points) for speed parity.
    save_at_dense = jnp.linspace(0.0, _T_END_MIN, 49)
    save_at_recovery = jnp.linspace(0.0, _T_END_MIN, 9)
    make_s1_16_figure(network, y0, save_at_dense)
    make_s1_17_figure(network, y0, save_at_recovery)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
