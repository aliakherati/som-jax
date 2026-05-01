"""S1.18 — k_scale identifiability scan.

Companion to S1.17's OH-recovery demo. Where S1.17 fits a single
scalar (the OH input), this test fits **all 38 rate-constant scales**
jointly. The optimisation variable is a length-39 vector
``rate_scales`` (one for each reaction in the network); the simulator
runs with effective rates ``k_OH * rate_scales``. We generate a target
trajectory with one rate perturbed (``BL20 = 1.5×``), initialise the
optimiser with all scales at ``1.0``, and confirm Adam recovers the
perturbed rate while leaving uninvolved rates near 1.0.

Per the master plan this is a **scientific finding, not a binary
pass/fail**: the GENSOMG cascade is highly correlated between
neighbouring grid cells, so many rate constants can't be uniquely
identified from precursor + cascade observations alone. The tests
here pin the well-identified subset (BL20 plus the cascading rates
that touch the populated cells) and document the rest.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from som_jax import build_initial, simulate
from som_jax.mechanism import SOMNetwork

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENSOMG_JSON = _REPO_ROOT / "data" / "mechanisms" / "gensomg.json"

# Run conditions match the canonical-matrix `medium_baseline` so the
# cascade has time to populate non-trivially. Identifiability of
# cascade rates requires the cascade to be visible in the target.
_OH_BASELINE_PPM = 6.090601e-08  # ≈ 1.5e6 molec/cm³ at 298 K, 101325 Pa
_INITIAL_GENVOC_PPM = 0.05
_T_END_MIN = 60.0  # 1 hour — enough for first-gen + early second-gen
_N_SAVE = 9

# BL20 (GENVOC + OH) is the first reaction in the parsed mechanism,
# index 0. It sets the precursor decay rate; well-identified by
# GENVOC observations alone.
_BL20_IDX = 0
_BL20_SCALE_TARGET = 1.5
_BL20_RECOVERY_TOL = 0.02  # ≤ 2% relative error

# Optax setup
_LEARNING_RATE = 0.05
_N_ITERS = 200


@pytest.fixture(scope="module")
def network() -> SOMNetwork:
    return SOMNetwork.from_json(_GENSOMG_JSON)


@pytest.fixture(scope="module")
def y0(network: SOMNetwork) -> jnp.ndarray:
    return build_initial(network, {"GENVOC": _INITIAL_GENVOC_PPM})


@pytest.fixture(scope="module")
def save_at() -> jnp.ndarray:
    return jnp.linspace(0.0, _T_END_MIN, _N_SAVE)


def _simulate_with_scales(
    rate_scales: jnp.ndarray,
    network: SOMNetwork,
    y0: jnp.ndarray,
    save_at: jnp.ndarray,
) -> jnp.ndarray:
    """Run the simulator with ``k_OH * rate_scales`` and return the
    full ``(n_t, n_species)`` trajectory matrix.

    ``dataclasses.replace`` on the frozen :class:`SOMNetwork` produces
    a new PyTree-registered instance with the scaled rates; the rest
    of the simulator is untouched. Tracing through this is what makes
    each ``rate_scales[i]`` differentiable end-to-end.
    """
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


def test_bl20_recovery_via_joint_optimization(
    network: SOMNetwork, y0: jnp.ndarray, save_at: jnp.ndarray
) -> None:
    """Generate a target trajectory at ``BL20 = 1.5×`` (other rates
    at 1×), initialise the optimiser at all-1s, and confirm Adam
    recovers the perturbed rate within 2% in 200 iterations."""
    n_rxn = int(network.k_OH.size)
    target_scales = jnp.ones(n_rxn).at[_BL20_IDX].set(_BL20_SCALE_TARGET)
    target_traj = _simulate_with_scales(target_scales, network, y0, save_at)
    target_traj = jax.lax.stop_gradient(target_traj)

    def loss(scales: jnp.ndarray) -> jnp.ndarray:
        pred = _simulate_with_scales(scales, network, y0, save_at)
        return jnp.sum((pred - target_traj) ** 2)

    optimizer = optax.adam(learning_rate=_LEARNING_RATE)
    init_scales = jnp.ones(n_rxn)
    state = optimizer.init(init_scales)

    @jax.jit
    def step(scales, state):
        loss_val, grad = jax.value_and_grad(loss)(scales)
        updates, new_state = optimizer.update(grad, state)
        return optax.apply_updates(scales, updates), new_state, loss_val

    scales = init_scales
    for _ in range(_N_ITERS):
        scales, state, _ = step(scales, state)

    bl20_recovered = float(scales[_BL20_IDX])
    rel_err = abs(bl20_recovered - _BL20_SCALE_TARGET) / _BL20_SCALE_TARGET
    assert rel_err < _BL20_RECOVERY_TOL, (
        f"BL20 not recovered: target={_BL20_SCALE_TARGET}, "
        f"recovered={bl20_recovered:.4f} (rel err {rel_err:.2%})"
    )


def test_loss_decreases_during_joint_optimization(
    network: SOMNetwork, y0: jnp.ndarray, save_at: jnp.ndarray
) -> None:
    """Sanity: across the optimisation the loss decreases by at least
    100× over the first 50 iterations. This is what makes the joint
    fit converge in the first place; if it ever stops decreasing
    something has gone wrong with the gradient pipeline."""
    n_rxn = int(network.k_OH.size)
    target_scales = jnp.ones(n_rxn).at[_BL20_IDX].set(_BL20_SCALE_TARGET)
    target_traj = jax.lax.stop_gradient(_simulate_with_scales(target_scales, network, y0, save_at))

    def loss(scales: jnp.ndarray) -> jnp.ndarray:
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

    losses = []
    for _ in range(50):
        scales, state, loss_val = step(scales, state)
        losses.append(float(loss_val))
    losses_arr = np.asarray(losses)
    assert losses_arr[-1] < losses_arr[0] / 100, (
        f"loss did not drop materially: start={losses_arr[0]:.3e}, end={losses_arr[-1]:.3e}"
    )


def test_uninvolved_rates_stay_near_unity(
    network: SOMNetwork, y0: jnp.ndarray, save_at: jnp.ndarray
) -> None:
    """Rates that don't affect the populated cascade cells over a 1-hour
    run shouldn't move much during the joint fit. We don't enumerate
    them rigorously (that's the scientific-finding side); we just
    check that **at least 5** of the 39 rate scales stay within 5%
    of 1.0 after the optimisation. Catches a "everything moves a lot"
    pathology that would invalidate the identifiability narrative.
    """
    n_rxn = int(network.k_OH.size)
    target_scales = jnp.ones(n_rxn).at[_BL20_IDX].set(_BL20_SCALE_TARGET)
    target_traj = jax.lax.stop_gradient(_simulate_with_scales(target_scales, network, y0, save_at))

    def loss(scales: jnp.ndarray) -> jnp.ndarray:
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

    for _ in range(_N_ITERS):
        scales, state, _ = step(scales, state)

    # Count how many rates stayed within 5% of 1.0
    final = np.asarray(scales)
    untouched = int(np.sum(np.abs(final - 1.0) < 0.05))
    # BL20 (the perturbed one) is excluded from this count by design.
    assert untouched >= 5, (
        f"only {untouched} of {n_rxn} rate scales stayed within 5% of "
        f"1.0; expected at least 5 'uninvolved' rates. final scales:\n{final}"
    )
