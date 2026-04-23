"""OH trajectory helpers.

``simulate()`` accepts ``oh`` as either a scalar (treated as constant over the
integration window) or a ``Callable[[Array], Array]`` that returns the OH
concentration at a given time. The callable form lets callers wire up ramps,
piecewise-linear chamber profiles, decay curves, or even learned / parametric
OH models — anything that's a pure JAX function of ``t`` works.

This module provides a small library of common shapes and the adapter used
internally to promote scalars to callables.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import jax.numpy as jnp
from jax import Array

OHCallable: TypeAlias = Callable[[Array], Array]
"""Type of a time-varying OH trajectory, ``t -> OH(t)`` as a JAX scalar."""

OHInput: TypeAlias = float | Array | OHCallable
"""User-facing OH input to :func:`som_jax.simulate`.

Scalars are promoted to a constant callable; callables are used as-is.
"""


def as_oh_callable(oh: OHInput) -> OHCallable:
    """Promote an ``OHInput`` into an :data:`OHCallable` with signature
    ``t -> OH(t)``.

    Scalars (Python floats or JAX scalars) become constant callables.
    Callables are returned unchanged — they're trusted to be pure JAX
    functions of a scalar time.
    """
    if callable(oh):
        return oh
    return oh_constant(oh)


def oh_constant(value: float | Array) -> OHCallable:
    """Return a callable that yields ``value`` at every time."""
    val = jnp.asarray(value)

    def _f(t: Array) -> Array:
        del t
        return val

    return _f


def oh_linear_ramp(
    t0: float | Array,
    t1: float | Array,
    value0: float | Array,
    value1: float | Array,
) -> OHCallable:
    """Return a callable that linearly interpolates OH from ``value0`` at
    ``t0`` to ``value1`` at ``t1``.

    Outside the ``[t0, t1]`` interval the callable clamps to the end-values,
    i.e. ``OH(t < t0) = value0`` and ``OH(t > t1) = value1``. This is the
    usual convention for chamber experiments where the OH lamp is stabilised
    before the precursor injection.
    """
    t0_ = jnp.asarray(t0)
    t1_ = jnp.asarray(t1)
    v0 = jnp.asarray(value0)
    v1 = jnp.asarray(value1)
    span = t1_ - t0_

    def _f(t: Array) -> Array:
        frac = jnp.clip((t - t0_) / span, 0.0, 1.0)
        return v0 + frac * (v1 - v0)

    return _f


def oh_piecewise_linear(times: Array, values: Array) -> OHCallable:
    """Return a callable that linearly interpolates OH between the given
    ``(times, values)`` knots.

    Outside the knot range the callable clamps to the nearest end-value
    (matches ``jnp.interp``'s default behaviour). ``times`` must be
    monotonically increasing.
    """
    ts = jnp.asarray(times)
    vs = jnp.asarray(values)

    def _f(t: Array) -> Array:
        return jnp.interp(t, ts, vs)

    return _f


def oh_exponential_decay(
    value0: float | Array,
    decay_rate: float | Array,
    t0: float | Array = 0.0,
) -> OHCallable:
    """Return a callable ``t -> value0 * exp(-decay_rate * (t - t0))``.

    Useful for simulating chamber experiments where OH photostationary
    state decays after lamp shutoff, or for ambient modelling with
    diurnal decay terms.
    """
    v0 = jnp.asarray(value0)
    k = jnp.asarray(decay_rate)
    t0_ = jnp.asarray(t0)

    def _f(t: Array) -> Array:
        return v0 * jnp.exp(-k * (t - t0_))

    return _f
