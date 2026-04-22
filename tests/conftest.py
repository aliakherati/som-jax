"""Pytest-wide JAX configuration.

Scientific ODEs in the SOM port need float64 precision; JAX defaults to
float32. We flip the switch here rather than in ``som_jax/__init__.py`` so
that users who want float32 (e.g., for GPU experiments) are free to do so
by importing ``som_jax`` without our test harness forcing the choice on
them. Callers outside the test suite should enable x64 explicitly:

    from jax import config
    config.update("jax_enable_x64", True)
"""

from __future__ import annotations

from jax import config

config.update("jax_enable_x64", True)
