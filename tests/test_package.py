"""Smoke tests — package is importable and exposes a version."""

import re

import som_jax


def test_version_is_exposed() -> None:
    assert isinstance(som_jax.__version__, str)
    assert som_jax.__version__


def test_version_matches_semver() -> None:
    # Allow "0.0.0+unknown" fallback during local dev when package is not installed.
    semver = re.compile(r"^\d+\.\d+\.\d+(?:[-+].+)?$")
    assert semver.match(som_jax.__version__)
