"""Differentiable Statistical Oxidation Model (SOM) in Python/JAX.

Port of the Fortran reference implementation embedded in the SAPRC-14 mechanism.
See the project README for scope, scientific references, and status.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("som-jax")
except PackageNotFoundError:  # pragma: no cover - only during uninstalled dev
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
