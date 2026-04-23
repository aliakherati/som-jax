"""Differentiable Statistical Oxidation Model (SOM) in Python/JAX.

Port of the Fortran reference implementation embedded in the SAPRC-14 mechanism.
See the project README for scope, scientific references, and status.
"""

from importlib.metadata import PackageNotFoundError, version

from som_jax.oh import (
    oh_constant,
    oh_exponential_decay,
    oh_linear_ramp,
    oh_piecewise_linear,
)
from som_jax.rhs import som_rhs
from som_jax.simulate import SOMTrajectory, build_initial, simulate

try:
    __version__ = version("som-jax")
except PackageNotFoundError:  # pragma: no cover - only during uninstalled dev
    __version__ = "0.0.0+unknown"

__all__ = [
    "SOMTrajectory",
    "__version__",
    "build_initial",
    "oh_constant",
    "oh_exponential_decay",
    "oh_linear_ramp",
    "oh_piecewise_linear",
    "simulate",
    "som_rhs",
]
