"""SOM reaction mechanism — Fortran parser, typed representation, JSON I/O.

The SOM reactions in the Fortran reference are embedded in the SAPRC-14
mechanism files (``saprc14_rev1.som``, ``saprc14_rev1.mod``, ``saprc14_rev1.doc``).
This subpackage extracts them into a Python dataclass tree and a committed
JSON artifact under ``data/mechanisms/``.

Only the extraction machinery lives here. Vectorized representations for use
with JAX (e.g. the stoichiometry matrix) are built in ``som_jax.mechanism.network``
from a parsed :class:`Mechanism`.
"""

from som_jax.mechanism.json_io import mechanism_from_json, mechanism_to_json
from som_jax.mechanism.network import SOMNetwork
from som_jax.mechanism.parser import parse_mechanism
from som_jax.mechanism.types import (
    Mechanism,
    MechanismMetadata,
    Product,
    Reaction,
    SourceFileRef,
    Species,
)

__all__ = [
    "Mechanism",
    "MechanismMetadata",
    "Product",
    "Reaction",
    "SOMNetwork",
    "SourceFileRef",
    "Species",
    "mechanism_from_json",
    "mechanism_to_json",
    "parse_mechanism",
]
