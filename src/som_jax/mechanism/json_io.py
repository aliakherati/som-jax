"""Deterministic JSON serialization for a parsed :class:`Mechanism`.

The JSON artifact under ``data/mechanisms/`` is the canonical, committed form
consumed at runtime by ``som_jax.mechanism.network`` (not yet implemented).
Formatting is pinned so that diffs across parser runs only reflect genuine
scientific changes, not whitespace/ordering noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from som_jax.mechanism.types import (
    Mechanism,
    MechanismMetadata,
    Product,
    Reaction,
    SourceFileRef,
    Species,
)

_SCHEMA_VERSION = 1


def _species_to_dict(sp: Species) -> dict[str, Any]:
    return {
        "name": sp.name,
        "carbon": sp.carbon,
        "oxygen": sp.oxygen,
        "molecular_weight": sp.molecular_weight,
        "is_precursor": sp.is_precursor,
        "source_line_mod": sp.source_line_mod,
    }


def _species_from_dict(d: dict[str, Any]) -> Species:
    return Species(
        name=d["name"],
        carbon=int(d["carbon"]),
        oxygen=int(d["oxygen"]),
        molecular_weight=float(d["molecular_weight"]),
        is_precursor=bool(d["is_precursor"]),
        source_line_mod=int(d["source_line_mod"]),
    )


def _product_to_dict(p: Product) -> dict[str, Any]:
    return {"name": p.name, "branch": p.branch, "oxy_yield": p.oxy_yield}


def _product_from_dict(d: dict[str, Any]) -> Product:
    return Product(name=d["name"], branch=float(d["branch"]), oxy_yield=float(d["oxy_yield"]))


def _reaction_to_dict(r: Reaction) -> dict[str, Any]:
    return {
        "label": r.label,
        "reactants": list(r.reactants),
        "products": [_product_to_dict(p) for p in r.products],
        "rate_cm3_per_mol_per_s": r.rate_cm3_per_mol_per_s,
        "source_line_mod": r.source_line_mod,
        "source_line_doc": r.source_line_doc,
    }


def _reaction_from_dict(d: dict[str, Any]) -> Reaction:
    return Reaction(
        label=d["label"],
        reactants=tuple(d["reactants"]),
        products=tuple(_product_from_dict(p) for p in d["products"]),
        rate_cm3_per_mol_per_s=float(d["rate_cm3_per_mol_per_s"]),
        source_line_mod=int(d["source_line_mod"]),
        source_line_doc=int(d["source_line_doc"]),
    )


def _metadata_to_dict(m: MechanismMetadata) -> dict[str, Any]:
    return {
        "parser_version": m.parser_version,
        "parsed_at_utc": m.parsed_at_utc,
        "source_files": [{"path": s.path, "sha256": s.sha256} for s in m.source_files],
    }


def _metadata_from_dict(d: dict[str, Any]) -> MechanismMetadata:
    return MechanismMetadata(
        parser_version=d["parser_version"],
        parsed_at_utc=d["parsed_at_utc"],
        source_files=tuple(
            SourceFileRef(path=s["path"], sha256=s["sha256"]) for s in d["source_files"]
        ),
    )


def mechanism_to_json(mech: Mechanism, path: Path | str) -> None:
    """Write ``mech`` to ``path`` as pretty-printed, deterministic JSON."""
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "family": mech.family,
        "precursor": mech.precursor,
        "grid": {"c_max": mech.grid_c_max, "o_max": mech.grid_o_max},
        "species": [_species_to_dict(sp) for sp in mech.species],
        "reactions": [_reaction_to_dict(r) for r in mech.reactions],
        "metadata": _metadata_to_dict(mech.metadata) if mech.metadata is not None else None,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def mechanism_from_json(path: Path | str) -> Mechanism:
    """Read a :class:`Mechanism` previously written by :func:`mechanism_to_json`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version={raw.get('schema_version')!r}; "
            f"this som-jax build expects {_SCHEMA_VERSION}"
        )
    metadata_raw = raw.get("metadata")
    metadata = _metadata_from_dict(metadata_raw) if metadata_raw is not None else None
    return Mechanism(
        family=raw["family"],
        precursor=raw["precursor"],
        grid_c_max=int(raw["grid"]["c_max"]),
        grid_o_max=int(raw["grid"]["o_max"]),
        species=tuple(_species_from_dict(sp) for sp in raw["species"]),
        reactions=tuple(_reaction_from_dict(r) for r in raw["reactions"]),
        metadata=metadata,
    )
