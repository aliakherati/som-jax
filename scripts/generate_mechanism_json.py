"""Regenerate ``data/mechanisms/<family>.json`` from the Fortran SAPRC-14 sources.

This is a development utility, not part of the runtime API. Run it whenever the
Fortran mechanism files change.

Usage
-----
    python scripts/generate_mechanism_json.py \\
        --mod ../som-tomas-app/src/saprc14_rev1.mod \\
        --doc ../som-tomas-app/src/saprc14_rev1.doc \\
        --som ../som-tomas-app/src/saprc14_rev1.som \\
        --family GENSOMG \\
        --output data/mechanisms/gensomg.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow "python scripts/..." to resolve the in-repo package without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from som_jax.mechanism import mechanism_to_json, parse_mechanism  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mod", type=Path, required=True, help="path to saprc14_rev1.mod")
    p.add_argument("--doc", type=Path, required=True, help="path to saprc14_rev1.doc")
    p.add_argument("--som", type=Path, default=None, help="path to saprc14_rev1.som (optional)")
    p.add_argument(
        "--family", default="GENSOMG", help="SOM family (only GENSOMG supported for now)"
    )
    p.add_argument("--output", type=Path, required=True, help="output JSON path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mech = parse_mechanism(
        mod_path=args.mod,
        doc_path=args.doc,
        som_path=args.som,
        family=args.family,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mechanism_to_json(mech, args.output)
    n_species = len(mech.species)
    n_reactions = len(mech.reactions)
    print(
        f"wrote {args.output}: family={mech.family}, "
        f"species={n_species}, reactions={n_reactions}, "
        f"grid=({mech.grid_c_max},{mech.grid_o_max})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
