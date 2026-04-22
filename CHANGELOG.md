# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package scaffolding: `pyproject.toml`, `src/` layout, MIT license.
- CI workflow: pytest + ruff + mypy on Python 3.11 and 3.12 (macOS + Linux, CPU only).
- Runtime dependencies pinned: `numpy`, `jax`, `jaxlib`, `diffrax`.
- README and CHANGELOG with accurate project status.
- **S1.1 — mechanism parser**: `som_jax.mechanism.{types, parser, json_io}` extracts the GENSOMG family (GENVOC precursor + 40 grid species; 39 OH-driven reactions) from the Fortran `saprc14_rev1.{mod,doc,som}` source into a typed `Mechanism` dataclass tree and a deterministic committed JSON artifact at `data/mechanisms/gensomg.json`. Products carry two coefficients (`branch` × `oxy_yield`) matching the Fortran `#a #b` convention. Rate labels: `BL20` (GENVOC+OH, 8.319×10³ cm³ mol⁻¹ s⁻¹) and `S1.1`…`S38.1` for the grid reactions.
- Parser records SHA-256 digests of the three source files in `Mechanism.metadata`, so mechanism drift is detectable.
- `scripts/generate_mechanism_json.py` — CLI to regenerate the committed JSON.
- `docs/references.bib` with Cappa & Wilson (2012) and Cappa et al. (2013).
- Unit tests: 16 tests covering species cards, reaction structure, rate values, JSON round-trip, and metadata integrity.

### Notes
- Product yields per reaction do **not** uniformly sum to 1.0: on-grid reactions sum to `on_grid_branch + N_frag × fragmentation_branch`. Fragmentation reactions with multiple fragment products (e.g. S2.1, sum ≈ 2.0) are consistent with one parent generating multiple smaller fragments. Fortran-encoded zero-yield pathways (e.g. S3.1, S4.1) are preserved verbatim; they represent mass loss out of the tracked grid.

## [0.1.0] — planned

First usable release. Gate: all 10 canonical Fortran goldens match within 0.1% relative L2 per species; `dLVP` recovery differentiability test passes.
