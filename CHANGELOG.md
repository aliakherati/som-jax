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

## [0.1.0] — planned

First usable release. Gate: all 10 canonical Fortran goldens match within 0.1% relative L2 per species; `dLVP` recovery differentiability test passes.
