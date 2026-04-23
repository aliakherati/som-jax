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
- **S1.4 — SOM network PyTree**: `som_jax.mechanism.network.SOMNetwork` — JAX-native representation of a parsed `Mechanism`. Species names and reaction labels are static PyTree aux data; numeric fields (`carbon`, `oxygen`, `molecular_weight`, `is_precursor`, `oh_reactant_idx`, `stoich`, `k_OH`) are `jax.numpy` arrays. Stoichiometry is a signed dense `(n_reactions, n_species)` matrix: `-1` at the reactant column, `+yield` at product columns. Constructable from an in-memory `Mechanism` (`SOMNetwork.from_mechanism`) or directly from committed JSON (`SOMNetwork.from_json`).
- `tests/conftest.py` enables `jax_enable_x64` for the test suite (scientific ODEs need float64).
- 12 additional unit tests for shape / dtype / stoichiometry correctness / PyTree flatten-unflatten / `jax.jit` compatibility. Total: 30 tests pass.
- **S1.6 — ODE right-hand side**: `som_jax.som_rhs(concentrations, oh, network)` — pure JAX function returning `dy/dt = stoich.T @ (k_OH * OH * y[reactant_idx])`. Unit-agnostic; caller maintains unit consistency. Linear in `oh`, bilinear in `(oh, concentrations)`; traces cleanly through `jax.jit`, `jax.grad`, and `jax.jacfwd`.
- 10 additional unit tests for the RHS: zero-concentration and zero-OH boundary cases, GENVOC-only activation (verified against hand-computed BL20 yields), single-grid-species activation, linearity in OH, `jit` vs eager parity (~1e-14 ULP-level tolerance), `grad` through OH and `jacfwd` through concentrations, cross-check vs an explicit Python loop. Total: **40 tests pass**.
- S1.6 figures under `docs/figures/s1.6/`: GENVOC-only dy/dt bar chart (first-gen activation pattern), uniform-state dy/dt (net-production/consumption structure), and full Jacobian `df/dy` heatmap (reaction-graph adjacency).
- **S1.7 — simulate() wrapper**: `som_jax.simulate(network, initial, oh, t_span, save_at)` — thin wrapper around `diffrax.Kvaerno5` with `PIDController` step size and `RecursiveCheckpointAdjoint`. Returns a `SOMTrajectory` PyTree (`.t`, `.y`, `.species_names`) with a `.y_of(name)` accessor. Ships with `build_initial(network, {name: value})` helper.
- Fixed a circular import between `som_jax/__init__.py` (which now re-exports `simulate`) and `som_jax.mechanism.parser` (which previously read `__version__` from the package root). Parser now reads the version directly via `importlib.metadata`.
- **S1.8 — analytic decay test**: `test_genvoc_first_order_decay_under_constant_oh` integrates GENVOC-only initial conditions under constant OH and asserts the trajectory matches `exp(-k_BL20 · OH · t)` within 1e-5 relative. Plus `test_genvoc_decay_scales_with_oh` verifying halving the decay time-constant when OH doubles. Total: **51 tests pass** (+11 for S1.7/S1.8 suite).
- Figures under `docs/figures/s1.7/` (all-species log-y trajectories, coloured by carbon number) and `docs/figures/s1.8/` (sim-vs-analytic overlay + residuals, test tolerance line).

### Notes
- Product yields per reaction do **not** uniformly sum to 1.0: on-grid reactions sum to `on_grid_branch + N_frag × fragmentation_branch`. Fragmentation reactions with multiple fragment products (e.g. S2.1, sum ≈ 2.0) are consistent with one parent generating multiple smaller fragments. Fortran-encoded zero-yield pathways (e.g. S3.1, S4.1) are preserved verbatim; they represent mass loss out of the tracked grid.
- **S1.figures-retrofit — visual diagnostics for S1.1 and S1.4**: `scripts/make_s1.1_figures.py` and `scripts/make_s1.4_figures.py` generate 6 committed PNGs under `docs/figures/`: (C, O) grid coverage, per-reaction yield-sum bar chart (regime-coloured), log-histogram of rate constants, full (39 × 41) signed stoichiometry heatmap, per-species reactant degree, per-species product degree. Matplotlib added to `[project.optional-dependencies].dev`.

## [0.1.0] — planned

First usable release. Gate: all 10 canonical Fortran goldens match within 0.1% relative L2 per species; `dLVP` recovery differentiability test passes.
