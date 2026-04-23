# som-jax

Differentiable Python/JAX implementation of the **Statistical Oxidation Model (SOM)** — a parameterization of multigenerational OH oxidation of organic precursors onto a carbon/oxygen product grid. Port of a Fortran reference (SOM as embedded in SAPRC-14).

**Why a port?** JAX gives us:

- `jax.grad` / `optax` through the full simulator → parameter estimation against chamber data (e.g., `dLVP`, fragmentation probability, per-reaction rate scaling).
- JIT compilation + GPU capability without a Fortran toolchain.
- Composability with the sibling packages (`saprc-jax`, `tomas-jax`) for full-box atmospheric simulations.

## Status

**Early alpha — scaffolding only.** Current state:

- Repository created, package skeleton installable.
- CI configured (pytest + ruff + mypy).
- No public API yet. Reaction network, ODE solver, and simulate API land incrementally.

## Scope

`som-jax` is **gas-phase only**. It integrates the SOM oxidation network under a prescribed OH(t) trajectory; it does not partition products into the condensed phase. Partitioning (equilibrium or kinetic) lives in `tomas-jax` once that repo comes online.

Target species family (v1): `GENSOMG` — 47 species on a 7×7 (C, O) grid, 38 OH-reactions. `AR1SOMG` (alkylaromatic SOM) is a follow-up.

## Install

```bash
git clone https://github.com/aliakherati/som-jax.git
cd som-jax
pip install -e ".[dev]"
```

Requires Python ≥ 3.11.

## Project status

| Module | Purpose | Status |
|---|---|---|
| `som_jax.mechanism.parser` | Parse Fortran `.som`/`.mod`/`.doc` → typed `Mechanism` | alpha (S1.1) |
| `som_jax.mechanism.types` | `Mechanism`, `Species`, `Reaction`, `Product` dataclasses | alpha (S1.1) |
| `som_jax.mechanism.json_io` | Deterministic JSON serialisation for committed mechanism | alpha (S1.1) |
| `data/mechanisms/gensomg.json` | Committed GENSOMG network (41 species, 39 reactions) | alpha (S1.1) |
| `som_jax.mechanism.network` | `SOMNetwork` PyTree (dense stoichiometry, rate constants as `jax.numpy`) | alpha (S1.4) |
| `som_jax.rhs` | ODE right-hand side: `stoich.T @ (k · OH · y[reactant_idx])` | alpha (S1.6) |
| `som_jax.simulate` | Public `simulate(network, initial, oh, t_span, save_at)` using `diffrax.Kvaerno5` | alpha (S1.7) |
| `som_jax.build_initial` | Helper: `{name: value}` dict → `(n_species,)` initial-condition array | alpha (S1.7) |
| `SOMTrajectory` | PyTree wrapping `(t, y, species_names)` with a `.y_of(name)` accessor | alpha (S1.7) |
| analytic first-order decay test | `[GENVOC](t) = exp(-k_BL20 · OH · t)` within 1e-5 relative (S1.8 headline) | alpha (S1.8) |
| regression suite | Vs Fortran goldens at ≤0.1% relative per species | not started (S1.10–S1.11) |
| differentiability suite | `dLVP` recovery demo via `optax.adam` | not started (S1.17) |

Tracked in the master plan as chunks `S1.0` … `S1.21`.

### Figures

Each scientific chunk ships matplotlib figures under `docs/figures/<chunk-id>/`, regenerable via `scripts/make_<chunk>_figures.py`.

**S1.1 — parsed mechanism**

| File | What it shows |
|---|---|
| [`docs/figures/s1.1/cover_grid.png`](docs/figures/s1.1/cover_grid.png) | Which (carbon, oxygen) cells of the 7×7 grid actually contain species. Precursor GENVOC at (C=7, O=0) flagged in red. |
| [`docs/figures/s1.1/yield_sum_per_reaction.png`](docs/figures/s1.1/yield_sum_per_reaction.png) | Σ(branch × oxy_yield) per reaction, coloured by regime — on-grid (~1), fragmentation (>1), or zero-yield mass-loss pathway. |
| [`docs/figures/s1.1/rate_constants.png`](docs/figures/s1.1/rate_constants.png) | Log-histogram of `k_OH` at 300 K across all 39 reactions, with BL20 (GENVOC+OH) annotated. |

**S1.4 — SOMNetwork**

| File | What it shows |
|---|---|
| [`docs/figures/s1.4/stoich_heatmap.png`](docs/figures/s1.4/stoich_heatmap.png) | Full (39 × 41) signed stoichiometry matrix. Blue = reactant (−1), red = product yield. |
| [`docs/figures/s1.4/reactant_degree.png`](docs/figures/s1.4/reactant_degree.png) | How many reactions each species drives as a reactant. Verifies that every grid species is consumed by exactly one reaction. |
| [`docs/figures/s1.4/product_degree.png`](docs/figures/s1.4/product_degree.png) | How many reactions each species appears in as a product. High-connectivity species (GENSOMG_01_*, 02_*, 03_*) reflect fragmentation routing. |

**S1.6 — RHS behaviour**

| File | What it shows |
|---|---|
| [`docs/figures/s1.6/genvoc_only_dydt.png`](docs/figures/s1.6/genvoc_only_dydt.png) | `dy/dt` at `t=0` when only GENVOC is nonzero. Exactly five species light up (BL20's reactant and its four first-gen products), confirming the activation pattern. |
| [`docs/figures/s1.6/uniform_state_dydt.png`](docs/figures/s1.6/uniform_state_dydt.png) | `dy/dt` at a uniform concentration. Shows which species are net produced vs net consumed when the whole grid is equally populated — a sanity check on coupling directions. |
| [`docs/figures/s1.6/jacobian_structure.png`](docs/figures/s1.6/jacobian_structure.png) | `df/dy` Jacobian at a uniform state. Diagonal is self-consumption; off-diagonal reds expose the reaction-graph's adjacency structure. |

**S1.7 — integrated trajectories**

| File | What it shows |
|---|---|
| [`docs/figures/s1.7/all_species_trajectories.png`](docs/figures/s1.7/all_species_trajectories.png) | All 41 species under constant OH over a 4 e-fold integration starting from GENVOC-only. Species coloured by carbon number — the oxidation cascade is visible as the brief early transient of first-gen (C=7) products and the slower accumulation of fragmentation products at lower C. |

**S1.8 — analytic-decay correctness**

| File | What it shows |
|---|---|
| [`docs/figures/s1.8/genvoc_decay_vs_analytic.png`](docs/figures/s1.8/genvoc_decay_vs_analytic.png) | Two-panel. Top: simulated GENVOC(t) overlaid on the exact analytic `exp(-k_BL20·OH·t)`. Bottom: relative error, oscillating between ~10⁻¹¹ and ~10⁻⁸ — well below the 10⁻⁵ test tolerance. |

To regenerate any chunk's figures: `python scripts/make_<chunk>_figures.py` (requires `pip install -e ".[dev]"`).

### Regenerating the mechanism JSON

`data/mechanisms/gensomg.json` is committed; normal development does not need to touch it. If the Fortran mechanism files change, regenerate with:

```bash
python scripts/generate_mechanism_json.py \
    --mod ../som-tomas-app/src/saprc14_rev1.mod \
    --doc ../som-tomas-app/src/saprc14_rev1.doc \
    --som ../som-tomas-app/src/saprc14_rev1.som \
    --family GENSOMG \
    --output data/mechanisms/gensomg.json
```

The resulting JSON embeds SHA-256 digests of the source files so regressions can pinpoint mechanism drift.

## Related packages

- [`atmos-jax-common`](https://github.com/aliakherati/atmos-jax-common) — shared Fortran-runner, goldens loader, unit helpers (dependency for regression tests once wired up).
- `saprc-jax` *(not yet created)* — full explicit gas-phase chemistry; `som-jax`'s network is a subset.
- `tomas-jax` *(not yet created)* — aerosol microphysics; consumes `som-jax` gas trajectories.

## Scientific references

BibTeX in [`docs/references.bib`](docs/references.bib); grows as modules land.

Current (S1.1):

- Cappa, C. D. and Wilson, K. R. (2012). Multi-generation gas-phase oxidation, equilibrium partitioning, and the formation and evolution of secondary organic aerosol. *Atmos. Chem. Phys.*, 12, 8399–8411. [doi:10.5194/acp-12-8399-2012](https://doi.org/10.5194/acp-12-8399-2012).
- Cappa, C. D. et al. (2013). Application of the Statistical Oxidation Model (SOM) to Secondary Organic Aerosol formation from photooxidation of C12 alkanes. *Atmos. Chem. Phys.*, 13, 1591–1606. [doi:10.5194/acp-13-1591-2013](https://doi.org/10.5194/acp-13-1591-2013).

Planned (later chunks):
- Epstein, Riipinen, Donahue (2010) — `Hvap = -11·log10(c*) + 131` (volatility module).
- Pankow & Asher (2008) — SIMPOL.1 (volatility module).

## License

MIT — see [LICENSE](LICENSE).
