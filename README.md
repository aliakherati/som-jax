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
| `som_jax.rhs` | ODE right-hand side: `stoich.T @ (k · OH · y[reactant_idx])` | not started (S1.6) |
| `som_jax.simulate` | Public `simulate(cfg, t_span, save_at)` using `diffrax.Kvaerno5` | not started (S1.7) |
| regression suite | Vs Fortran goldens at ≤0.1% relative per species | not started (S1.10–S1.11) |
| differentiability suite | `dLVP` recovery demo via `optax.adam` | not started (S1.17) |

Tracked in the master plan as chunks `S1.0` … `S1.21`.

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
