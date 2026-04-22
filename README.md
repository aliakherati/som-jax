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
| `som_jax.mechanism.parser` | Parse Fortran `.som`/`.mod`/`.doc` → JSON intermediate | not started |
| `som_jax.mechanism.network` | `SOMNetwork` PyTree (species, stoichiometry, rate constants) | not started |
| `som_jax.rhs` | ODE right-hand side: `stoich.T @ (k · OH · y[reactant_idx])` | not started |
| `som_jax.simulate` | Public `simulate(cfg, t_span, save_at)` using `diffrax.Kvaerno5` | not started |
| regression suite | Vs Fortran goldens at ≤0.1% relative per species | not started |
| differentiability suite | `dLVP` recovery demo via `optax.adam` | not started |

Tracked in the master plan as chunks `S1.0` … `S1.21`.

## Related packages

- [`atmos-jax-common`](https://github.com/aliakherati/atmos-jax-common) — shared Fortran-runner, goldens loader, unit helpers (dependency for regression tests once wired up).
- `saprc-jax` *(not yet created)* — full explicit gas-phase chemistry; `som-jax`'s network is a subset.
- `tomas-jax` *(not yet created)* — aerosol microphysics; consumes `som-jax` gas trajectories.

## Scientific references

**To be populated.** Intended citations (awaiting confirmation):

- Cappa & Wilson (2012) — Statistical Oxidation Model formulation.
- Cappa et al. (2013) / Jathar et al. (2014) — multigenerational SOM extension.
- Epstein et al. (2010) — `Hvap = -11·log10(c*) + 131`.

## License

MIT — see [LICENSE](LICENSE).
