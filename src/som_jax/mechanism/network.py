"""JAX-native representation of a parsed :class:`Mechanism`.

:class:`SOMNetwork` holds the same information as a :class:`Mechanism` but
reshaped for efficient use in a JAX ODE right-hand side:

- Species-name and reaction-label tuples are **static** (registered as PyTree
  aux data, so changing them triggers a JIT retrace rather than a silent
  numerical error).
- Per-species and per-reaction numeric quantities live as ``jax.numpy`` arrays
  (PyTree children, traced under ``jit``/``grad``).
- The stoichiometry is stored as a signed dense matrix ``(n_reactions, n_species)``
  where row *i* has ``-1`` at the reactant column and ``+yield`` at each product
  column. This form supports the RHS expression ``stoich.T @ rate_vector`` in a
  single ``matmul`` fusable by XLA.

Scope note: ``molecular_weight`` is total MW. ``MW_HC`` (hydrocarbon-only MW,
needed by the volatility formula in the Fortran reference) is intentionally
absent here — volatility belongs in ``tomas-jax`` per decision D2 in the master
plan; ``som-jax`` is gas-phase only.

References
----------
- Cappa, C. D. and Wilson, K. R. (2012). Multi-generation gas-phase oxidation,
  equilibrium partitioning, and the formation and evolution of secondary organic
  aerosol. *Atmos. Chem. Phys.*, 12, 8399-8411. doi:10.5194/acp-12-8399-2012.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from som_jax.mechanism.json_io import mechanism_from_json
from som_jax.mechanism.types import Mechanism


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class SOMNetwork:
    """JAX-native SOM reaction network.

    Shapes (``n = n_species, r = n_reactions``)

    - ``species_names``: ``tuple[str, n]``. Static — changes trigger JIT retrace.
    - ``reaction_labels``: ``tuple[str, r]``. Static.
    - ``carbon``, ``oxygen``: ``int32[n]``.
    - ``molecular_weight``: ``float[n]``, total MW in g/mol.
    - ``is_precursor``: ``bool_[n]``, ``True`` for the one precursor species
      (GENVOC for GENSOMG).
    - ``oh_reactant_idx``: ``int32[r]``. ``species_names[oh_reactant_idx[i]]`` is
      the SOM reactant of reaction *i* (the other reactant is always OH and is
      implicit — carrying its concentration is the caller's job).
    - ``stoich``: ``float[r, n]``. Signed stoichiometric coefficients:
      ``stoich[i, j] == -1`` if ``j`` is the reactant of reaction *i*, else
      ``+yield_j`` for each product, else 0.
    - ``k_OH``: ``float[r]``. Bimolecular rate constants, cm³ mol⁻¹ s⁻¹
      at 300 K as read from ``saprc14_rev1.doc``.

    Use :meth:`from_json` or :meth:`from_mechanism` to construct.
    """

    species_names: tuple[str, ...]
    reaction_labels: tuple[str, ...]
    carbon: Array
    oxygen: Array
    molecular_weight: Array
    is_precursor: Array
    oh_reactant_idx: Array
    stoich: Array
    k_OH: Array

    # --- basic accessors -------------------------------------------------

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_labels)

    def species_index(self, name: str) -> int:
        """Return the index of ``name`` in :attr:`species_names`."""
        try:
            return self.species_names.index(name)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"species {name!r} not in network") from exc

    def reaction_index(self, label: str) -> int:
        """Return the index of ``label`` in :attr:`reaction_labels`."""
        try:
            return self.reaction_labels.index(label)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"reaction {label!r} not in network") from exc

    # --- PyTree registration --------------------------------------------

    def tree_flatten(
        self,
    ) -> tuple[tuple[Array, ...], tuple[tuple[str, ...], tuple[str, ...]]]:
        children = (
            self.carbon,
            self.oxygen,
            self.molecular_weight,
            self.is_precursor,
            self.oh_reactant_idx,
            self.stoich,
            self.k_OH,
        )
        aux = (self.species_names, self.reaction_labels)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[tuple[str, ...], tuple[str, ...]],
        children: tuple[Array, ...],
    ) -> SOMNetwork:
        species_names, reaction_labels = aux
        (
            carbon,
            oxygen,
            molecular_weight,
            is_precursor,
            oh_reactant_idx,
            stoich,
            k_OH,
        ) = children
        return cls(
            species_names=species_names,
            reaction_labels=reaction_labels,
            carbon=carbon,
            oxygen=oxygen,
            molecular_weight=molecular_weight,
            is_precursor=is_precursor,
            oh_reactant_idx=oh_reactant_idx,
            stoich=stoich,
            k_OH=k_OH,
        )

    # --- factories ------------------------------------------------------

    @classmethod
    def from_mechanism(cls, mechanism: Mechanism) -> SOMNetwork:
        """Build a :class:`SOMNetwork` from a parsed :class:`Mechanism`.

        Raises
        ------
        ValueError
            If the mechanism has no species or no reactions.
        """
        if not mechanism.species:
            raise ValueError("mechanism has no species")
        if not mechanism.reactions:
            raise ValueError("mechanism has no reactions")

        name_to_idx = {sp.name: i for i, sp in enumerate(mechanism.species)}
        n_species = len(mechanism.species)
        n_reactions = len(mechanism.reactions)

        # Assemble in numpy (eager) then move to jnp arrays once at the end.
        # This keeps PyTree children contiguous and avoids per-element tracing.
        stoich = np.zeros((n_reactions, n_species), dtype=np.float64)
        oh_reactant_idx = np.zeros(n_reactions, dtype=np.int32)
        k_OH = np.zeros(n_reactions, dtype=np.float64)

        for i, rxn in enumerate(mechanism.reactions):
            reactant_name = rxn.reactants[0]  # always GENVOC or GENSOMG_xx_yy
            r_idx = name_to_idx[reactant_name]
            oh_reactant_idx[i] = r_idx
            stoich[i, r_idx] -= 1.0
            for product in rxn.products:
                p_idx = name_to_idx[product.name]
                # Use += so duplicate products in one reaction accumulate.
                stoich[i, p_idx] += product.yield_
            k_OH[i] = rxn.rate_cm3_per_mol_per_s

        carbon = np.array([sp.carbon for sp in mechanism.species], dtype=np.int32)
        oxygen = np.array([sp.oxygen for sp in mechanism.species], dtype=np.int32)
        mw = np.array([sp.molecular_weight for sp in mechanism.species], dtype=np.float64)
        is_precursor = np.array([sp.is_precursor for sp in mechanism.species], dtype=np.bool_)

        return cls(
            species_names=tuple(sp.name for sp in mechanism.species),
            reaction_labels=tuple(rxn.label for rxn in mechanism.reactions),
            carbon=jnp.asarray(carbon),
            oxygen=jnp.asarray(oxygen),
            molecular_weight=jnp.asarray(mw),
            is_precursor=jnp.asarray(is_precursor),
            oh_reactant_idx=jnp.asarray(oh_reactant_idx),
            stoich=jnp.asarray(stoich),
            k_OH=jnp.asarray(k_OH),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> SOMNetwork:
        """Load a network from the committed mechanism JSON.

        Convenience wrapper: :func:`mechanism_from_json` then
        :meth:`from_mechanism`.
        """
        return cls.from_mechanism(mechanism_from_json(path))
