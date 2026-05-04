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
from typing import ClassVar

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
    # Arrhenius parametrisation per reaction. For reactions listed as
    # constant rates in ``.doc`` (no Arrhenius triplet), we set
    # ``arrhenius_a = k_OH``, ``arrhenius_ea = 0``, ``arrhenius_b = 0`` so
    # ``k_OH_at(T)`` is T-independent and equal to ``k_OH``. Reactions with
    # an explicit Arrhenius triplet (currently only BL20 in GENSOMG) carry
    # the values from ``.doc``. See :meth:`k_OH_at`.
    arrhenius_a: Array
    arrhenius_ea_kcal_per_mol: Array
    arrhenius_b: Array

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

    # --- temperature-dependent rates -----------------------------------

    # SAPRC convention from saprc14_rev1.doc:
    #     k(T) = A * exp(-Ea / (R * T)) * (T / T_ref) ** B
    # with R = 0.0019872 kcal/mol/K and T_ref = 300 K. For SOM reactions
    # the .doc lists Ea = 0 throughout, so the exponential factor is
    # always 1 and only the (T / T_ref) ** B power-law term varies. We
    # implement the standard form for forward-compatibility with future
    # SAPRC reactions that carry non-zero Ea.
    R_KCAL_PER_MOL_PER_K: ClassVar[float] = 0.0019872
    T_REF_K: ClassVar[float] = 300.0

    def k_OH_at(self, temperature_K: float | Array) -> Array:
        """Return per-reaction rate constants evaluated at ``temperature_K``.

        ``T`` may be a Python float or a JAX array (scalar). Differentiable
        through ``jax.grad`` w.r.t. ``T``. For reactions with no Arrhenius
        triplet in the source mechanism (the SOM cascade), the stored
        ``arrhenius_a = k_OH`` and ``arrhenius_b = 0`` make the result
        T-independent and equal to :attr:`k_OH`.
        """
        T = jnp.asarray(temperature_K, dtype=self.k_OH.dtype)
        exp_term = jnp.exp(-self.arrhenius_ea_kcal_per_mol / (self.R_KCAL_PER_MOL_PER_K * T))
        power_term = (T / self.T_REF_K) ** self.arrhenius_b
        return self.arrhenius_a * exp_term * power_term

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
            self.arrhenius_a,
            self.arrhenius_ea_kcal_per_mol,
            self.arrhenius_b,
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
            arrhenius_a,
            arrhenius_ea_kcal_per_mol,
            arrhenius_b,
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
            arrhenius_a=arrhenius_a,
            arrhenius_ea_kcal_per_mol=arrhenius_ea_kcal_per_mol,
            arrhenius_b=arrhenius_b,
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
        arrhenius_a = np.zeros(n_reactions, dtype=np.float64)
        arrhenius_ea = np.zeros(n_reactions, dtype=np.float64)
        arrhenius_b = np.zeros(n_reactions, dtype=np.float64)

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
            # If a reaction has no Arrhenius triplet in .doc (constant rate),
            # we set A = k(298) with B = 0 so k_OH_at(T) is T-independent
            # and equal to the listed k. Reactions with the triplet (e.g.
            # BL20) carry the A / Ea / B values from the .doc.
            if rxn.arrhenius_a is None:
                arrhenius_a[i] = rxn.rate_cm3_per_mol_per_s
                arrhenius_ea[i] = 0.0
                arrhenius_b[i] = 0.0
            else:
                arrhenius_a[i] = rxn.arrhenius_a
                arrhenius_ea[i] = rxn.arrhenius_ea_kcal_per_mol or 0.0
                arrhenius_b[i] = rxn.arrhenius_b or 0.0

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
            arrhenius_a=jnp.asarray(arrhenius_a),
            arrhenius_ea_kcal_per_mol=jnp.asarray(arrhenius_ea),
            arrhenius_b=jnp.asarray(arrhenius_b),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> SOMNetwork:
        """Load a network from the committed mechanism JSON.

        Convenience wrapper: :func:`mechanism_from_json` then
        :meth:`from_mechanism`.
        """
        return cls.from_mechanism(mechanism_from_json(path))
