"""Typed dataclasses for the SOM mechanism.

A :class:`Mechanism` is the result of parsing the Fortran SAPRC-14 source files
(``.som``, ``.mod``, ``.doc``) for a single SOM family (e.g. ``GENSOMG``).

References
----------
- Cappa, C. D. and Wilson, K. R. (2012). Multi-generation gas-phase oxidation,
  equilibrium partitioning, and the formation and evolution of secondary organic
  aerosol. *Atmos. Chem. Phys.*, 12, 8399-8411. doi:10.5194/acp-12-8399-2012.
- Cappa et al. (2013). Application of the Statistical Oxidation Model (SOM) to
  photooxidation of C12 alkanes. *Atmos. Chem. Phys.*, 13, 1591-1606.
  doi:10.5194/acp-13-1591-2013.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Species:
    """A SOM species: precursor (e.g. GENVOC) or product (e.g. GENSOMG_04_03).

    Parameters
    ----------
    name
        Species identifier as it appears in the Fortran files.
    carbon
        Number of carbon atoms per molecule.
    oxygen
        Number of oxygen atoms per molecule.
    molecular_weight
        Total molecular weight in g/mol, read from the ``.mod`` species card.
    is_precursor
        ``True`` if the species is the parent hydrocarbon that seeds the
        oxidation chain (e.g. ``GENVOC``). ``False`` for SOM products.
    source_line_mod
        1-based line number in ``saprc14_rev1.mod`` where this species card
        appears. Preserved for auditing; not used at runtime.
    """

    name: str
    carbon: int
    oxygen: int
    molecular_weight: float
    is_precursor: bool
    source_line_mod: int


@dataclass(frozen=True, slots=True)
class Product:
    """A single product appearing on the RHS of a SOM reaction.

    Fortran ``.mod`` encodes each SOM product with two ``#`` coefficients:
    ``#<branch> #<oxy_yield> NAME``. The effective stoichiometric yield
    is the product ``branch * oxy_yield``.

    For a reaction with on-grid branch ``p`` and fragmentation branch ``1-p``,
    on-grid products have ``branch == p`` and ``oxy_yield`` summing to 1.0
    across the on-grid products; fragment products have ``branch == 1-p`` and
    separate ``oxy_yield`` bookkeeping per fragment class.

    Parameters
    ----------
    name
        Product species name (must exist in :attr:`Mechanism.species`).
    branch
        On-grid/fragment branching fraction for this product.
    oxy_yield
        Per-branch yield for this product (e.g. distribution over adjacent
        oxygen columns for on-grid products).
    """

    name: str
    branch: float
    oxy_yield: float

    @property
    def yield_(self) -> float:
        """Effective stoichiometric yield ``branch * oxy_yield``."""
        return self.branch * self.oxy_yield


@dataclass(frozen=True, slots=True)
class Reaction:
    """A SOM oxidation reaction (currently always ``X + OH``).

    Parameters
    ----------
    label
        Rate-constant identifier in ``saprc14_rev1.doc`` (``BL20`` for the
        GENVOC+OH entry point, ``S1.1`` ... ``S38.1`` for the grid reactions).
    reactants
        LHS species names, always ``[reactant, "OH"]`` for SOM.
    products
        RHS :class:`Product` list.
    rate_cm3_per_mol_per_s
        Bimolecular rate constant at 300 K in cm³ mol⁻¹ s⁻¹, as read from
        ``.doc``. For Arrhenius forms, this is the evaluated value at 300 K
        rather than the pre-exponential ``A``.
    source_line_mod
        1-based line number in ``.mod`` where this reaction begins (``T`` line).
    source_line_doc
        1-based line number in ``.doc`` where the rate label appears.
    """

    label: str
    reactants: tuple[str, ...]
    products: tuple[Product, ...]
    rate_cm3_per_mol_per_s: float
    source_line_mod: int
    source_line_doc: int

    @property
    def total_yield(self) -> float:
        """Sum of effective yields over all products.

        For non-fragmenting reactions this equals 1.0. For reactions with a
        fragmentation branch producing *N* fragment products, the sum is
        ``p_on_grid + (1 - p_on_grid) * N_frag``, which exceeds 1.0 because
        one parent molecule generates multiple smaller fragments.
        """
        return sum(p.yield_ for p in self.products)


@dataclass(frozen=True, slots=True)
class SourceFileRef:
    """Provenance record for one Fortran source file read by the parser."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class MechanismMetadata:
    """Audit metadata emitted alongside the parsed mechanism."""

    parser_version: str
    parsed_at_utc: str
    source_files: tuple[SourceFileRef, ...]


@dataclass(frozen=True, slots=True)
class Mechanism:
    """Parsed SOM mechanism for a single family (e.g. GENSOMG).

    Parameters
    ----------
    family
        Family identifier (``"GENSOMG"`` or ``"AR1SOMG"``).
    precursor
        Name of the parent precursor (``"GENVOC"`` for GENSOMG).
    grid_c_max, grid_o_max
        Maximum carbon and oxygen indices in the product grid. For GENSOMG,
        both are 7 (indices 1..7 for C, 1..7 for O; 47 species in total after
        excluding the non-present (C=1, O=0) point).
    species
        All species in the mechanism, precursor first, then products in
        lexicographic order of their names.
    reactions
        All OH-driven reactions, in the order they appear in ``.mod`` (GENVOC+OH
        first, then GENSOMG_XX_YY + OH in ``.mod`` file order). Reaction index
        matches the ``.doc`` rate label sequence for the grid reactions.
    metadata
        Provenance and parser version information.
    """

    family: str
    precursor: str
    grid_c_max: int
    grid_o_max: int
    species: tuple[Species, ...] = field(default_factory=tuple)
    reactions: tuple[Reaction, ...] = field(default_factory=tuple)
    metadata: MechanismMetadata | None = None

    def species_index(self, name: str) -> int:
        """Return the position of ``name`` in :attr:`species`.

        Raises
        ------
        KeyError
            If ``name`` is not a species in the mechanism.
        """
        for i, sp in enumerate(self.species):
            if sp.name == name:
                return i
        raise KeyError(f"species {name!r} not found in mechanism {self.family!r}")

    def species_by_name(self, name: str) -> Species:
        """Return the :class:`Species` named ``name``."""
        return self.species[self.species_index(name)]
