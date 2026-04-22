"""Parse Fortran SAPRC-14 mechanism files into a typed :class:`Mechanism`.

The SOM reactions in the Fortran reference live in three files:

- ``saprc14_rev1.mod`` — compiled mechanism. Holds the species cards for the
  GENSOMG family and the ``T``/``F`` continuation lines that encode each
  reaction's stoichiometry with two ``#`` coefficients per product
  (``#<branch> #<oxy_yield> NAME``).
- ``saprc14_rev1.doc`` — human-readable listing of rate constants, with labels
  (``BL20`` for GENVOC+OH; ``S1.1`` … ``S38.1`` for the GENSOMG OH-reactions).
- ``saprc14_rev1.som`` — per-species cards for the *AR1SOMG* family. Not used
  by the GENSOMG parser; kept here for future AR1SOMG support.

The parser treats species-name indexing as authoritative; integer offsets in
the Fortran mechanism file (see ``box.f:445``, ``OFFSET=13``) are never
propagated.

References
----------
- Cappa, C. D. and Wilson, K. R. (2012). Multi-generation gas-phase oxidation,
  equilibrium partitioning, and the formation and evolution of secondary organic
  aerosol. *Atmos. Chem. Phys.*, 12, 8399-8411. doi:10.5194/acp-12-8399-2012.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from pathlib import Path

from som_jax import __version__ as _package_version
from som_jax.mechanism.types import (
    Mechanism,
    MechanismMetadata,
    Product,
    Reaction,
    SourceFileRef,
    Species,
)

# --- regexes --------------------------------------------------------------

# Species card in .mod: "GENSOMG_04_03    103.00  4.00  0  0  3.00  1.00  0.000E+00"
# We only need: name, MW, carbon count, oxygen count. N and S are always 0 for SOM.
_SPECIES_CARD_RE = re.compile(
    r"""^
    (?P<name>GENSOMG_\d\d_\d\d)          # GENSOMG_CC_OO
    \s+(?P<mw>\d+\.\d+)                  # molecular weight
    \s+(?P<c>\d+\.\d+)                   # carbon count
    \s+\d+                               # N atoms (always 0 for SOM)
    \s+\d+                               # S atoms (always 0 for SOM)
    \s+(?P<o>\d+\.\d+)                   # oxygen count (xno field)
    \s+
    """,
    re.VERBOSE,
)

# Precursor species card (GENVOC lives earlier in the file, different format).
# Format: "GENVOC            92.14  7.00  0  0  0.00  0.00  0.000E+00"
_PRECURSOR_CARD_RE = re.compile(
    r"""^
    (?P<name>GENVOC)
    \s+(?P<mw>\d+\.\d+)
    \s+(?P<c>\d+\.\d+)
    \s+\d+
    \s+\d+
    \s+(?P<o>\d+\.\d+)
    \s+
    """,
    re.VERBOSE,
)

# Rate-constant line in .doc.
# Format: "  285  BL20     8.319E+03  (...)   GENVOC + OH = ..."
#         "  359  S1.1     1.759E+03            GENSOMG_02_01 + OH = ..."
_RATE_LINE_RE = re.compile(
    r"""^
    \s*(?P<rxn_idx>\d+)                  # reaction sequence number in .doc
    \s+(?P<label>[A-Z][A-Z0-9.]*)        # label (BL20, S1.1, S38.1, etc.)
    \s+(?P<rate>[+\-]?\d+\.?\d*[eE][+\-]?\d+)
    """,
    re.VERBOSE,
)

# Identifies a SOM reaction we care about: GENVOC+OH or GENSOMG+OH on the LHS.
_SOM_REACTANT_RE = re.compile(r"^(GENVOC|GENSOMG_\d\d_\d\d)\s*\+\s*OH\b")


# --- helpers --------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _join_reaction_lines(raw: str) -> str:
    """Collapse ``.mod`` Fortran continuation lines into a single string.

    A reaction spans several physical lines joined by ``&`` at the end of a line
    and ``F`` at the start of the next. :func:`_extract_som_reaction_blocks`
    already strips the leading ``F ``, so this function only needs to remove
    the ``&`` markers and collapse surrounding whitespace.
    """
    # Drop any standalone or trailing "&" marker. `\b&\b` doesn't quite work
    # because "&" has no word character on either side in Python's regex, so
    # we match it with surrounding whitespace or line boundaries instead.
    without_amp = re.sub(r"\s*&\s*", " ", raw)
    return " ".join(without_amp.split())


def _parse_products(rhs: str) -> tuple[Product, ...]:
    """Tokenise a reaction RHS into typed :class:`Product` entries.

    Supports three coefficient shapes:

    - ``#<branch> #<oxy_yield> NAME`` — SOM two-coefficient form (the form
      used by every GENSOMG+OH reaction and by GENVOC+OH).
    - ``#<coef> NAME`` — standard SAPRC single-coefficient form.
    - ``NAME`` — bare species, coefficient 1.

    Parameters
    ----------
    rhs
        The right-hand side of the ``=`` sign, already line-joined.
    """
    # Tokens are separated by " + ". Leading/trailing whitespace already stripped.
    products: list[Product] = []
    for raw_term in rhs.split(" + "):
        term = raw_term.strip()
        if not term:
            continue
        tokens = term.split()
        # Collect leading "#<value>" coefficients, then the species name.
        coeffs: list[float] = []
        name: str | None = None
        for tok in tokens:
            if tok.startswith("#"):
                coeffs.append(float(tok[1:]))
            else:
                name = tok
                break
        if name is None:
            raise ValueError(f"cannot parse product term: {term!r}")
        if len(coeffs) == 0:
            branch, oxy_yield = 1.0, 1.0
        elif len(coeffs) == 1:
            branch, oxy_yield = coeffs[0], 1.0
        elif len(coeffs) == 2:
            branch, oxy_yield = coeffs[0], coeffs[1]
        else:
            raise ValueError(
                f"expected at most two '#' coefficients in product term, got {len(coeffs)} "
                f"in {term!r}"
            )
        products.append(Product(name=name, branch=branch, oxy_yield=oxy_yield))
    return tuple(products)


def _parse_species(mod_text: str) -> tuple[list[Species], int]:
    """Scan ``.mod`` for GENVOC + all GENSOMG species cards.

    Returns
    -------
    species
        Precursor first (``GENVOC``), then GENSOMG products in the order they
        appear in the file (which is lexicographic).
    c_max, o_max
        Maximum carbon / oxygen indices observed. Returned together for the
        caller; here we return ``c_max`` only.  (``o_max`` is extracted from
        the species list by the caller.)
    """
    species: list[Species] = []
    c_max = 0
    for lineno, line in enumerate(mod_text.splitlines(), start=1):
        m_pre = _PRECURSOR_CARD_RE.match(line)
        if m_pre is not None:
            species.append(
                Species(
                    name=m_pre.group("name"),
                    carbon=int(float(m_pre.group("c"))),
                    oxygen=int(float(m_pre.group("o"))),
                    molecular_weight=float(m_pre.group("mw")),
                    is_precursor=True,
                    source_line_mod=lineno,
                )
            )
            c_max = max(c_max, int(float(m_pre.group("c"))))
            continue
        m_sp = _SPECIES_CARD_RE.match(line)
        if m_sp is not None:
            carbon = int(float(m_sp.group("c")))
            oxygen = int(float(m_sp.group("o")))
            species.append(
                Species(
                    name=m_sp.group("name"),
                    carbon=carbon,
                    oxygen=oxygen,
                    molecular_weight=float(m_sp.group("mw")),
                    is_precursor=False,
                    source_line_mod=lineno,
                )
            )
            c_max = max(c_max, carbon)
    return species, c_max


def _extract_som_reaction_blocks(mod_text: str) -> list[tuple[int, str]]:
    """Return ``(source_line, joined_text)`` for every SOM reaction in ``.mod``.

    A SOM reaction is any ``T …`` block whose LHS matches :data:`_SOM_REACTANT_RE`
    (``GENVOC + OH`` or ``GENSOMG_CC_OO + OH``). Continuation lines (``F …``)
    following the ``T`` line, delimited by ``&``, are merged into one string.
    """
    lines = mod_text.splitlines()
    blocks: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("T "):
            lhs_body = line[2:].lstrip()
            if _SOM_REACTANT_RE.match(lhs_body):
                # Accumulate this T-line and any following F continuation lines.
                parts = [line[2:]]
                while parts[-1].rstrip().endswith("&"):
                    i += 1
                    if i >= len(lines):
                        break
                    cont = lines[i]
                    if not cont.startswith("F "):
                        break
                    parts.append(cont[2:])
                joined = _join_reaction_lines("\n".join(parts))
                # Strip a trailing "&" if the loop exited for any reason.
                joined = joined.rstrip().rstrip("&").rstrip()
                blocks.append((i + 1 - len(parts) + 1, joined))
        i += 1
    return blocks


def _parse_reactions_without_rates(
    mod_text: str,
) -> list[tuple[int, tuple[str, ...], tuple[Product, ...]]]:
    """Return ``(source_line, reactants, products)`` tuples for every SOM reaction.

    Rates are filled in by :func:`_parse_rates` and assembled in
    :func:`parse_mechanism`.
    """
    out: list[tuple[int, tuple[str, ...], tuple[Product, ...]]] = []
    for source_line, joined in _extract_som_reaction_blocks(mod_text):
        if "=" not in joined:
            raise ValueError(f"reaction block at .mod line {source_line} lacks '='")
        lhs, rhs = (s.strip() for s in joined.split("=", 1))
        reactants = tuple(s.strip() for s in lhs.split("+"))
        products = _parse_products(rhs)
        out.append((source_line, reactants, products))
    return out


def _parse_rates(doc_text: str) -> tuple[dict[str, tuple[int, float]], dict[str, str]]:
    """Extract rate constants for BL20 (GENVOC+OH) and S{1..38}.1 (GENSOMG).

    Returns
    -------
    rates
        ``{label: (source_line, k_cm3_per_mol_per_s)}``. Labels are ``"BL20"``
        and ``"S1.1"`` … ``"S38.1"``.
    label_sequence
        ``{ordinal: label}`` giving the order ``S1.1, S2.1, …`` as they appear
        in ``.doc``. Callers use this to align rates with reactions parsed from
        ``.mod`` by position.
    """
    rates: dict[str, tuple[int, float]] = {}
    s_order: list[tuple[int, str]] = []  # (rxn_idx_from_doc, label)
    for lineno, line in enumerate(doc_text.splitlines(), start=1):
        m = _RATE_LINE_RE.match(line)
        if m is None:
            continue
        label = m.group("label")
        if label == "BL20" or re.fullmatch(r"S\d+\.1", label):
            rate = float(m.group("rate"))
            rates[label] = (lineno, rate)
            if label.startswith("S"):
                s_order.append((int(m.group("rxn_idx")), label))
    # Preserve .doc ordering for S labels.
    s_order.sort()
    label_sequence = {f"pos{i}": label for i, (_, label) in enumerate(s_order)}
    return rates, label_sequence


# --- public API -----------------------------------------------------------


def parse_mechanism(
    mod_path: Path | str,
    doc_path: Path | str,
    som_path: Path | str | None = None,
    *,
    family: str = "GENSOMG",
) -> Mechanism:
    """Parse the Fortran SOM mechanism for ``family`` into a :class:`Mechanism`.

    Parameters
    ----------
    mod_path
        Path to ``saprc14_rev1.mod``.
    doc_path
        Path to ``saprc14_rev1.doc``.
    som_path
        Path to ``saprc14_rev1.som``. Only required for AR1SOMG and above;
        ignored for GENSOMG. Still recorded in metadata if provided.
    family
        SOM family to extract. Only ``"GENSOMG"`` is supported at present; the
        data model is ready for ``"AR1SOMG"`` as a follow-up.

    Raises
    ------
    ValueError
        On structural inconsistencies (unknown product species, missing rate
        label, incorrect number of rate labels for the reaction set).
    """
    if family != "GENSOMG":
        raise NotImplementedError(f"family {family!r} not yet supported; use 'GENSOMG'")

    mod_path = Path(mod_path)
    doc_path = Path(doc_path)
    mod_text = mod_path.read_text(encoding="utf-8")
    doc_text = doc_path.read_text(encoding="utf-8")

    species_list, c_max = _parse_species(mod_text)
    o_max = max((sp.oxygen for sp in species_list), default=0)
    known_names = {sp.name for sp in species_list}

    raw_reactions = _parse_reactions_without_rates(mod_text)

    # Validate that every product species is known.
    for source_line, reactants, products in raw_reactions:
        for p in products:
            if p.name not in known_names:
                raise ValueError(
                    f"reaction at .mod line {source_line} references unknown product "
                    f"{p.name!r} (reactants={reactants})"
                )

    rates, _s_sequence = _parse_rates(doc_text)
    if "BL20" not in rates:
        raise ValueError("rate label BL20 (GENVOC+OH) not found in .doc")
    # .doc lists rates labeled S1.1 … S38.1 for the GENSOMG grid reactions.
    expected_s = len(raw_reactions) - 1  # minus 1 for GENVOC+OH
    for i in range(1, expected_s + 1):
        if f"S{i}.1" not in rates:
            raise ValueError(f"rate label S{i}.1 missing from .doc")

    # Assemble Reaction objects in .mod order, pairing each with its rate label.
    reactions: list[Reaction] = []
    s_index = 1
    for source_line_mod, reactants, products in raw_reactions:
        head = reactants[0].strip()
        if head == "GENVOC":
            label = "BL20"
        else:
            label = f"S{s_index}.1"
            s_index += 1
        source_line_doc, rate = rates[label]
        reactions.append(
            Reaction(
                label=label,
                reactants=tuple(r.strip() for r in reactants),
                products=products,
                rate_cm3_per_mol_per_s=rate,
                source_line_mod=source_line_mod,
                source_line_doc=source_line_doc,
            )
        )

    source_refs = [
        SourceFileRef(path=mod_path.name, sha256=_sha256_of_file(mod_path)),
        SourceFileRef(path=doc_path.name, sha256=_sha256_of_file(doc_path)),
    ]
    if som_path is not None:
        som_path = Path(som_path)
        source_refs.append(SourceFileRef(path=som_path.name, sha256=_sha256_of_file(som_path)))

    metadata = MechanismMetadata(
        parser_version=_package_version,
        parsed_at_utc=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        source_files=tuple(source_refs),
    )

    return Mechanism(
        family=family,
        precursor="GENVOC",
        grid_c_max=c_max,
        grid_o_max=o_max,
        species=tuple(species_list),
        reactions=tuple(reactions),
        metadata=metadata,
    )
