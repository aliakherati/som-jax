"""Right-hand side of the SOM ODE.

Given a :class:`SOMNetwork`, :func:`som_rhs` returns ``dy/dt`` at a point
``(concentrations, OH)`` as a pure, JIT- and ``grad``-compatible function.

The solver wrapper (see ``som_jax.simulate``, S1.7) calls this with time-
dependent OH trajectories; this module is deliberately unit-agnostic and
unaware of time-stepping — those concerns live one level up.

Rate expression
---------------
For reaction *i* with reactant species index ``r_i``:

.. math::
   \\text{rate}_i = k_{\\text{OH},i} \\cdot [\\text{OH}] \\cdot [y_{r_i}]

And the derivative for every species *j* is the stoichiometry-weighted sum:

.. math::
   \\frac{dy_j}{dt} = \\sum_i S_{ij} \\cdot \\text{rate}_i

where :math:`S` is the signed :attr:`SOMNetwork.stoich` matrix
(``-1`` at reactant columns, ``+yield`` at product columns). In code this is
a single ``stoich.T @ rate_vec`` matmul, fusable by XLA.

Unit conventions
----------------
The module does **not** assume a particular unit system. If concentrations are
in ppm and ``OH`` is in ppm, then ``k_OH`` must be in ppm^-1 * (time unit)^-1
and the returned derivative is in ppm / (time unit). Callers are responsible
for unit consistency; the simulate wrapper documents a canonical choice.

References
----------
- Cappa, C. D. and Wilson, K. R. (2012). Multi-generation gas-phase oxidation,
  equilibrium partitioning, and the formation and evolution of secondary organic
  aerosol. *Atmos. Chem. Phys.*, 12, 8399-8411. doi:10.5194/acp-12-8399-2012.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax import Array

if TYPE_CHECKING:
    from som_jax.mechanism.network import SOMNetwork


def som_rhs(
    concentrations: Array,
    oh: Array | float,
    network: SOMNetwork,
) -> Array:
    """Return ``dy/dt`` at the given state.

    Parameters
    ----------
    concentrations
        1-D array of shape ``(n_species,)`` giving the current species
        concentrations, in the same unit system as ``oh``.
    oh
        Scalar OH concentration.
    network
        The SOM network defining species, reactions, and rate constants.

    Returns
    -------
    dydt
        1-D array of shape ``(n_species,)`` giving the time derivative in
        concentration units per time unit. The time unit is whatever
        ``k_OH * oh`` implies (reciprocal time).

    Notes
    -----
    - ``OH`` enters linearly, so ``jax.grad`` w.r.t. ``oh`` returns
      ``stoich.T @ (k_OH * y[reactant_idx])``.
    - ``network`` is a PyTree, so this function traces cleanly through
      ``jit``, ``vmap``, and ``grad`` on its numeric leaves
      (``k_OH``, ``stoich``). Changing ``species_names`` triggers a retrace.
    """
    reactants = concentrations[network.oh_reactant_idx]  # (n_rxn,)
    reaction_rates = network.k_OH * oh * reactants  # (n_rxn,)
    dydt = network.stoich.T @ reaction_rates  # (n_species,)
    return dydt
