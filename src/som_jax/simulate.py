"""Public ``simulate()`` wrapper around the SOM RHS.

Uses diffrax for time-stepping — specifically ``Kvaerno5`` (5th-order ESDIRK
implicit) with a PI step-size controller. The wrapper is deliberately thin;
all chemistry lives in :func:`som_jax.rhs.som_rhs`, and solver tuning knobs
(tolerances, max steps, adjoint method) are forwarded as kwargs.

Scope for v1 (S1.7)
-------------------
- Constant OH as a scalar (ppm or any self-consistent unit).
- Prescribed save points via ``save_at``.
- No dLVP, no partitioning, no time-varying OH — those land in later chunks
  (``som-jax`` stays gas-phase only per decision D2; a time-varying
  ``oh_trajectory`` callable is S1.9).

Unit conventions
----------------
Unit-agnostic, matching ``som_rhs``. If ``initial`` and ``oh`` are in ppm
and ``network.k_OH`` is in ppm⁻¹ * time⁻¹, the output trajectory is in ppm
and ``t_span`` / ``save_at`` are in the same time unit.

References
----------
- Cappa, C. D. and Wilson, K. R. (2012). Multi-generation gas-phase oxidation,
  equilibrium partitioning, and the formation and evolution of secondary
  organic aerosol. *Atmos. Chem. Phys.*, 12, 8399-8411.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
from jax import Array

from som_jax.mechanism.network import SOMNetwork
from som_jax.oh import OHCallable, OHInput, as_oh_callable
from som_jax.rhs import som_rhs


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class SOMTrajectory:
    """Output of :func:`simulate`.

    Attributes
    ----------
    t
        ``(n_save,)`` array of times corresponding to the rows of :attr:`y`.
    y
        ``(n_save, n_species)`` array of species concentrations in the same
        unit system as the initial condition passed to :func:`simulate`.
    species_names
        Static tuple of species names; row *j* of :attr:`y` tracks
        ``species_names[j]``.
    """

    t: Array
    y: Array
    species_names: tuple[str, ...]

    def y_of(self, name: str) -> Array:
        """Return the ``(n_save,)`` trajectory for species ``name``."""
        try:
            idx = self.species_names.index(name)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"species {name!r} not in trajectory") from exc
        return self.y[:, idx]

    # --- PyTree registration so jit/vmap transforms can cross simulate ---

    def tree_flatten(self) -> tuple[tuple[Array, Array], tuple[tuple[str, ...]]]:
        children = (self.t, self.y)
        aux = (self.species_names,)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[tuple[str, ...]],
        children: tuple[Array, Array],
    ) -> SOMTrajectory:
        (species_names,) = aux
        t, y = children
        return cls(t=t, y=y, species_names=species_names)


def build_initial(network: SOMNetwork, values: Mapping[str, float]) -> Array:
    """Build a ``(n_species,)`` initial-concentration vector from a dict.

    Missing species default to zero.

    Parameters
    ----------
    network
        The SOM network whose species order is authoritative.
    values
        Mapping ``{species_name: concentration}``.

    Raises
    ------
    KeyError
        If ``values`` references a species not in ``network``.
    """
    y = jnp.zeros(network.n_species, dtype=jnp.float64)
    for name, val in values.items():
        y = y.at[network.species_index(name)].set(val)
    return y


def _make_vector_field(oh_callable: OHCallable) -> Callable[[Any, Array, Any], Array]:
    """Build a diffrax vector field closed over ``oh_callable``.

    A closure is cleaner than threading the callable through ``args`` —
    functions are not first-class PyTrees, and stuffing one into ``args``
    forces awkward static-argnum gymnastics. Capturing it here makes the
    callable part of the compiled function's identity, which matches
    diffrax's own expectations.
    """

    def vf(t: Any, y: Array, args: Any) -> Array:
        (network,) = args
        oh_t = oh_callable(jnp.asarray(t))
        return som_rhs(y, oh_t, network)

    return vf


def simulate(
    network: SOMNetwork,
    initial: Array,
    oh: OHInput,
    t_span: tuple[float, float],
    save_at: Array | Iterable[float],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-14,
    max_steps: int = 10_000,
    solver: Any = None,
    adjoint: Any = None,
) -> SOMTrajectory:
    """Integrate the SOM ODE from ``t_span[0]`` to ``t_span[1]``.

    Parameters
    ----------
    network
        Reaction network; see :class:`SOMNetwork`.
    initial
        ``(n_species,)`` array of initial concentrations. Species order
        matches ``network.species_names``. Use :func:`build_initial` to
        construct from a ``{name: value}`` mapping.
    oh
        OH concentration. May be a scalar (constant over the integration)
        or a callable ``t -> OH(t)``. See :mod:`som_jax.oh` for helpers
        (``oh_constant``, ``oh_linear_ramp``, ``oh_piecewise_linear``,
        ``oh_exponential_decay``).  Same unit system as ``initial``.
    t_span
        ``(t0, t1)`` integration bounds.
    save_at
        Times at which to save the state. Must satisfy ``t0 <= t <= t1`` and
        be monotonically increasing.
    rtol, atol
        Relative and absolute tolerances for the PI step-size controller.
        Defaults target ≤0.1% scientific-faithfulness bar from the master
        plan (rtol 1e-6 gives ~1000x headroom below the 0.1% threshold).
    max_steps
        Safety cap on the number of solver steps. Raise if integration
        fails for a stiff regime; diffrax will error out rather than loop.
    solver
        Override the default ``diffrax.Kvaerno5``. Any diffrax stiff solver
        works; non-stiff solvers are unlikely to converge on this system.
    adjoint
        Override the default ``diffrax.RecursiveCheckpointAdjoint``.
        Use ``BacksolveAdjoint()`` when memory-constrained on very long
        integrations.

    Returns
    -------
    SOMTrajectory
        Saved times and species concentrations.

    Raises
    ------
    ValueError
        If ``initial.shape != (network.n_species,)``.
    """
    if initial.shape != (network.n_species,):
        raise ValueError(f"initial has shape {initial.shape}; expected ({network.n_species},)")

    solver = solver if solver is not None else diffrax.Kvaerno5()
    adjoint = adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint()

    oh_callable = as_oh_callable(oh)
    term = diffrax.ODETerm(_make_vector_field(oh_callable))
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=jnp.asarray(save_at))

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=float(t_span[0]),
        t1=float(t_span[1]),
        dt0=None,
        y0=initial,
        args=(network,),
        saveat=saveat,
        stepsize_controller=controller,
        adjoint=adjoint,
        max_steps=max_steps,
    )

    return SOMTrajectory(
        t=sol.ts,
        y=sol.ys,
        species_names=network.species_names,
    )
