"""Classical (Newtonian) comparison utilities.

These functions implement the *non-relativistic* limits

.. math::

    p_{cl} = m v, \\qquad K_{cl} = \\tfrac{1}{2} m v^2, \\qquad
    x_{cl}(t) = x_0 + v_0 t + \\tfrac{1}{2} (F/m) t^2

They are provided **only** for scientific comparison and visualisation
(e.g. "here is where relativity and Newtonian mechanics diverge").  They
are *not* part of the relativistic engine and must never be used to compute
ground-truth data for the simulator.

Units: m in kg, v in m/s, F in N, t in s, x in m, E in J.
"""
from __future__ import annotations

import numpy as np

from ._common import broadcast, require_finite, require_positive_mass
from .exceptions import InvalidSimulationParameterError

__all__ = ["classical_velocity", "classical_position", "classical_kinetic_energy", "classical_momentum"]


def _validate_mass_velocity(mass, velocity) -> tuple[np.ndarray, np.ndarray]:
    m, _ = require_positive_mass(mass)
    v, _ = require_finite(velocity, "velocity")
    m, v = broadcast(m, v)
    return m, v


def classical_momentum(mass: float | np.ndarray, velocity: float | np.ndarray) -> float | np.ndarray:
    r"""Newtonian momentum :math:`p = m v`.

    Unlike the relativistic momentum this has **no speed limit**: the
    function intentionally accepts any finite velocity (for comparison
    purposes only).
    """
    m, v = _validate_mass_velocity(mass, velocity)
    result = m * v
    return float(result) if np.isscalar(mass) and np.isscalar(velocity) else result


def classical_kinetic_energy(
    mass: float | np.ndarray, velocity: float | np.ndarray
) -> float | np.ndarray:
    r"""Newtonian kinetic energy :math:`K = \\tfrac{1}{2} m v^2`."""
    m, v = _validate_mass_velocity(mass, velocity)
    result = 0.5 * m * v * v
    return float(result) if np.isscalar(mass) and np.isscalar(velocity) else result


def classical_velocity(
    mass: float | np.ndarray,
    force: float | np.ndarray,
    initial_velocity: float | np.ndarray,
    time: float | np.ndarray,
) -> float | np.ndarray:
    r"""Newtonian velocity under constant force: :math:`v(t) = v_0 + (F/m) t`.

    Note this can exceed ``c``; that is exactly the classical-prediction
    artefact that relativity corrects.

    Parameters
    ----------
    mass, force, initial_velocity, time : float | np.ndarray
        SI units (kg, N, m/s, s).  All broadcast.
    """
    m, v0 = _validate_mass_velocity(mass, initial_velocity)
    F, _ = require_finite(force, "force")
    t, _ = require_finite(time, "time")
    m, v0, F, t = broadcast(m, v0, F, t)
    result = v0 + (F / m) * t
    return float(result) if np.isscalar(mass) and np.isscalar(force) and np.isscalar(initial_velocity) and np.isscalar(time) else result


def classical_position(
    mass: float | np.ndarray,
    force: float | np.ndarray,
    initial_velocity: float | np.ndarray,
    initial_position: float | np.ndarray,
    time: float | np.ndarray,
) -> float | np.ndarray:
    r"""Newtonian position under constant force:

    .. math::

        x(t) = x_0 + v_0 t + \\tfrac{1}{2} (F/m) t^2.

    Parameters
    ----------
    mass, force, initial_velocity, initial_position, time : float | np.ndarray
        SI units (kg, N, m/s, m, s).  All broadcast.
    """
    m, v0 = _validate_mass_velocity(mass, initial_velocity)
    F, _ = require_finite(force, "force")
    x0, _ = require_finite(initial_position, "initial_position")
    t, _ = require_finite(time, "time")
    m, v0, F, x0, t = broadcast(m, v0, F, x0, t)
    result = x0 + v0 * t + 0.5 * (F / m) * t * t
    scalars = (np.isscalar(mass), np.isscalar(force), np.isscalar(initial_velocity),
               np.isscalar(initial_position), np.isscalar(time))
    return float(result) if all(scalars) else result