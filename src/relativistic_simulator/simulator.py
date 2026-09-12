"""Public simulation entry point.

The high-level :func:`simulate` function produces a :class:`Trajectory`
(a full time history of every physical quantity) for a massive particle
under a constant external force in 1D special relativity.
"""
from __future__ import annotations

import numpy as np

from ._common import (
    require_finite,
    require_nonnegative_time,
    require_positive_mass,
    require_subluminal,
    validate_duration,
    validate_time_step,
)
from .constants import C
from .dynamics import analyze_constant_force, integrate_rk4
from .exceptions import InvalidSimulationParameterError
from .trajectory import Trajectory

__all__ = ["simulate"]

_METHODS = ("analytic", "numerical")


def simulate(
    mass: float | np.ndarray,
    force: float | np.ndarray,
    initial_velocity: float | np.ndarray = 0.0,
    initial_position: float | np.ndarray = 0.0,
    duration: float = 1.0,
    dt: float = 0.01,
    method: str = "analytic",
) -> Trajectory:
    r"""Simulate a constant-force relativistic trajectory in 1D.

    Solves :math:`dp/dt = F` exactly (``method="analytic"``) or by RK4
    numerical integration (``method="numerical"``).

    Parameters
    ----------
    mass : float
        Rest mass in kg (``m > 0``).
    force : float
        Constant applied force in N (any sign; ``F == 0`` gives uniform
        motion ``v = v0``).
    initial_velocity : float, default 0.0
        Initial velocity in m/s, ``|v0| < c``.
    initial_position : float, default 0.0
        Initial position in m.
    duration : float, default 1.0
        Coordinate-time simulation horizon in seconds (``duration >= 0``).
        ``duration == 0`` returns a single point at ``t = 0``.
    dt : float, default 0.01
        Requested time step in seconds (``dt > 0``).  The uniform grid uses
        an actual step of ``duration / n_steps <= dt`` so the final point is
        exactly at ``t = duration``.
    method : str, default "analytic"
        Either ``"analytic"`` (exact closed form; default, recommended) or
        ``"numerical"`` (fixed-step RK4 cross-check integrator).

    Returns
    -------
    Trajectory
        Time history: ``time | position | momentum | velocity | beta |
        gamma | kinetic_energy | total_energy | proper_time`` plus the
        simulation parameters.  See :class:`Trajectory`.

    Raises
    ------
    InvalidMassError
        If ``mass <= 0`` or non-finite.
    VelocityLimitError
        If ``|initial_velocity| >= c`` (or the evolved state hits ``c``
        within double precision).
    InvalidTimeStepError
        If ``dt <= 0`` or non-finite.
    InvalidSimulationParameterError
        If ``duration < 0``, non-finite values occur in ``force`` /
        ``initial_position``, or ``method`` is unknown.

    Examples
    --------
    >>> from relativistic_simulator import simulate
    >>> result = simulate(mass=1000.0, force=1e6, initial_velocity=0.5*C,
    ...                    duration=10.0, dt=0.01)
    >>> result.beta[-1]  # doctest: +SKIP
    >>> result.validate().passed  # doctest: +SKIP
    True
    """
    mass_f = float(require_positive_mass(mass, "mass")[0].reshape(()))
    force_f = float(require_finite(force, "force")[0].reshape(()))
    v0_f = float(require_subluminal(initial_velocity, "initial_velocity")[0].reshape(()))
    x0_f = float(require_finite(initial_position, "initial_position")[0].reshape(()))
    duration_f = validate_duration(duration)
    dt_f = validate_time_step(dt)
    if method not in _METHODS:
        raise InvalidSimulationParameterError(
            f"method must be one of {_METHODS}; got {method!r}."
        )

    if duration_f == 0.0:
        n_steps = 0
    else:
        # ceil() guarantees actual step duration/n <= dt.
        n_steps = int(np.ceil(duration_f / dt_f))
    time = np.linspace(0.0, duration_f, n_steps + 1)

    if method == "analytic":
        state = analyze_constant_force(mass_f, force_f, v0_f, x0_f, time)
    else:
        state = integrate_rk4(
            mass_f, force_f, v0_f, x0_f, time=time
        )

    return Trajectory(
        mass=mass_f,
        force=force_f,
        initial_velocity=v0_f,
        initial_position=x0_f,
        method=method,
        **state,  # type: ignore[arg-type]
    )