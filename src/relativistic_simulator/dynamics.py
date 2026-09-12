"""Constant-force special-relativistic dynamics in one spatial dimension.

Governing equation
------------------
The relativistic momentum obeys Newton's second law in its *momentum* form::

    dp/dt = F,     p = gamma * m * v

with ``F`` constant.  This is *not* ``F = m a`` for relativistic motion.

Exact closed-form solution (see ``docs/physics.md`` for the derivation)::

    p(t)  = p0 + F * t
    gamma = sqrt(1 + (p / (m c))^2)
    beta  = p / (m c * gamma)
    v     = c * beta
    E     = sqrt(m^2 c^4 + p^2 c^2)
    K     = E - m c^2                       (computed cancellation-free)
    x     = x0 + c * t * (q + q0) / (gamma + gamma0)   (F != 0; cancellation-free)
          = x0 + v0 * t                                  (F == 0)
    tau   = (m c / F) * log1p(stable rapidity increment)   (F != 0)
          = t / gamma0                      (F == 0)

An independent fixed-step RK4 integrator is provided for cross-checking.
"""
from __future__ import annotations

import numpy as np

from ._common import (
    broadcast,
    require_finite,
    require_nonnegative_time,
    require_positive_mass,
    require_subluminal,
)
from .constants import C
from .exceptions import (
    InvalidSimulationParameterError,
    InvalidTimeStepError,
    VelocityLimitError,
)
from .relativity import (
    energy_from_momentum,
    gamma_from_momentum,
    kinetic_energy_from_momentum,
    momentum_from_velocity,
    velocity_from_momentum,
)

__all__ = ["analyze_constant_force", "integrate_rk4"]

STATE_KEYS = (
    "time",
    "position",
    "momentum",
    "velocity",
    "beta",
    "gamma",
    "kinetic_energy",
    "total_energy",
    "proper_time",
)


def analyze_constant_force(
    mass: float | np.ndarray,
    force: float | np.ndarray,
    initial_velocity: float | np.ndarray,
    initial_position: float | np.ndarray = 0.0,
    time: float | np.ndarray = 0.0,
) -> dict[str, float | np.ndarray]:
    r"""Evaluate the exact constant-force solution at arbitrary times.

    Fully vectorised: all arguments broadcast against each other, so this
    works for a single trajectory (scalar ``time``) *and* for batched
    dataset generation (array ``time`` and/or array parameters).

    Parameters
    ----------
    mass : float | np.ndarray
        Rest mass in kg (``m > 0``).
    force : float | np.ndarray
        Constant applied force in newtons (any sign, ``F == 0`` allowed).
    initial_velocity : float | np.ndarray
        Initial velocity in m/s, with ``|v0| < c``.
    initial_position : float | np.ndarray, default 0.0
        Initial position in m.
    time : float | np.ndarray, default 0.0
        Coordinate time(s) in seconds, ``t >= 0``.

    Returns
    -------
    dict[str, float | np.ndarray]
        Keys ``time | position | momentum | velocity | beta | gamma |
        kinetic_energy | total_energy | proper_time``.  For scalar input,
        scalar (float) values are returned; for array input, float64 arrays.

    Raises
    ------
    InvalidMassError, VelocityLimitError, InvalidSimulationParameterError
        For unphysical parameters, or if momentum overflows double precision.

    Notes
    -----
    Position uses the rationalised work–energy form
    :math:`\Delta x = c\,t\,(q+q_0)/(\gamma+\gamma_0)`, which is exact and
    free of the catastrophic cancellation that plagues ``\Delta K / F``
    when the kinetic energy is large compared with its change.
    """
    m, _ = require_positive_mass(mass)
    F, _ = require_finite(force, "force")
    v0, _ = require_subluminal(initial_velocity, "initial_velocity")
    x0, _ = require_finite(initial_position, "initial_position")
    t, was_scalar = require_nonnegative_time(time)

    m, F, v0, x0, t = broadcast(m, F, v0, x0, t)

    p0 = momentum_from_velocity(m, v0)  # gamma0 * m * v0
    p = p0 + F * t
    if not np.all(np.isfinite(p)):
        raise InvalidSimulationParameterError(
            "momentum p(t) = p0 + F*t overflowed double precision. "
            "Reduce |force| * time or increase the mass."
        )

    gamma = gamma_from_momentum(m, p)
    gamma0 = gamma_from_momentum(m, p0)
    velocity = velocity_from_momentum(m, p)
    beta = velocity / C
    if np.any(np.abs(beta) >= 1.0):
        raise VelocityLimitError(
            "Computed |beta| reached 1 (|v| >= c) within double precision. "
            "This indicates momentum so large that float64 can no longer "
            "resolve v < c; it is rejected rather than silently clipped."
        )

    total_energy = energy_from_momentum(m, p)
    kinetic = kinetic_energy_from_momentum(m, p)

    flow_is_zero = F == 0.0
    F_safe = np.where(flow_is_zero, 1.0, F)

    # Dimensionless momenta q = p/(m c); q0 for the initial state.
    q = p / (m * C)
    q0 = p0 / (m * C)

    # Position. The textbook form (K - K0)/F suffers catastrophic
    # cancellation when the kinetic energy is large and only changes by a
    # tiny amount (high-beta coasting with a small force).  Rationalising
    #     (gamma - gamma0) = (q^2 - q0^2) / (gamma + gamma0)
    # gives the *exact*, cancellation-free closed form
    #     x = x0 + c * t * (q + q0) / (gamma + gamma0),
    # which reduces to the classical t*(v+v0)/2 at low speeds.
    position = np.where(
        flow_is_zero,
        x0 + v0 * t,
        x0 + C * t * (q + q0) / (gamma + gamma0),
    )

    # Proper time: d tau = dt / gamma.  The textbook closed form
    #   tau = (m c / F) * (asinh(q) - asinh(q0))
    # loses all precision when the rapidity change is tiny compared with
    # the rapidity itself (high-beta with a small impulse), because
    # asinh(q) ~ asinh(q0) cancels.  Stable rewrite (exact identity):
    #   asinh(q) - asinh(q0) = log1p((r - r0) / r0),   r = q + gamma,
    #   r - r0 = (q - q0) + (gamma - gamma0),
    #   gamma - gamma0 = (q - q0)(q + q0) / (gamma + gamma0),
    # where q - q0 = F t / (m c) is the physical impulse (never a
    # difference of two nearly equal numbers).
    delta_q = F * t / (m * C)
    delta_gamma = delta_q * (q + q0) / (gamma + gamma0)
    delta_r = delta_q + delta_gamma
    r0 = q0 + gamma0
    proper_time = np.where(
        flow_is_zero,
        t / gamma0,
        (m * C / F_safe) * np.log1p(delta_r / r0),
    )

    result: dict[str, float | np.ndarray] = {
        "time": t,
        "position": position,
        "momentum": p,
        "velocity": velocity,
        "beta": beta,
        "gamma": gamma,
        "kinetic_energy": kinetic,
        "total_energy": total_energy,
        "proper_time": proper_time,
    }
    if was_scalar:
        return {key: float(value) for key, value in result.items()}
    return result


def integrate_rk4(
    mass: float,
    force: float,
    initial_velocity: float,
    initial_position: float = 0.0,
    time: np.ndarray | list[float] | None = None,
    duration: float | None = None,
    dt: float | None = None,
) -> dict[str, np.ndarray]:
    r"""Independent fixed-step RK4 integration of ``dx/dt = v(p)``, ``dp/dt = F``.

    This is a **numerical cross-check** for the closed-form solution.  It
    integrates the exact ODE system

    .. math::

        \frac{dx}{dt} = v(p), \quad \frac{dp}{dt} = F, \quad
        \frac{d\tau}{dt} = 1/\gamma(p)

    with the standard 4th-order Runge–Kutta scheme on a uniform grid.  The
    momentum equation ``dp/dt = F`` is integrated by the same scheme, so the
    position and proper-time values here come from a *different numerical
    route* than the closed-form ``(K-K0)/F`` and ``arsinh`` expressions.

    Parameters
    ----------
    mass : float
        Rest mass in kg (``m > 0``).
    force : float
        Constant force in N (may be zero or negative).
    initial_velocity : float
        Initial velocity in m/s, ``|v0| < c``.
    initial_position : float, default 0.0
        Initial position in m.
    time : np.ndarray | list[float] | None
        Time grid (must start at 0, be finite, strictly increasing,
        non-negative).  Mutually exclusive with ``(duration, dt)``.
    duration : float | None
        Integration duration in seconds.  Requires ``dt``.
    dt : float | None
        Uniform step size in seconds.  Requires ``duration``.

    Returns
    -------
    dict[str, np.ndarray]
        ``time | position | momentum | velocity | beta | gamma |
        kinetic_energy | total_energy | proper_time`` float64 arrays.

    Raises
    ------
    InvalidSimulationParameterError
        If the grid arguments are inconsistent or the integration diverges.
    """
    mass_f = float(require_positive_mass(mass)[0].reshape(()))
    force_f = float(require_finite(force, "force")[0].reshape(()))
    v0_f = float(require_subluminal(initial_velocity, "initial_velocity")[0].reshape(()))
    x0_f = float(require_finite(initial_position, "initial_position")[0].reshape(()))

    if (time is None and (duration is None or dt is None)) or (
        time is not None and (duration is not None or dt is not None)
    ):
        raise InvalidSimulationParameterError(
            "Provide exactly one of: (time grid) or (duration + dt)."
        )

    if time is not None:
        t, _ = require_nonnegative_time(time)
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        if t.size < 1:
            raise InvalidSimulationParameterError("time grid must not be empty.")
        if abs(float(t[0])) != 0.0 or np.any(np.diff(t) <= 0.0):
            raise InvalidSimulationParameterError(
                "time grid must start at t = 0 and be strictly increasing."
            )
    else:
        duration_f = float(duration)
        if duration_f <= 0.0:
            raise InvalidSimulationParameterError("duration must be > 0 s for RK4 integration.")
        dt_f = float(dt)
        if dt_f <= 0.0:
            raise InvalidTimeStepError("dt must be > 0 s for RK4 integration.")
        n = max(2, int(np.ceil(duration_f / dt_f)))
        t = np.linspace(0.0, duration_f, n)

    # Initial state.
    p0 = float(momentum_from_velocity(mass_f, v0_f))
    x = x0_f
    p = p0
    tau = 0.0

    position = np.empty(t.size)
    momentum = np.empty(t.size)
    vel = np.empty(t.size)
    beta = np.empty(t.size)
    gamma = np.empty(t.size)
    kinetic = np.empty(t.size)
    total_energy = np.empty(t.size)
    proper_time = np.empty(t.size)

    position[0] = x
    momentum[0] = p
    vel[0] = v0_f
    gamma[0] = float(gamma_from_momentum(mass_f, p0))
    beta[0] = v0_f / C
    total_energy[0] = float(energy_from_momentum(mass_f, p0))
    kinetic[0] = float(kinetic_energy_from_momentum(mass_f, p0))
    proper_time[0] = 0.0

    for i in range(1, t.size):
        h = float(t[i] - t[i - 1])

        def rhs(mom: float) -> tuple[float, float, float]:
            q = mom / (mass_f * C)
            g = float(np.hypot(1.0, q))
            vv = C * mom / np.hypot(mass_f * C, mom)
            return vv, force_f, 1.0 / g

        k1v, k1f, k1t = rhs(p)
        k2v, k2f, k2t = rhs(p + 0.5 * h * k1f)
        k3v, k3f, k3t = rhs(p + 0.5 * h * k2f)
        k4v, k4f, k4t = rhs(p + h * k3f)

        x = x + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        p = p + (h / 6.0) * (k1f + 2.0 * k2f + 2.0 * k3f + k4f)
        tau = tau + (h / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)

        if not (np.isfinite(x) and np.isfinite(p) and np.isfinite(tau)):
            raise InvalidSimulationParameterError(
                "RK4 integration produced non-finite values (step too large "
                "or momentum overflow)."
            )

        position[i] = x
        momentum[i] = p
        gamma[i] = float(gamma_from_momentum(mass_f, p))
        vel[i] = C * p / np.hypot(mass_f * C, p)
        beta[i] = vel[i] / C
        total_energy[i] = float(energy_from_momentum(mass_f, p))
        kinetic[i] = float(kinetic_energy_from_momentum(mass_f, p))
        proper_time[i] = tau

    return {
        "time": t,
        "position": position,
        "momentum": momentum,
        "velocity": vel,
        "beta": beta,
        "gamma": gamma,
        "kinetic_energy": kinetic,
        "total_energy": total_energy,
        "proper_time": proper_time,
    }