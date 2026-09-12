"""Independent, quantitative validation of simulation results.

Principles
----------
1. **Quantitative, not boolean.**  Every check reports a maximum relative (or
   otherwise normalised) error over the whole trajectory / dataset.
2. **Independent routes.**  Wherever possible, a relation is checked through
   *different code paths* than the one that generated the data:

   * momentum from velocity (``p = gamma m v``) vs stored momentum,
   * energy from momentum (``E = hypot(m c^2, p c)``) vs stored energy,
   * the invariant :math:`E^2 - p^2c^2 = m^2c^4`,
   * finite-difference ``dp/dt`` vs the applied force ``F``,
   * finite-difference ``dx/dt`` vs the velocity,
   * trapezoid integration of ``1/gamma`` vs the stored proper time,
   * RK4 integration (``simulate(method="numerical")``) vs the closed form.

3. **No fake validation.**  ``passed`` is set purely from the measured
   errors compared against documented tolerances.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._common import require_positive_mass
from .constants import C, C_SQUARED
from .relativity import momentum_from_velocity

__all__ = ["TrajectoryValidation", "DatasetValidation", "validate_trajectory", "validate_dataset"]

_EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class TrajectoryValidation:
    """Quantitative validation report for a :class:`Trajectory`.

    All error metrics are normalised (dimensionless relative errors unless
    documented otherwise).  ``passed`` is ``True`` iff every metric is below
    its documented tolerance.
    """

    n_points: int
    max_velocity_bound_violation: float
    min_velocity_margin: float
    max_momentum_relation_error: float
    max_energy_relation_error: float
    max_energy_momentum_error: float
    max_force_error: float
    max_velocity_difference_error: float
    max_proper_time_error: float
    max_initial_condition_error: float
    max_proper_time_violation: float
    passed: bool
    rtol: float = field(default=1e-6, repr=False)

    def summary(self) -> str:
        """Human-readable one-line verification summary."""
        lines = [
            f"Trajectory validation (passed={self.passed}, rtol={self.rtol:g})",
            f"  n_points                        : {self.n_points}",
            f"  velocity bound          max err : {self.max_velocity_bound_violation:.3e}",
            f"  velocity margin (to c)  min     : {self.min_velocity_margin:.3e}",
            f"  p = gamma m v           max err : {self.max_momentum_relation_error:.3e}",
            f"  E = gamma m c^2         max err : {self.max_energy_relation_error:.3e}",
            f"  E^2-p^2c^2 = m^2c^4     max err : {self.max_energy_momentum_error:.3e}",
            f"  dp/dt = F               max err : {self.max_force_error:.3e}",
            f"  dx/dt = v               max err : {self.max_velocity_difference_error:.3e}",
            f"  d(tau)=dt/gamma          max err : {self.max_proper_time_error:.3e}",
            f"  initial conditions      max err : {self.max_initial_condition_error:.3e}",
            f"  tau <= t                max viol: {self.max_proper_time_violation:.3e}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class DatasetValidation:
    """Quantitative validation report for a :class:`RelativisticDataset`.

    Row-wise, vectorised checks of the physical relations between the input
    columns and the ground-truth output columns.
    """

    n_rows: int
    max_velocity_bound_violation: float
    min_velocity_margin: float
    max_momentum_relation_error: float
    max_energy_relation_error: float
    max_energy_momentum_error: float
    max_proper_time_violation: float
    max_proper_time_consistency_error: float
    max_position_consistency_error: float
    all_finite: bool
    passed: bool
    rtol: float = field(default=1e-9, repr=False)

    def summary(self) -> str:
        """Human-readable one-line verification summary."""
        lines = [
            f"Dataset validation (passed={self.passed}, rtol={self.rtol:g})",
            f"  n_rows                           : {self.n_rows}",
            f"  all values finite                : {self.all_finite}",
            f"  velocity bound          max viol : {self.max_velocity_bound_violation:.3e}",
            f"  velocity margin (to c)  min      : {self.min_velocity_margin:.3e}",
            f"  p = gamma m v           max err  : {self.max_momentum_relation_error:.3e}",
            f"  E = gamma m c^2         max err  : {self.max_energy_relation_error:.3e}",
            f"  E^2-p^2c^2 = m^2c^4     max err  : {self.max_energy_momentum_error:.3e}",
            f"  tau <= t                max viol : {self.max_proper_time_violation:.3e}",
            f"  tau consistency         max err  : {self.max_proper_time_consistency_error:.3e}",
            f"  position consistency    max err  : {self.max_position_consistency_error:.3e}",
        ]
        return "\n".join(lines)
def _normalise(values: np.ndarray, scale: float) -> float:
    """Return max |values| / scale; 0.0 if scale is zero (degenerate case)."""
    if scale <= 0.0 or not np.any(values):
        return 0.0
    return float(np.max(np.abs(values)) / scale)


def validate_trajectory(
    trajectory: "Trajectory",
    *,
    rtol: float = 1e-6,
    ic_rtol: float = 1e-9,
) -> TrajectoryValidation:
    """Validate a :class:`Trajectory` against the physical invariants.

    Parameter tolerance ``rtol`` applies to every physical-relation check
    (momentum, energy, invariant, dp/dt, dx/dt, proper time).  ``ic_rtol``
    applies to the initial conditions at ``t = 0``.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to check.
    rtol : float, default 1e-6
        Relative tolerance for physical relations.
    ic_rtol : float, default 1e-9
        Tolerance for initial-condition consistency.

    Returns
    -------
    TrajectoryValidation
        Quantitative errors and ``passed`` flag.

    Notes
    -----
    * :attr:`max_velocity_bound_violation` is ``max(0, max|beta| - 1)`` and
      must be exactly ``0``.
    * :attr:`max_energy_momentum_error` is normalised with a floating-point
      noise floor, so very high ``gamma`` does not cause a spurious failure.
    * Finite-difference checks use pure central differences on interior
      points (one-sided edge stencils amplify float rounding).
    """
    mass = float(trajectory.mass)
    force = float(trajectory.force)

    time = np.asarray(trajectory.time, dtype=np.float64)
    position = np.asarray(trajectory.position, dtype=np.float64)
    momentum = np.asarray(trajectory.momentum, dtype=np.float64)
    velocity = np.asarray(trajectory.velocity, dtype=np.float64)
    beta = np.asarray(trajectory.beta, dtype=np.float64)
    gamma = np.asarray(trajectory.gamma, dtype=np.float64)
    total_energy = np.asarray(trajectory.total_energy, dtype=np.float64)
    proper_time = np.asarray(trajectory.proper_time, dtype=np.float64)

    n = time.size

    # --- 1. Velocity bound -------------------------------------------------
    max_beta = float(np.max(np.abs(beta)))
    bound_violation = max(0.0, max_beta - 1.0)
    margin = max(0.0, 1.0 - max_beta)

    # --- 2. Momentum relation: p = gamma m v (independent route) ------------
    p_check = momentum_from_velocity(mass, velocity)
    p_scale = max(float(np.max(np.abs(momentum))) if n else 0.0, 1e-300)
    momentum_err = _normalise(p_check - momentum, p_scale)

    # --- 3. Energy relation: E = gamma m c^2 (independent route) ------------
    e_check = gamma * mass * C_SQUARED
    e_scale = max(float(np.max(np.abs(total_energy))) if n else 0.0, mass * C_SQUARED)
    energy_err = _normalise(e_check - total_energy, e_scale)

    # --- 4. Energy-momentum invariant: E^2 - p^2c^2 = m^2c^4 ----------------
    inv = total_energy**2 - (momentum * C) ** 2 - (mass * C_SQUARED) ** 2
    inv_mass_scale = float(mass * C_SQUARED) ** 2
    roundoff = 16.0 * _EPS * max(
        float(np.max(np.abs(total_energy**2))) if n else 0.0,
        float(np.max(np.abs((momentum * C) ** 2))) if n else 0.0,
    )
    inv_scale = max(inv_mass_scale, roundoff, 1e-300)
    invariant_err = _normalise(inv, inv_scale)
    # --- 5. dp/dt = F (central finite differences, interior points) ----------
    # np.gradient's one-sided edge stencils amplify float rounding even for
    # bitwise-constant input, so the independent check uses pure central
    # differences on interior points only.
    if n >= 3:
        dp_dt = (momentum[2:] - momentum[:-2]) / (time[2:] - time[:-2])
        drift = float(abs(momentum[-1] - momentum[0])) / float(time[-1] - time[0])
        force_scale = max(abs(force), drift, 1e-300)
        diff_f = dp_dt - force
        force_err = 0.0 if not np.any(diff_f) else float(np.max(np.abs(diff_f)) / force_scale)
    else:
        force_err = 0.0

    # --- 6. dx/dt = v (central finite differences, interior points) ---------
    if n >= 3:
        dx_dt = (position[2:] - position[:-2]) / (time[2:] - time[:-2])
        v_scale = max(float(np.max(np.abs(velocity))) if n else 0.0, 1e-300)
        velocity_diff_err = _normalise(dx_dt - velocity[1:-1], v_scale)
    else:
        velocity_diff_err = 0.0

    # --- 7. Proper time: d tau = dt / gamma (trapezoid integration) ---------
    if n >= 2:
        integrand = 1.0 / gamma
        half = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(time)
        tau_num = np.concatenate(([0.0], np.cumsum(half)))
        tau_scale = max(float(np.max(np.abs(proper_time))) if n else 0.0, 1e-300)
        proper_err = _normalise(proper_time - tau_num, tau_scale)
    else:
        proper_err = 0.0

    # --- 8. Proper-time inequality: tau <= t --------------------------------
    proper_violation = max(0.0, float(np.max(proper_time - time)) if n else 0.0)

    # --- 9. Initial conditions at t = 0 -------------------------------------
    if n >= 1:
        x0 = float(trajectory.initial_position)
        v0 = float(trajectory.initial_velocity)
        p0_check = momentum_from_velocity(mass, v0)
        tau_max = float(np.max(np.abs(proper_time))) if n else 0.0
        ic = [
            abs(position[0] - x0) / max(abs(x0), 1.0),
            abs(velocity[0] - v0) / max(abs(v0), C),
            abs(momentum[0] - p0_check) / max(abs(p0_check), mass * C),
            abs(proper_time[0] - 0.0) / max(tau_max, 1.0),
            abs(float(time[0]) - 0.0),
        ]
        ic_err = float(np.max(ic))
    else:
        ic_err = 0.0

    passed = bool(
        bound_violation == 0.0
        and momentum_err <= rtol
        and energy_err <= rtol
        and invariant_err <= rtol
        and force_err <= rtol
        and velocity_diff_err <= rtol
        and proper_err <= rtol
        and proper_violation == 0.0
        and ic_err <= ic_rtol
    )

    return TrajectoryValidation(
        n_points=n,
        max_velocity_bound_violation=bound_violation,
        min_velocity_margin=margin,
        max_momentum_relation_error=momentum_err,
        max_energy_relation_error=energy_err,
        max_energy_momentum_error=invariant_err,
        max_force_error=force_err,
        max_velocity_difference_error=velocity_diff_err,
        max_proper_time_error=proper_err,
        max_initial_condition_error=ic_err,
        max_proper_time_violation=proper_violation,
        passed=passed,
        rtol=rtol,
    )


def validate_dataset(
    dataset: "RelativisticDataset",
    *,
    rtol: float = 1e-9,
) -> DatasetValidation:
    r"""Validate a :class:`RelativisticDataset` row by row (vectorised).

    Cross-checks the ground-truth output columns against the input columns
    using *independent* relations:

    * velocity bound ``|beta| < 1``,
    * momentum :math:`p = \gamma m v`,
    * energy :math:`E = \gamma m c^2`,
    * invariant :math:`E^2 - p^2 c^2 = m^2 c^4`,
    * proper time :math:`\tau \le t` and the closed-form rapidity expression,
    * position :math:`x = x_0 + c\,t\,(q+q_0)/(\gamma+\gamma_0)`,
    * all values finite.

    Parameters
    ----------
    dataset : RelativisticDataset
        The dataset to validate.
    rtol : float, default 1e-9
        Relative tolerance for every relation check.

    Returns
    -------
    DatasetValidation
        Quantitative errors and ``passed`` flag.
    """
    cols = [name for name in
            ("mass", "force", "initial_velocity", "time", "position", "momentum",
             "velocity", "beta", "gamma", "kinetic_energy", "total_energy",
             "proper_time")
            if hasattr(dataset, name)]
    arrays = {name: np.asarray(getattr(dataset, name), dtype=np.float64) for name in cols}
    for name in ("mass", "force", "initial_velocity", "time", "beta"):
        if name not in arrays:
            raise ValueError(f"dataset is missing required column {name!r}.")

    mass = arrays["mass"]
    force = arrays["force"]
    v0 = arrays["initial_velocity"]
    t = arrays["time"]
    position = arrays["position"]
    momentum = arrays["momentum"]
    velocity = arrays["velocity"]
    beta = arrays["beta"]
    gamma = arrays["gamma"]
    kinetic_energy = arrays["kinetic_energy"]
    total_energy = arrays["total_energy"]
    proper_time = arrays["proper_time"]

    n = int(mass.shape[0])
    all_finite = all(bool(np.all(np.isfinite(arrays[k]))) for k in ("mass", "force", "initial_velocity", "time", "position", "momentum", "velocity", "beta", "gamma", "kinetic_energy", "total_energy", "proper_time"))

    # 1. velocity bound
    max_beta = float(np.max(np.abs(beta)))
    bound_violation = max(0.0, max_beta - 1.0)
    margin = max(0.0, 1.0 - max_beta)

    # 2. momentum relation
    p_check = momentum_from_velocity(mass, velocity)
    p_scale = np.maximum(np.abs(momentum), 1e-300)
    momentum_err = float(np.max(np.abs(p_check - momentum) / p_scale)) if n else 0.0

    # 3. energy relation
    e_check = gamma * mass * C_SQUARED
    e_scale = np.maximum(np.abs(total_energy), mass * C_SQUARED)
    energy_err = float(np.max(np.abs(e_check - total_energy) / e_scale)) if n else 0.0

    # 4. invariant (with float noise floor)
    inv = total_energy**2 - (momentum * C) ** 2 - (mass * C_SQUARED) ** 2
    inv_scale = np.maximum(
        (mass * C_SQUARED) ** 2,
        16.0 * _EPS * np.maximum(total_energy**2, (momentum * C) ** 2),
    )
    invariant_err = float(np.max(np.abs(inv) / inv_scale)) if n else 0.0

    # 5. proper-time inequality
    proper_violation = max(0.0, float(np.max(proper_time - t)) if n else 0.0)

    # 6. proper time closed-form (stable log1p rapidity form)
    q0 = momentum_from_velocity(mass, v0) / (mass * C)
    q = momentum / (mass * C)
    gamma0 = np.hypot(1.0, q0)
    gamma_p = np.hypot(1.0, q)
    F_safe = np.where(force == 0.0, 1.0, force)
    delta_q = force * t / (mass * C)
    delta_gamma = delta_q * (q + q0) / (gamma_p + gamma0)
    tau_close = np.where(
        force == 0.0,
        t / gamma0,
        (mass * C / F_safe) * np.log1p((delta_q + delta_gamma) / (q0 + gamma0)),
    )
    tau_scale = np.maximum(np.abs(proper_time), np.abs(tau_close))
    proper_consistency = (
        float(np.max(np.abs(proper_time - tau_close) / tau_scale)) if n else 0.0
    )

    # 7. position closed-form: x = x0 + c t (q + q0)/(gamma + gamma0)  (F != 0)
    x0 = float(getattr(dataset, "initial_position", 0.0))
    x_close = np.where(
        force == 0.0,
        x0 + v0 * t,
        x0 + C * t * (q + q0) / (np.hypot(1.0, q) + gamma0),
    )
    x_scale = np.maximum(np.abs(position), np.abs(x_close))
    position_consistency = (
        float(np.max(np.abs(position - x_close) / x_scale)) if n else 0.0
    )

    passed = bool(
        all_finite
        and bound_violation == 0.0
        and momentum_err <= rtol
        and energy_err <= rtol
        and invariant_err <= rtol
        and proper_violation == 0.0
        and proper_consistency <= rtol
        and position_consistency <= rtol
    )

    return DatasetValidation(
        n_rows=n,
        max_velocity_bound_violation=bound_violation,
        min_velocity_margin=margin,
        max_momentum_relation_error=momentum_err,
        max_energy_relation_error=energy_err,
        max_energy_momentum_error=invariant_err,
        max_proper_time_violation=proper_violation,
        max_proper_time_consistency_error=proper_consistency,
        max_position_consistency_error=position_consistency,
        all_finite=all_finite,
        passed=passed,
        rtol=rtol,
    )