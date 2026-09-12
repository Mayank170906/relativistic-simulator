"""Container for simulation results with convenient conversions.

A :class:`Trajectory` is a plain, immutable data holder produced by
:func:`relativistic_simulator.simulate`.  It owns the full time history of
every state variable together with the simulation parameters, and provides
conversions to NumPy arrays, plain dictionaries, and (optionally) pandas
DataFrames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dynamics import STATE_KEYS

__all__ = ["Trajectory"]

TIME_VARYING_COLUMNS: tuple[str, ...] = STATE_KEYS


@dataclass(frozen=True)
class Trajectory:
    """Full solution of a constant-force relativistic trajectory.

    Parameters
    ----------
    mass : float
        Rest mass in kg.
    force : float
        Constant applied force in N.
    initial_velocity : float
        Initial velocity in m/s (``|v0| < c``).
    initial_position : float
        Initial position in m.
    method : str
        How the trajectory was computed, ``"analytic"`` (exact closed form)
        or ``"numerical"`` (RK4 cross-check).
    time : np.ndarray
        Coordinate time grid in seconds (1-D float64, starts at 0).
    position, momentum, velocity, beta, gamma, kinetic_energy, total_energy,
    proper_time : np.ndarray
        Time histories of the state variables (SI units: m, kg·m/s, m/s,
        dimensionless, dimensionless, J, J, s).

    Notes
    -----
    The dataclass is frozen, so ``dataclasses.replace`` can be used to
    derive modified copies (useful for validation experiments).  NumPy
    arrays themselves remain mutable; prefer not to mutate them in place.
    """

    mass: float
    force: float
    initial_velocity: float
    initial_position: float
    method: str
    time: np.ndarray
    position: np.ndarray
    momentum: np.ndarray
    velocity: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    proper_time: np.ndarray

    def __post_init__(self) -> None:
        n = np.asarray(self.time).shape[0]
        for name in TIME_VARYING_COLUMNS:
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(
                    f"Trajectory column {name!r} must be a 1-D array with "
                    f"{n} points, got shape {arr.shape}."
                )
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "force", float(self.force))
        object.__setattr__(self, "initial_velocity", float(self.initial_velocity))
        object.__setattr__(self, "initial_position", float(self.initial_position))
        object.__setattr__(self, "method", str(self.method))

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def n_points(self) -> int:
        """Number of recorded time points."""
        return int(self.time.shape[0])

    @property
    def duration(self) -> float:
        """Final coordinate time (s)."""
        return float(self.time[-1])

    @property
    def columns(self) -> tuple[str, ...]:
        """Names of the time-varying columns (the DataFrame/array columns)."""
        return TIME_VARYING_COLUMNS

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return (
            f"Trajectory(mass={self.mass!r} kg, force={self.force!r} N, "
            f"initial_velocity={self.initial_velocity!r} m/s, "
            f"duration={self.duration!r} s, points={self.n_points}, "
            f"method={self.method!r})"
        )

    # ------------------------------------------------------------------ #
    # Conversions
    # ------------------------------------------------------------------ #
    def to_numpy(self) -> np.ndarray:
        """Return the time-varying columns as a single float64 array.

        Returns
        -------
        np.ndarray
            Shape ``(n_points, n_columns)`` with column order given by
            :attr:`columns`.
        """
        return np.column_stack([getattr(self, name) for name in TIME_VARYING_COLUMNS])

    def to_dict(self) -> dict[str, Any]:
        """Return a dict of time-varying arrays plus simulation parameters."""
        data: dict[str, Any] = {name: getattr(self, name) for name in TIME_VARYING_COLUMNS}
        data.update(
            {
                "mass": self.mass,
                "force": self.force,
                "initial_velocity": self.initial_velocity,
                "initial_position": self.initial_position,
                "method": self.method,
            }
        )
        return data

    def to_dataframe(self) -> "pd.DataFrame":
        """Return a pandas DataFrame (requires optional ``pandas``).

        Raises
        ------
        ImportError
            If pandas is not installed (it is an optional dependency;
            install with ``pip install relativistic-simulator[pandas]``).
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover (exercised in wheel tests)
            raise ImportError(
                "Trajectory.to_dataframe() requires the optional 'pandas' "
                "extra: `pip install 'relativistic-simulator[pandas]'`."
            ) from exc
        frame = pd.DataFrame({name: getattr(self, name) for name in TIME_VARYING_COLUMNS})
        frame.attrs["mass"] = self.mass
        frame.attrs["force"] = self.force
        frame.attrs["initial_velocity"] = self.initial_velocity
        frame.attrs["initial_position"] = self.initial_position
        frame.attrs["method"] = self.method
        return frame

    def final_state(self) -> dict[str, float]:
        """Return the last state of the trajectory as a dict of float scalars."""
        return {name: float(getattr(self, name)[-1]) for name in TIME_VARYING_COLUMNS}

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self, **kwargs: Any):
        """Run :func:`relativistic_simulator.validate_trajectory` on this trajectory.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`~relativistic_simulator.validate_trajectory`
            (e.g. ``rtol=...``).

        Returns
        -------
        TrajectoryValidation
            Quantitative validation report.
        """
        from .validation import validate_trajectory

        return validate_trajectory(self, **kwargs)