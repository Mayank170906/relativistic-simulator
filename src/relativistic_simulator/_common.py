"""Private helpers for input validation and type conversion.

These helpers keep the *public* modules free of duplicated validation
plumbing.  They are internal: not re-exported, not part of the documented
API, and subject to change without notice.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .constants import C
from .exceptions import (
    InvalidMassError,
    InvalidSimulationParameterError,
    InvalidTimeStepError,
    VelocityLimitError,
)

ArrayLike = float | np.ndarray


def as_float_array(value: Any, name: str = "value") -> tuple[np.ndarray, bool]:
    """Convert a scalar-or-array input to a float64 ndarray.

    Returns ``(array, was_scalar)``.  Complex input, non-numeric input and
    non-convertible input raise :class:`InvalidSimulationParameterError`.
    """
    try:
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.complexfloating):
            raise InvalidSimulationParameterError(
                f"{name} must be real-valued; got complex input."
            )
        out = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:  # non-numeric / non-convertible
        raise InvalidSimulationParameterError(
            f"{name} must be a real number or a real-valued array; got {type(value).__name__}."
        ) from exc
    return out, np.isscalar(value)


def scalar_or_array(array: np.ndarray, was_scalar: bool) -> float | np.ndarray:
    """Return a Python float for scalar input, otherwise the ndarray."""
    return float(array) if was_scalar else array


def broadcast(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Broadcast several arrays to a common shape with a clear error message."""
    try:
        return np.broadcast_arrays(*arrays)
    except ValueError as exc:
        shapes = [a.shape for a in arrays]
        raise InvalidSimulationParameterError(
            f"Input shapes are not broadcast-compatible: {shapes}."
        ) from exc


def require_positive_mass(mass: ArrayLike, name: str = "mass") -> tuple[np.ndarray, bool]:
    """Validate strictly positive, finite rest mass; returns ``(array, was_scalar)``."""
    arr, was_scalar = as_float_array(mass, name)
    if not np.all(np.isfinite(arr)):
        raise InvalidMassError(f"{name} must be finite; got non-finite value(s).")
    if np.any(arr <= 0.0):
        raise InvalidMassError(
            f"{name} must be strictly positive (m > 0 kg); got minimum value {arr.min()!r}."
        )
    return arr, was_scalar


def require_subluminal(velocity: ArrayLike, name: str = "velocity") -> tuple[np.ndarray, bool]:
    """Validate finite velocity with ``|v| < c`` for every element."""
    arr, was_scalar = as_float_array(velocity, name)
    if not np.all(np.isfinite(arr)):
        raise VelocityLimitError(f"{name} must be finite; got non-finite value(s).")
    if np.any(np.abs(arr) >= C):
        worst = float(np.max(np.abs(arr)))
        raise VelocityLimitError(
            f"{name} violates the speed limit |v| < c: "
            f"max |v| = {worst} m/s while c = {C} m/s. "
            "Massive particles can never reach or exceed the speed of light; "
            "the value was not silently clipped."
        )
    return arr, was_scalar


def require_valid_beta(beta: ArrayLike, name: str = "beta") -> tuple[np.ndarray, bool]:
    """Validate speed ratio ``|beta| < 1`` (equivalent to ``|v| < c``)."""
    arr, was_scalar = as_float_array(beta, name)
    if not np.all(np.isfinite(arr)):
        raise VelocityLimitError(f"{name} must be finite; got non-finite value(s).")
    if np.any(np.abs(arr) >= 1.0):
        worst = float(np.max(np.abs(arr)))
        raise VelocityLimitError(
            f"{name} violates the bound |beta| < 1: got |beta| = {worst}. "
            "beta = v/c for a massive particle must satisfy |beta| < 1."
        )
    return arr, was_scalar


def require_finite(value: ArrayLike, name: str = "value") -> tuple[np.ndarray, bool]:
    """Validate that every element is finite (reject NaN / +/-inf)."""
    arr, was_scalar = as_float_array(value, name)
    if not np.all(np.isfinite(arr)):
        raise InvalidSimulationParameterError(f"{name} must be finite; got non-finite value(s).")
    return arr, was_scalar


def scalar_float(value: ArrayLike, name: str, *, positive: bool = False) -> float:
    """Require a scalar (size-1) float input; optionally strictly positive."""
    arr, _ = as_float_array(value, name)
    if arr.size != 1:
        raise InvalidSimulationParameterError(
            f"{name} must be a scalar number, got an array of shape {arr.shape}."
        )
    out = float(arr.reshape(()))
    if not np.isfinite(out):
        raise InvalidSimulationParameterError(f"{name} must be finite; got {out!r}.")
    if positive and out <= 0.0:
        raise InvalidSimulationParameterError(f"{name} must be > 0; got {out!r}.")
    return out


def validate_duration(duration: ArrayLike) -> float:
    """Duration must be a scalar, finite and non-negative (seconds)."""
    arr, _ = as_float_array(duration, "duration")
    if arr.size != 1:
        raise InvalidSimulationParameterError(
            f"duration must be a scalar number, got an array of shape {arr.shape}."
        )
    out = float(arr.reshape(()))
    if not np.isfinite(out):
        raise InvalidSimulationParameterError(f"duration must be finite; got {out!r}.")
    if out < 0.0:
        raise InvalidSimulationParameterError(
            f"duration must be >= 0 s; got {out!r}. (A negative integration interval is not a "
            "forward-in-time simulation.)"
        )
    return out


def validate_time_step(dt: ArrayLike) -> float:
    """Time step must be a scalar, finite and strictly positive (seconds)."""
    arr, _ = as_float_array(dt, "dt")
    if arr.size != 1:
        raise InvalidTimeStepError(f"dt must be a scalar number, got an array of shape {arr.shape}.")
    out = float(arr.reshape(()))
    if not np.isfinite(out):
        raise InvalidTimeStepError(f"dt must be finite; got {out!r}.")
    if out <= 0.0:
        raise InvalidTimeStepError(f"dt must be strictly positive (dt > 0 s); got {out!r}.")
    return out


def require_nonnegative_time(
    time: ArrayLike, name: str = "time"
) -> tuple[np.ndarray, bool]:
    """Validate coordinate time array: finite and ``t >= 0`` for all elements."""
    arr, was_scalar = as_float_array(time, name)
    if not np.all(np.isfinite(arr)):
        raise InvalidSimulationParameterError(f"{name} must be finite; got non-finite value(s).")
    if np.any(arr < 0.0):
        raise InvalidSimulationParameterError(
            f"{name} must be non-negative (t >= 0 s); got minimum value {arr.min()!r}. "
            "This simulator only evolves forward in coordinate time."
        )
    return arr, was_scalar