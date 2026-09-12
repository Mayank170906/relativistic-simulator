"""Exception hierarchy for :mod:`relativistic_simulator`.

The package **never silently clips or coerces unphysical values**.  When a
physical state is invalid (e.g. ``|v| >= c``, ``m <= 0``, ``dt <= 0``), the
corresponding exception below is raised with an informative message.

Hierarchy::

    RelativisticSimulatorError
    ├── InvalidMassError
    ├── VelocityLimitError
    ├── InvalidTimeStepError
    └── InvalidSimulationParameterError
"""
from __future__ import annotations

__all__ = [
    "RelativisticSimulatorError",
    "InvalidMassError",
    "VelocityLimitError",
    "InvalidTimeStepError",
    "InvalidSimulationParameterError",
]


class RelativisticSimulatorError(Exception):
    """Base class for every exception raised by this package."""


class InvalidMassError(RelativisticSimulatorError):
    """Rest mass is not a strictly positive finite number (``m > 0 kg``).

    Raised for zero, negative, ``NaN`` or infinite mass.
    """


class VelocityLimitError(RelativisticSimulatorError):
    """A velocity violates the special-relativistic bound ``|v| < c``.

    Also raised when ``|beta| = |v|/c`` reaches or exceeds 1, and when a
    velocity or speed ratio is non-finite.  Massive particles can never
    reach or exceed the speed of light, so this package rejects such states
    instead of clipping them.
    """


class InvalidTimeStepError(RelativisticSimulatorError):
    """Integration time step is not a strictly positive finite number (``dt > 0 s``)."""


class InvalidSimulationParameterError(RelativisticSimulatorError):
    """A simulation parameter is invalid: bad type, non-finite value, or out of range

    (e.g. negative duration, empty range tuple, invalid sampling seed...).
    """