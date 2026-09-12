"""Fundamental physical constants (SI units).

This module is the **single source of truth** for the speed of light.
No other module in the package may re-declare the value of ``c``; they
must import it from here.
"""
from __future__ import annotations

C: float = 299792458.0
"""Speed of light in vacuum, ``c = 299 792 458 m/s`` (exact, by definition of the SI metre)."""

C_SQUARED: float = C * C
"""Speed of light squared, ``c**2`` in ``m^2/s^2``. Precomputed exactly as ``float(C * C)``
to avoid repeating the multiplication inside hot numerical loops."""

__all__ = ["C", "C_SQUARED"]