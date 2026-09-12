# Changelog

All notable changes to this project will be documented here.

## [0.1.0] - 2026-09-13

### Added

- 1D special-relativistic massive-particle dynamics under constant force.
- Numerically stable relativistic kinematics, energy, momentum, and proper-time functions.
- Exact analytical solver and independent RK4 cross-check integrator.
- Vectorized reproducible NumPy dataset generation.
- Quantitative physical validation reports.
- Optional pandas conversion and matplotlib plotting utilities.
- Pytest test suite, examples, and uv/PyPI packaging configuration.

### Scientific scope

The model assumes flat Minkowski spacetime, one spatial dimension, constant rest mass,
and an externally applied constant 1D force. It does not model gravity, fields,
radiation reaction, quantum effects, or faster-than-light trajectories.