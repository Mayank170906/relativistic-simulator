[![PyPI version](https://badge.fury.io/py/relativistic-simulator.svg)](https://pypi.org/project/relativistic-simulator/)

# relativistic-simulator

Research-grade, lightweight Python simulator for **1D special-relativistic massive-particle dynamics under a constant external force**.

This package is the ground-truth physics engine for a synthetic-data/ML research pipeline. It generates data from known equations; it does not discover physics and it does not contain an ML model.

## Scientific assumptions

- flat Minkowski spacetime and special relativity;
- one spatial dimension;
- constant rest mass;
- externally applied constant 1D force;
- SI units;
- no gravity, curved spacetime, electromagnetic-field model, radiation reaction, quantum effects, or FTL dynamics.

The governing equation is `dp/dt = F` with `p = gamma*m*v`, not classical `F = m*a`.

## Installation

```bash
pip install relativistic-simulator
pip install "relativistic-simulator[all]"  # optional pandas + matplotlib
```

Development uses uv:

```bash
uv sync
```

## Quick start

```python
from relativistic_simulator import C, simulate

result = simulate(
    mass=1000.0, force=1e6, initial_velocity=0.9 * C,
    duration=100.0, dt=0.01,
)

print(result.beta)
print(result.gamma)
print(result.momentum)
print(result.kinetic_energy)
print(result.total_energy)
print(result.proper_time)
print(result.validate().summary())
```

`simulate()` returns a `Trajectory` exposing `time`, `position`, `velocity`, `beta`, `gamma`, `momentum`, `kinetic_energy`, `total_energy`, and `proper_time`, plus `to_numpy()`, `to_dict()`, `to_dataframe()`, and `final_state()`.

## Equations and numerical method

For initial momentum `p0 = gamma0*m*v0`:

```text
p(t) = p0 + F*t
q(t) = p(t)/(m*c)
gamma(t) = hypot(1, q(t))
v(t) = c*q(t)/gamma(t)
E(t) = c*hypot(m*c, p(t))
K(t) = p(t)^2*c^2 / (E(t) + m*c^2)
x(t) = x0 + c*t*(q(t) + q0)/(gamma(t) + gamma0)   [F != 0]
```

The position formula is algebraically equivalent to `(K-K0)/F`, but avoids catastrophic cancellation at high beta and small impulse. Proper time uses `dτ = dt/gamma` and a stable `log1p` rapidity-increment form. `simulate(method="numerical")` supplies an independent RK4 cross-check.

See [`docs/physics.md`](docs/physics.md) for derivations and precision details.

## Relativity API

```python
from relativistic_simulator import (
    C, beta_from_velocity, velocity_from_beta, gamma_from_beta,
    gamma_from_velocity, momentum_from_velocity, velocity_from_momentum,
    energy_from_velocity, energy_from_momentum,
    kinetic_energy_from_velocity, kinetic_energy_from_momentum, rest_energy,
)

gamma = gamma_from_beta(0.99999)
```

All functions accept scalars and NumPy arrays where practical. Massive-particle states must satisfy `abs(v) < C` and `abs(beta) < 1`; invalid states raise typed exceptions and are never silently clipped.

## Dataset generation

```python
from relativistic_simulator import generate_dataset

dataset = generate_dataset(
    n_samples=100_000,
    mass_range=(100.0, 10_000.0),
    force_range=(1e4, 1e7),
    beta_range=(0.0, 0.8),
    time_range=(0.0, 100.0),
    random_seed=42,
)

dataset.to_csv("train.csv")
print(dataset.input_columns)
print(dataset.output_columns)
print(dataset.validate().summary())
```

Generation is fully vectorized with NumPy and uses `numpy.random.default_rng`, so seeded datasets are reproducible without global randomness. Inputs are `mass`, `force`, `initial_velocity`, `initial_beta`, and `time`; targets are exact `position`, `velocity`, `beta`, `gamma`, `momentum`, `kinetic_energy`, `total_energy`, and `proper_time`.

Regimes are configurable rather than hard-coded:

```python
low = generate_dataset(beta_range=(0.0, 0.5), random_seed=1)
relativistic = generate_dataset(beta_range=(0.5, 0.9), random_seed=2)
ultra = generate_dataset(beta_range=(0.9, 0.9999), random_seed=3)
training = generate_dataset(beta_range=(0.0, 0.8), random_seed=42)
extrapolation = generate_dataset(beta_range=(0.8, 0.9999), random_seed=43)
```

## Validation

`validate_trajectory` and `validate_dataset` report quantitative errors for `p = gamma*m*v`, `E = gamma*m*c²`, the energy-momentum invariant, central finite differences for `dp/dt` and `dx/dt`, proper time, initial conditions, velocity bound, and `tau <= t`. They use independent routes rather than merely repeating the generating expression.

```python
report = result.validate()
print(report.max_energy_momentum_error)
print(report.max_force_error)
print(report.max_velocity_difference_error)
print(report.passed)
```

## Optional plotting and classical comparison

```python
from relativistic_simulator.plotting import plot_trajectory, plot_gamma_vs_beta
plot_trajectory(result, path="trajectory.png", show=False)
plot_gamma_vs_beta(path="gamma.png", show=False)
```

The `classical` module is comparison-only: it provides Newtonian `p=m*v`, `K=0.5*m*v**2`, and `x=x0+v0*t+0.5*(F/m)*t**2`. It is not used by the relativistic engine.

## Examples

```bash
uv run python examples/basic_simulation.py
uv run python examples/high_beta.py
uv run python examples/generate_dataset.py
uv run python examples/benchmark.py
```

## Development, testing, and build

```bash
uv sync
uv run pytest
uv build
```

The suite covers high beta through `0.99999c`, invalid physical states, exact/RK4 agreement, vectorization, reproducibility, dataset validation, and optional plotting. `uv build` creates wheel and source distributions in `dist/`.

## Precision and limitations

The numerical representation is IEEE-754 float64. The package rejects `|v| >= c` rather than clipping and rejects states for which floating-point arithmetic cannot represent a strictly subluminal velocity. It does not model gravity, curved spacetime, electromagnetic fields, variable mass, radiation reaction, quantum effects, or FTL/spacelike trajectories.

## License

MIT License. See [`LICENSE`](LICENSE).
