"""Example 3: generate a ground-truth dataset for the future ML experiment.

Research pipeline this supports (the ML part is NOT part of this package):

    physics equations
      -> relativistic-simulator (this package, exact physics)
        -> thousands/millions of synthetic observations
          -> (external) ML model
            -> predictions on unseen ultra-relativistic regimes
              -> comparison against exact physics

Here we generate:
  * a training regime      beta in (0.0, 0.8)
  * an extrapolation regime beta in (0.8, 0.9999)
with a fixed random seed for reproducibility, and save both to CSV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from relativistic_simulator import generate_dataset

OUTPUT_DIR = Path(__file__).parent / "output"
N_TRAIN = 100_000
N_EXTRAP = 20_000
SEED = 42


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Generating ground-truth data (exact physics, seeded) ...")
    train = generate_dataset(
        n_samples=N_TRAIN,
        mass_range=(100.0, 10_000.0),
        force_range=(1e4, 1e7),
        beta_range=(0.0, 0.8),          # training regime
        time_range=(0.0, 100.0),
        random_seed=SEED,
    )
    extrap = generate_dataset(
        n_samples=N_EXTRAP,
        mass_range=(100.0, 10_000.0),
        force_range=(1e4, 1e7),
        beta_range=(0.8, 0.9999),       # unseen ultra-relativistic regime
        time_range=(0.0, 100.0),
        random_seed=SEED + 1,
    )

    train_csv = OUTPUT_DIR / "train_beta_0_0.8.csv"
    extrap_csv = OUTPUT_DIR / "extrap_beta_0.8_0.9999.csv"
    train.to_csv(str(train_csv))
    extrap.to_csv(str(extrap_csv))

    print()
    print(f"train    : {len(train):>7} rows, beta in [{train.initial_beta.min():.4f}, "
          f"{train.initial_beta.max():.4f}] -> {train_csv.name}")
    print(f"extrap   : {len(extrap):>7} rows, beta in [{extrap.initial_beta.min():.4f}, "
          f"{extrap.initial_beta.max():.4f}] -> {extrap_csv.name}")
    print()
    print("input columns :", train.input_columns)
    print("target columns:", train.output_columns)

    report = train.validate()
    assert report.passed, report.summary()
    print()
    print(f"train physics validation: passed={report.passed}, "
          f"max invariant err = {report.max_energy_momentum_error:.2e}, "
          f"max momentum err  = {report.max_momentum_relation_error:.2e}")

    # Reproducibility check: same seed -> bitwise-identical data
    again = generate_dataset(
        n_samples=N_TRAIN,
        mass_range=(100.0, 10_000.0),
        force_range=(1e4, 1e7),
        beta_range=(0.0, 0.8),
        time_range=(0.0, 100.0),
        random_seed=SEED,
    )
    assert np.array_equal(train.momentum, again.momentum)
    print("reproducibility: same seed reproduces bitwise-identical data")


if __name__ == "__main__":
    main()
