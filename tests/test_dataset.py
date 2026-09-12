"""Tests for the vectorised dataset-generation API."""
from __future__ import annotations

import time

import numpy as np
import pytest

from relativistic_simulator import C
from relativistic_simulator.dataset import COLUMN_NAMES, generate_dataset
from relativistic_simulator.exceptions import (
    InvalidSimulationParameterError,
    VelocityLimitError,
)


def test_reproducible_with_seed() -> None:
    a = generate_dataset(n_samples=1000, random_seed=42)
    b = generate_dataset(n_samples=1000, random_seed=42)
    for name in COLUMN_NAMES:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_different_seeds_differ() -> None:
    a = generate_dataset(n_samples=1000, random_seed=1)
    b = generate_dataset(n_samples=1000, random_seed=2)
    assert not np.array_equal(a.mass, b.mass)


def test_requested_sample_count() -> None:
    ds = generate_dataset(n_samples=1234, random_seed=0)
    assert len(ds) == 1234
    for name in COLUMN_NAMES:
        assert getattr(ds, name).shape == (1234,)


def test_ranges_respected() -> None:
    ds = generate_dataset(
        n_samples=10_000,
        mass_range=(100.0, 500.0),
        force_range=(1e4, 1e5),
        beta_range=(0.5, 0.7),
        time_range=(1.0, 2.0),
        random_seed=7,
    )
    assert np.all(ds.mass >= 100.0) and np.all(ds.mass <= 500.0)
    assert np.all(ds.force >= 1e4) and np.all(ds.force <= 1e5)
    assert np.all(ds.initial_beta >= 0.5) and np.all(ds.initial_beta <= 0.7)
    assert np.all(ds.time >= 1.0) and np.all(ds.time <= 2.0)


def test_no_invalid_velocities() -> None:
    ds = generate_dataset(n_samples=5000, beta_range=(0.99, 0.9999), random_seed=3)
    assert np.all(np.abs(ds.beta) < 1.0)
    assert np.all(np.abs(ds.velocity) < C)
    assert np.all(ds.gamma >= 1.0)


def test_all_finite() -> None:
    ds = generate_dataset(n_samples=5000, random_seed=11)
    for name in COLUMN_NAMES:
        assert np.all(np.isfinite(getattr(ds, name))), name


def test_validation_passes_on_generated_data() -> None:
    ds = generate_dataset(n_samples=2000, random_seed=99)
    report = ds.validate()
    assert report.passed
    assert report.n_rows == 2000
    assert report.max_energy_momentum_error < 1e-12
    assert report.max_proper_time_violation == 0.0


def test_validation_zero_crossing_force() -> None:
    """Force range crossing zero (including F -> 0 rows) must stay physical."""
    ds = generate_dataset(n_samples=2000, force_range=(-1e3, 1e3), random_seed=5)
    assert ds.validate().passed


def test_reproducible_large() -> None:
    a = generate_dataset(n_samples=50_000, random_seed=123)
    b = generate_dataset(n_samples=50_000, random_seed=123)
    np.testing.assert_array_equal(a.momentum, b.momentum)


def test_csv_roundtrip(tmp_path) -> None:
    ds = generate_dataset(n_samples=200, random_seed=1)
    out = tmp_path / "ds.csv"
    ds.to_csv(str(out), index=False)
    assert out.exists()
    loaded = np.loadtxt(out, delimiter=",", skiprows=1)
    assert loaded.shape == (200, len(COLUMN_NAMES))
    np.testing.assert_allclose(loaded, ds.to_numpy(), rtol=1e-12, atol=1e-12)


def test_to_numpy_selected_columns() -> None:
    ds = generate_dataset(n_samples=50, random_seed=0)
    subset = ds.to_numpy(columns=("mass", "force", "time"))
    assert subset.shape == (50, 3)
    np.testing.assert_allclose(subset[:, 0], ds.mass)


def test_to_dataframe(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    ds = generate_dataset(n_samples=100, random_seed=2)
    df = ds.to_dataframe()
    assert list(df.columns) == list(COLUMN_NAMES)
    # CSV via pandas path
    out = tmp_path / "ds_pd.csv"
    ds.to_csv(str(out))
    assert out.exists()


def test_getitem_returns_row_dict() -> None:
    ds = generate_dataset(n_samples=5, random_seed=0)
    row = ds[2]
    assert set(row) == set(COLUMN_NAMES)
    assert row["mass"] == pytest.approx(ds.mass[2], rel=1e-15)


@pytest.mark.parametrize("n", [0, -5, 2.5, "10"])
def test_invalid_n_samples(n) -> None:
    with pytest.raises(InvalidSimulationParameterError):
        generate_dataset(n_samples=n)


def test_invalid_ranges() -> None:
    with pytest.raises(InvalidSimulationParameterError):
        generate_dataset(mass_range=(0.0, 1.0))        # zero mass
    with pytest.raises(VelocityLimitError):
        generate_dataset(beta_range=(0.0, 1.0))        # beta reaches 1
    with pytest.raises(VelocityLimitError):
        generate_dataset(beta_range=(1.0, 1.2))        # above c
    with pytest.raises(InvalidSimulationParameterError):
        generate_dataset(time_range=(-1.0, 1.0))       # negative time
    with pytest.raises(InvalidSimulationParameterError):
        generate_dataset(mass_range=(500.0, 100.0))    # lo > hi
    with pytest.raises(InvalidSimulationParameterError):
        generate_dataset(random_seed="not-a-seed")


def test_point_masses_in_dataset_validated() -> None:
    """Ultra-relativistic regime stays physically valid when validated."""
    ds = generate_dataset(n_samples=2000, beta_range=(0.999, 0.9999), random_seed=8)
    assert ds.validate().passed


def test_performance_smoke() -> None:
    """100k rows must be generated quickly (vectorised)."""
    t0 = time.perf_counter()
    ds = generate_dataset(n_samples=100_000, random_seed=0)
    elapsed = time.perf_counter() - t0
    assert len(ds) == 100_000
    assert ds.validate().passed
    assert elapsed < 30.0, f"100k rows took {elapsed:.1f}s"