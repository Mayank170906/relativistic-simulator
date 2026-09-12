"""Measure vectorized dataset-generation throughput on the current machine."""
from __future__ import annotations

import time

from relativistic_simulator import generate_dataset


def main() -> None:
    print(f"{'samples':>12} {'seconds':>12} {'samples/sec':>16} {'MiB':>12}")
    print("-" * 56)
    for n in (10_000, 100_000, 1_000_000):
        start = time.perf_counter()
        dataset = generate_dataset(n_samples=n, random_seed=42)
        elapsed = time.perf_counter() - start
        mib = dataset.to_numpy().nbytes / (1024.0**2)
        print(f"{n:12,d} {elapsed:12.6f} {n / elapsed:16,.0f} {mib:12.2f}")


if __name__ == "__main__":
    main()