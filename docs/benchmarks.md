# Benchmarks

Run the benchmark yourself with:

```bash
uv run python examples/benchmark.py
```

The script reports wall-clock time, samples per second, and the NumPy storage footprint
for 10,000, 100,000, and 1,000,000 vectorized observations. Benchmark values are
machine-dependent and are intentionally not hard-coded in this document.