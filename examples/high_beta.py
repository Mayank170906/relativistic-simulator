"""Example 2: high-beta simulations.

Sweeps initial beta through the relativistic ladder
0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999 and shows how a
constant force barely moves the speed toward c (velocity saturation),
while momentum, gamma and kinetic energy keep growing without bound.

This is the regime the ground-truth generator is designed for.
"""
from __future__ import annotations

import numpy as np

from relativistic_simulator import C, simulate

BETAS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999)

MASS = 1000.0   # kg
FORCE = 1e6     # N
T = 100.0       # s


def main() -> None:
    print(f"High-beta sweep: m = {MASS:g} kg, F = {FORCE:g} N, t = {T:g} s")
    print(f"{'beta0':>8} | {'beta_f':>10} | {'gamma_f':>12} | {'p_f (kg m/s)':>13} "
          f"| {'K_f (J)':>10} | {'d(beta_f-beta0)':>14}")
    print("-" * 84)

    for b0 in BETAS:
        result = simulate(mass=MASS, force=FORCE, initial_velocity=b0 * C, duration=T, dt=0.05)
        report = result.validate()
        assert report.passed, report.summary()
        beta_f = float(result.beta[-1])
        gamma_f = float(result.gamma[-1])
        p_f = float(result.momentum[-1])
        K_f = float(result.kinetic_energy[-1])
        print(f"{b0:8.5f} | {beta_f:10.7f} | {gamma_f:12.6f} | {p_f:13.6e} "
              f"| {K_f:10.3e} | {beta_f - b0:+14.3e}")

    print()
    print("Observe: the impulse F*t = 1e8 kg*m/s barely changes beta near c,")
    print("while gamma and momentum keep increasing. Velocity saturates below c.")


if __name__ == "__main__":
    main()
