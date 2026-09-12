"""Example 1: basic simulation of a massive particle under constant force.

A 1000 kg object starting at rest, pushed by a constant 1e6 N force in +x
for 100 seconds.  In the classical picture this would reach 1e5 m/s
(0.03% of c) after 100 s; the relativistic solution is essentially identical
because beta stays tiny.
"""
from __future__ import annotations

import numpy as np

from relativistic_simulator import C, simulate

def main() -> None:
    result = simulate(
        mass=1000.0,           # kg
        force=1e6,             # N
        initial_velocity=0.0,  # m/s (at rest)
        duration=100.0,        # s
        dt=0.01,               # s
    )

    final = result.final_state()
    classical_v = (1e6 / 1000.0) * 100.0  # v = (F/m) * t, Newtonian
    classical_K = 0.5 * 1000.0 * classical_v**2

    print("Basic simulation: 1000 kg at rest, F = 1e6 N, t = 100 s")
    print(f"  n_points            : {result.n_points}")
    print(f"  final position      : {final['position']:.6e} m")
    print(f"  final velocity      : {final['velocity']:.6e} m/s  (beta = {final['beta']:.6e})")
    print(f"  classical velocity  : {classical_v:.6e} m/s")
    print(f"  final gamma         : {final['gamma']:.12f}")
    print(f"  final momentum      : {final['momentum']:.6e} kg*m/s  (classical m*v = {1000*classical_v:.6e})")
    print(f"  final kinetic energy: {final['kinetic_energy']:.6e} J  (classical = {classical_K:.6e} J)")
    print(f"  proper time         : {final['proper_time']:.9f} s (<= 100 s: {final['proper_time'] <= 100.0})")

    report = result.validate()
    print()
    print(report.summary())


if __name__ == "__main__":
    main()
