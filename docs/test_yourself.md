The correct API in published `0.1.3` package is:

```python
simulate(
    mass=...,
    force=...,
    initial_velocity=...,
    duration=...,
    dt=...,
)
```


Here is the **fully 18-test script** using the API.

```python
import numpy as np

from relativistic_simulator import (
    C,
    simulate,
    generate_dataset,
)


# ============================================================
# 1. Basic simulation
# ============================================================

def test_basic_simulation():
    traj = simulate(
        mass=1000.0,
        force=1e6,
        initial_velocity=0.0,
        duration=10.0,
        dt=0.1,
    )

    assert len(traj.time) > 1
    assert len(traj.position) == len(traj.time)
    assert len(traj.velocity) == len(traj.time)
    assert len(traj.momentum) == len(traj.time)

    assert np.isclose(traj.time[0], 0.0)
    assert np.isclose(traj.position[0], 0.0)
    assert np.isclose(traj.velocity[0], 0.0)


# ============================================================
# 2. Time grid
# ============================================================

def test_time_grid():
    traj = simulate(
        mass=1000.0,
        force=1e6,
        initial_velocity=0.0,
        duration=10.0,
        dt=0.1,
    )

    assert np.isclose(traj.time[0], 0.0)
    assert np.isclose(traj.time[-1], 10.0)

    differences = np.diff(traj.time)

    assert np.allclose(
        differences,
        0.1,
        rtol=1e-10,
        atol=1e-12,
    )


# ============================================================
# 3. Initial conditions
# ============================================================

def test_initial_conditions():
    mass = 1000.0
    force = 1e6
    velocity = 0.2 * C

    traj = simulate(
        mass=mass,
        force=force,
        initial_velocity=velocity,
        duration=5.0,
        dt=0.05,
    )

    expected_gamma = 1.0 / np.sqrt(
        1.0 - (velocity / C) ** 2
    )

    expected_momentum = (
        expected_gamma * mass * velocity
    )

    assert np.isclose(
        traj.velocity[0],
        velocity,
        rtol=1e-10,
    )

    assert np.isclose(
        traj.beta[0],
        velocity / C,
        rtol=1e-10,
    )

    assert np.isclose(
        traj.gamma[0],
        expected_gamma,
        rtol=1e-10,
    )

    assert np.isclose(
        traj.momentum[0],
        expected_momentum,
        rtol=1e-10,
    )


# ============================================================
# 4. Constant force -> momentum changes linearly
#
# p(t) = p0 + F*t
# ============================================================

def test_constant_force_momentum():
    mass = 1000.0
    force = 1e6
    initial_velocity = 0.2 * C

    traj = simulate(
        mass=mass,
        force=force,
        initial_velocity=initial_velocity,
        duration=10.0,
        dt=0.01,
    )

    gamma0 = 1.0 / np.sqrt(
        1.0 - (initial_velocity / C) ** 2
    )

    p0 = gamma0 * mass * initial_velocity

    expected_momentum = (
        p0 + force * traj.time
    )

    assert np.allclose(
        traj.momentum,
        expected_momentum,
        rtol=1e-10,
        atol=1e-3,
    )


# ============================================================
# 5. Relativistic momentum identity
#
# p = gamma*m*v
# ============================================================

def test_momentum_identity():
    mass = 1000.0

    traj = simulate(
        mass=mass,
        force=1e7,
        initial_velocity=0.3 * C,
        duration=10.0,
        dt=0.01,
    )

    expected = (
        traj.gamma
        * mass
        * traj.velocity
    )

    assert np.allclose(
        traj.momentum,
        expected,
        rtol=1e-10,
        atol=1e-3,
    )


# ============================================================
# 6. Relativistic energy identity
#
# E = gamma*m*c^2
# ============================================================

def test_energy_identity():
    mass = 1000.0

    traj = simulate(
        mass=mass,
        force=1e7,
        initial_velocity=0.3 * C,
        duration=10.0,
        dt=0.01,
    )

    expected = (
        traj.gamma
        * mass
        * C**2
    )

    assert np.allclose(
        traj.total_energy,
        expected,
        rtol=1e-10,
        atol=1e8,
    )


# ============================================================
# 7. Energy-momentum invariant
#
# E^2 = (p*c)^2 + (m*c^2)^2
# ============================================================

def test_energy_momentum_invariant():
    mass = 1000.0

    traj = simulate(
        mass=mass,
        force=1e7,
        initial_velocity=0.5 * C,
        duration=10.0,
        dt=0.01,
    )

    lhs = traj.total_energy ** 2

    rhs = (
        (traj.momentum * C) ** 2
        + (mass * C**2) ** 2
    )

    assert np.allclose(
        lhs,
        rhs,
        rtol=1e-10,
        atol=1e30,
    )


# ============================================================
# 8. Velocity must remain below c
# ============================================================

def test_velocity_bound():
    traj = simulate(
        mass=1000.0,
        force=1e9,
        initial_velocity=0.0,
        duration=100.0,
        dt=0.1,
    )

    assert np.all(
        np.abs(traj.velocity) < C
    )


# ============================================================
# 9. Beta = v/c
# ============================================================

def test_beta_identity():
    traj = simulate(
        mass=1000.0,
        force=1e7,
        initial_velocity=0.2 * C,
        duration=20.0,
        dt=0.1,
    )

    expected_beta = (
        traj.velocity / C
    )

    assert np.allclose(
        traj.beta,
        expected_beta,
        rtol=1e-10,
        atol=1e-12,
    )


# ============================================================
# 10. Gamma = 1/sqrt(1-beta^2)
# ============================================================

def test_gamma_identity():
    traj = simulate(
        mass=1000.0,
        force=1e7,
        initial_velocity=0.2 * C,
        duration=20.0,
        dt=0.1,
    )

    expected_gamma = (
        1.0
        / np.sqrt(1.0 - traj.beta**2)
    )

    assert np.allclose(
        traj.gamma,
        expected_gamma,
        rtol=1e-10,
        atol=1e-10,
    )


# ============================================================
# 11. dx/dt = v
# ============================================================

def test_position_velocity_consistency():
    traj = simulate(
        mass=1000.0,
        force=1e6,
        initial_velocity=0.1 * C,
        duration=10.0,
        dt=0.001,
    )

    numerical_velocity = np.gradient(
        traj.position,
        traj.time,
    )

    assert np.allclose(
        numerical_velocity[2:-2],
        traj.velocity[2:-2],
        rtol=1e-5,
        atol=10.0,
    )


# ============================================================
# 12. Proper time
#
# dτ = dt/gamma
# therefore τ <= t
# ============================================================

def test_proper_time():
    traj = simulate(
        mass=1000.0,
        force=1e7,
        initial_velocity=0.5 * C,
        duration=20.0,
        dt=0.01,
    )

    assert np.all(
        traj.proper_time
        <= traj.time + 1e-10
    )

    assert np.all(
        np.diff(traj.proper_time)
        >= -1e-10
    )


# ============================================================
# 13. Zero force -> velocity remains constant
# ============================================================

def test_zero_force_velocity():
    velocity = 0.4 * C

    traj = simulate(
        mass=1000.0,
        force=0.0,
        initial_velocity=velocity,
        duration=20.0,
        dt=0.1,
    )

    assert np.allclose(
        traj.velocity,
        velocity,
        rtol=1e-10,
        atol=1e-10,
    )


# ============================================================
# 14. Zero force -> momentum remains constant
# ============================================================

def test_zero_force_momentum():
    velocity = 0.4 * C

    traj = simulate(
        mass=1000.0,
        force=0.0,
        initial_velocity=velocity,
        duration=20.0,
        dt=0.1,
    )

    assert np.allclose(
        traj.momentum,
        traj.momentum[0],
        rtol=1e-10,
        atol=1e-3,
    )


# ============================================================
# 15. Analytic vs numerical RK4
# ============================================================

def test_analytic_vs_numerical():

    kwargs = dict(
        mass=1000.0,
        force=1e7,
        initial_velocity=0.2 * C,
        duration=10.0,
        dt=0.01,
    )

    analytic = simulate(
        **kwargs,
        method="analytic",
    )

    numerical = simulate(
        **kwargs,
        method="numerical",
    )

    assert np.allclose(
        analytic.time,
        numerical.time,
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.allclose(
        analytic.position,
        numerical.position,
        rtol=1e-6,
        atol=1e-2,
    )

    assert np.allclose(
        analytic.velocity,
        numerical.velocity,
        rtol=1e-7,
        atol=1e-2,
    )

    assert np.allclose(
        analytic.momentum,
        numerical.momentum,
        rtol=1e-8,
        atol=1e-2,
    )


# ============================================================
# 16. Dataset size
# ============================================================

def test_dataset_size():

    dataset = generate_dataset(
        n_samples=100,
        random_seed=42,
    )

    data = dataset.to_numpy()

    assert data.shape[0] == 100


# ============================================================
# 17. Dataset columns
# ============================================================

def test_dataset_columns():

    dataset = generate_dataset(
        n_samples=10,
        random_seed=42,
    )

    expected_inputs = (
        "mass",
        "force",
        "initial_velocity",
        "initial_beta",
        "time",
    )

    expected_outputs = (
        "position",
        "velocity",
        "beta",
        "gamma",
        "momentum",
        "kinetic_energy",
        "total_energy",
        "proper_time",
    )

    assert dataset.input_columns == expected_inputs
    assert dataset.output_columns == expected_outputs


# ============================================================
# 18. Dataset reproducibility
# ============================================================

def test_dataset_reproducibility():

    dataset1 = generate_dataset(
        n_samples=100,
        random_seed=123,
    )

    dataset2 = generate_dataset(
        n_samples=100,
        random_seed=123,
    )

    data1 = dataset1.to_numpy()
    data2 = dataset2.to_numpy()

    assert np.array_equal(
        data1,
        data2,
    )


# ============================================================
# RUN ALL TESTS
# ============================================================

tests = [
    test_basic_simulation,
    test_time_grid,
    test_initial_conditions,
    test_constant_force_momentum,
    test_momentum_identity,
    test_energy_identity,
    test_energy_momentum_invariant,
    test_velocity_bound,
    test_beta_identity,
    test_gamma_identity,
    test_position_velocity_consistency,
    test_proper_time,
    test_zero_force_velocity,
    test_zero_force_momentum,
    test_analytic_vs_numerical,
    test_dataset_size,
    test_dataset_columns,
    test_dataset_reproducibility,
]


passed = 0
failed = 0

print("=" * 60)
print("RELATIVISTIC SIMULATOR TEST SUITE")
print("=" * 60)

for test in tests:

    try:
        test()

        print(f"PASS  {test.__name__}")

        passed += 1

    except Exception as e:

        print(f"FAIL  {test.__name__}")
        print(
            f"      {type(e).__name__}: {e}"
        )

        failed += 1


print("=" * 60)
print(
    f"RESULT: {passed}/{len(tests)} tests passed"
)

if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(
        f"{failed} TEST(S) FAILED"
    )

print("=" * 60)

assert failed == 0
```

The important correction is that the simulation calls now use:

```python
duration=10.0,
dt=0.01,
```


