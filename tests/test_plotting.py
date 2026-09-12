"""Smoke tests for the optional matplotlib plotting utilities."""
from __future__ import annotations

import matplotlib  # noqa: F401

matplotlib.use("Agg")

import pytest  # noqa: E402

from relativistic_simulator import simulate  # noqa: E402


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_plot_trajectory_saves_png(tmp_path) -> None:
    from relativistic_simulator.plotting import plot_trajectory

    traj = simulate(mass=1000.0, force=1e6, duration=10.0, dt=0.01)
    out = tmp_path / "traj.png"
    fig, axes = plot_trajectory(traj, path=str(out), show=False)
    assert out.exists()
    assert out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_plot_energy_vs_beta(tmp_path) -> None:
    from relativistic_simulator.plotting import plot_energy_vs_beta

    fig, ax = plot_energy_vs_beta(mass=1.0, path=str(tmp_path / "e.png"), show=False)
    assert (tmp_path / "e.png").exists()
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_plot_gamma_vs_beta(tmp_path) -> None:
    from relativistic_simulator.plotting import plot_gamma_vs_beta

    fig, ax = plot_gamma_vs_beta(path=str(tmp_path / "g.png"), show=False)
    assert (tmp_path / "g.png").exists()
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_plot_classical_comparison(tmp_path) -> None:
    from relativistic_simulator.plotting import plot_relativistic_vs_classical

    fig, axes = plot_relativistic_vs_classical(1000.0, 1e6, path=str(tmp_path / "c.png"), show=False)
    assert (tmp_path / "c.png").exists()
    import matplotlib.pyplot as plt

    plt.close(fig)