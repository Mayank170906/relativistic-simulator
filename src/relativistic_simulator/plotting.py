"""Optional matplotlib plotting utilities.

Matplotlib is **not** a required dependency.  Importing this module never
fails; the individual plotting functions raise a helpful :class:`ImportError`
only when matplotlib is missing from the environment.

Plots are returned as ``(fig, axes)`` so they work headless (save to file,
``path=...``) and interactively (``show=True``).
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "plot_trajectory",
    "plot_energy_vs_beta",
    "plot_gamma_vs_beta",
    "plot_relativistic_vs_classical",
]


def _get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover (wheel tests cover it)
        raise ImportError(
            "matplotlib is an optional dependency of relativistic-simulator. "
            "Install it with `pip install 'relativistic-simulator[plot]'`."
        ) from exc
    return plt


def _finish(fig, axes: Any, path: str | None, show: bool) -> tuple[Any, Any]:
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        fig.show()
    return fig, axes


def plot_trajectory(
    trajectory: Any,
    *,
    path: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (14.0, 9.0),
) -> tuple[Any, Any]:
    """Plot the time history of a :class:`Trajectory`.

    Produces a multi-panel figure with velocity, beta, gamma, momentum,
    kinetic energy, position and proper time vs coordinate time.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to plot.
    path : str | None
        Optional output file path (e.g. ``"traj.png"``); saves a PNG.
    show : bool, default True
        Whether to call ``plt.show()``.
    figsize : tuple[float, float]
        Matplotlib figure size in inches.

    Returns
    -------
    (Figure, ndarray of Axes)
        For further styling or saving.
    """
    plt = _get_pyplot()
    panels = [
        ("velocity", "v (m/s)"),
        ("beta", "$\\beta = v/c$"),
        ("gamma", "$\\gamma$"),
        ("momentum", "p (kg·m/s)"),
        ("kinetic_energy", "K (J)"),
        ("position", "x (m)"),
        ("proper_time", "$\\tau$ (s)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = np.asarray(axes).ravel()
    t = np.asarray(trajectory.time)
    for ax, (name, ylabel) in zip(axes, panels):
        ax.plot(t, np.asarray(getattr(trajectory, name)))
        ax.set_xlabel("t (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[7].axis("off")
    fig.suptitle(
        f"m = {trajectory.mass:g} kg, F = {trajectory.force:g} N, "
        f"v0 = {trajectory.initial_velocity:g} m/s ({trajectory.method})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, axes, path, show)


def plot_gamma_vs_beta(
    *,
    max_beta: float = 0.9999,
    n_points: int = 500,
    path: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot :math:`\\gamma = (1-\\beta^2)^{-1/2}` vs :math:`\\beta`."""
    plt = _get_pyplot()
    from .relativity import gamma_from_beta

    beta = np.linspace(0.0, max_beta, n_points)
    gamma = gamma_from_beta(beta)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(beta, gamma)
    ax.set_xlabel("$\\beta = v/c$")
    ax.set_ylabel("$\\gamma$")
    ax.set_title(r"Lorentz factor vs $\beta$")
    ax.grid(True, alpha=0.3)
    return _finish(fig, ax, path, show)


def plot_energy_vs_beta(
    mass: float = 1.0,
    *,
    max_beta: float = 0.9999,
    n_points: int = 500,
    log_scale: bool = True,
    path: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot relativistic kinetic energy :math:`K` vs :math:`\\beta` for a mass."""
    plt = _get_pyplot()
    from .relativity import kinetic_energy_from_velocity
    from .constants import C

    beta = np.linspace(0.0, max_beta, n_points)
    kinetic = kinetic_energy_from_velocity(mass * np.ones_like(beta), beta * C)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(beta, kinetic)
    ax.set_xlabel("$\\beta = v/c$")
    ax.set_ylabel("$K$ (J)")
    ax.set_title(rf"Relativistic kinetic energy ($m = {mass:g}$ kg)")
    if log_scale and kinetic[0] > 0:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    return _finish(fig, ax, path, show)


def plot_relativistic_vs_classical(
    mass: float,
    force: float,
    initial_velocity: float = 0.0,
    *,
    duration: float = 10.0,
    dt: float = 0.01,
    path: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Compare relativistic vs classical kinetic energy along a trajectory.

    Parameters
    ----------
    mass : float
        Rest mass in kg.
    force : float
        Constant force in N.
    initial_velocity : float, default 0.0
        Initial velocity in m/s.
    duration, dt : float
        Simulation horizon and time step (s).
    path : str | None
        Optional PNG output path.
    show : bool, default True
        Whether to ``plt.show()``.

    Returns
    -------
    (Figure, Axes)
        A two-panel figure (time trace and relative difference).
    """
    plt = _get_pyplot()
    from .simulator import simulate
    from .classical import classical_kinetic_energy

    traj = simulate(mass, force, initial_velocity=initial_velocity, duration=duration, dt=dt)
    t = np.asarray(traj.time)
    k_rel = np.asarray(traj.kinetic_energy)
    k_cl = classical_kinetic_energy(mass, np.asarray(traj.velocity))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(k_rel > 0, np.abs(k_cl - k_rel) / k_rel, 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(t, k_rel, label="relativistic $K$")
    ax1.plot(t, k_cl, "--", label="classical $K$")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("K (J)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(t, rel_diff, label=r"$|K_{cl} - K_{rel}|/K_{rel}$")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("relative difference")
    ax2.set_title("Where does Newtonian energy diverge?")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    return _finish(fig, (ax1, ax2), path, show)