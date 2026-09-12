"""Vectorised ground-truth dataset generation for ML research.

The dataset generator draws random physical input parameters (rest mass,
constant force, initial speed ratio, observation time) and evaluates the
exact relativistic constant-force solution at that instant.  Each row is
one synthetic observation: inputs (``mass, force, initial_velocity,
initial_beta, time``) plus ground-truth outputs (``position, velocity,
beta, gamma, momentum, kinetic_energy, total_energy, proper_time``).

Everything is vectorised with NumPy (no per-row Python loops), so
generating 100k–1M rows is practical on a laptop.

Reproducibility
---------------
``random_seed`` is threaded through :class:`numpy.random.Generator`
(``np.random.default_rng``).  No global randomness is used, so the same
seed always produces the same dataset for a given package version.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import C
from .dynamics import analyze_constant_force
from .exceptions import InvalidSimulationParameterError, VelocityLimitError

__all__ = ["RelativisticDataset", "generate_dataset"]

INPUT_COLUMNS: tuple[str, ...] = (
    "mass",
    "force",
    "initial_velocity",
    "initial_beta",
    "time",
)
OUTPUT_COLUMNS: tuple[str, ...] = (
    "position",
    "velocity",
    "beta",
    "gamma",
    "momentum",
    "kinetic_energy",
    "total_energy",
    "proper_time",
)
COLUMN_NAMES: tuple[str, ...] = INPUT_COLUMNS + OUTPUT_COLUMNS
@dataclass(frozen=True)
class RelativisticDataset:
    """Columnar ground-truth dataset of synthetic relativistic observations.

    Parameters
    ----------
    n_samples : int
        Number of rows.
    mass, force, initial_velocity, initial_beta, time : np.ndarray
        Input columns (SI: kg, N, m/s, dimensionless, s).
    position, velocity, beta, gamma, momentum, kinetic_energy, total_energy,
    proper_time : np.ndarray
        Ground-truth output columns at ``time``.
    random_seed : int | None
        Seed used by the generator (``None`` = non-reproducible).
    mass_range, force_range, beta_range, time_range : tuple[float, float]
        Parameter ranges the samples were drawn from.
    initial_position : float, default 0.0
        Initial position (m) shared by all rows.

    Examples
    --------
    >>> from relativistic_simulator import generate_dataset
    >>> ds = generate_dataset(n_samples=10_000, random_seed=42)
    >>> ds.to_dataframe().shape  # doctest: +SKIP  (pandas optional)
    """

    n_samples: int
    # inputs
    mass: np.ndarray
    force: np.ndarray
    initial_velocity: np.ndarray
    initial_beta: np.ndarray
    time: np.ndarray
    # outputs
    position: np.ndarray
    velocity: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    momentum: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    proper_time: np.ndarray
    # metadata
    random_seed: int | None
    mass_range: tuple[float, float]
    force_range: tuple[float, float]
    beta_range: tuple[float, float]
    time_range: tuple[float, float]
    initial_position: float = 0.0

    def __post_init__(self) -> None:
        n = int(self.n_samples)
        for name in COLUMN_NAMES:
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(
                    f"column {name!r} must be a 1-D array of length {n}, got {arr.shape}."
                )

    # ------------------------------------------------------------------ #
    @property
    def input_columns(self) -> tuple[str, ...]:
        return INPUT_COLUMNS

    @property
    def output_columns(self) -> tuple[str, ...]:
        return OUTPUT_COLUMNS

    @property
    def columns(self) -> tuple[str, ...]:
        return COLUMN_NAMES

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict[str, float]:
        """Return row ``index`` as a dict of float scalars."""
        return {name: float(np.asarray(getattr(self, name))[index]) for name in COLUMN_NAMES}

    def __repr__(self) -> str:
        return (
            f"RelativisticDataset(n_samples={self.n_samples}, "
            f"mass_range={self.mass_range}, force_range={self.force_range}, "
            f"beta_range={self.beta_range}, time_range={self.time_range}, "
            f"random_seed={self.random_seed!r})"
        )

    # ------------------------------------------------------------------ #
    # Conversions
    # ------------------------------------------------------------------ #
    def to_numpy(self, columns: tuple[str, ...] | None = None) -> np.ndarray:
        """Return selected columns as a float64 array of shape ``(n, k)``.

        Parameters
        ----------
        columns : tuple[str, ...] | None
            Which columns to stack (default: all ``COLUMN_NAMES``).

        Returns
        -------
        np.ndarray
            The selected columns, row-major.
        """
        selected = COLUMN_NAMES if columns is None else tuple(columns)
        return np.column_stack(
            [np.asarray(getattr(self, name), dtype=np.float64) for name in selected]
        )

    def to_dict(self, columns: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        """Return a dict mapping column name -> float64 array."""
        selected = COLUMN_NAMES if columns is None else tuple(columns)
        return {
            name: np.asarray(getattr(self, name), dtype=np.float64) for name in selected
        }

    def to_dataframe(self, columns: tuple[str, ...] | None = None) -> "pd.DataFrame":
        """Return a pandas DataFrame (requires the optional ``pandas`` extra)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover (wheel tests cover it)
            raise ImportError(
                "RelativisticDataset.to_dataframe() requires the optional "
                "'pandas' extra: `pip install 'relativistic-simulator[pandas]'`."
            ) from exc
        frame = pd.DataFrame(self.to_dict(columns))
        frame.attrs["random_seed"] = self.random_seed
        frame.attrs["mass_range"] = self.mass_range
        frame.attrs["force_range"] = self.force_range
        frame.attrs["beta_range"] = self.beta_range
        frame.attrs["time_range"] = self.time_range
        return frame

    def to_csv(self, path: str, columns: tuple[str, ...] | None = None, **kwargs: Any) -> None:
        """Write the dataset to a CSV file.

        Uses ``pandas.DataFrame.to_csv`` when pandas is installed (for
        maximum interoperability) and falls back to ``numpy.savetxt``
        otherwise, so the core never depends on pandas.

        Parameters
        ----------
        path : str
            Destination file path.
        columns : tuple[str, ...] | None
            Which columns to write (default: all).
        **kwargs
            Forwarded to ``pandas.DataFrame.to_csv`` (e.g. ``index=False``).
        """
        try:
            import pandas as pd
        except ImportError:
            data = self.to_numpy(columns)
            names = COLUMN_NAMES if columns is None else tuple(columns)
            np.savetxt(
                path, data, delimiter=",", header=",".join(names), comments="", fmt="%.17g"
            )
        else:
            frame = pd.DataFrame(self.to_dict(columns))
            kwargs.setdefault("index", False)
            frame.to_csv(path, **kwargs)

    def validate(self, **kwargs: Any):
        """Validate the physical consistency of every row (vectorised).

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`~relativistic_simulator.validate_dataset`.

        Returns
        -------
        DatasetValidation
            Quantitative per-relation errors and ``passed`` flag.
        """
        from .validation import validate_dataset

        return validate_dataset(self, **kwargs)
def _validate_range(name: str, rng: tuple[float, float], *, lower: float | None = None,
                    upper: float | None = None) -> tuple[float, float]:
    """Validate a sampling range ``(lo, hi)`` with optional hard bounds."""
    if not isinstance(rng, (tuple, list)) or len(rng) != 2:
        raise InvalidSimulationParameterError(
            f"{name} must be a (lo, hi) tuple; got {rng!r}."
        )
    lo, hi = (float(rng[0]), float(rng[1]))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise InvalidSimulationParameterError(f"{name} bounds must be finite; got {rng!r}.")
    if lo > hi:
        raise InvalidSimulationParameterError(
            f"{name} must satisfy lo <= hi; got {rng!r}."
        )
    if lower is not None and lo < lower:
        raise InvalidSimulationParameterError(
            f"{name} lower bound must be >= {lower}; got {rng!r}."
        )
    if upper is not None and hi > upper:
        raise InvalidSimulationParameterError(
            f"{name} upper bound must be <= {upper}; got {rng!r}."
        )
    return lo, hi


def generate_dataset(
    n_samples: int = 100_000,
    *,
    mass_range: tuple[float, float] = (100.0, 10_000.0),
    force_range: tuple[float, float] = (1e4, 1e7),
    beta_range: tuple[float, float] = (0.0, 0.9),
    time_range: tuple[float, float] = (0.0, 100.0),
    initial_position: float = 0.0,
    random_seed: int | None = None,
) -> RelativisticDataset:
    """Generate a ground-truth dataset of synthetic relativistic observations.

    Each row draws ``mass``, ``force``, ``initial_beta`` and ``time``
    uniformly from the requested ranges and evaluates the *exact*
    constant-force relativistic solution at that instant.  The sampled
    initial-velocity column is ``initial_velocity = beta * c``.

    Parameters
    ----------
    n_samples : int, default 100_000
        Number of rows to generate (``>= 1``).
    mass_range : tuple[float, float], default (100.0, 10000.0)
        Rest-mass sampling range in kg (strictly positive).
    force_range : tuple[float, float], default (1e4, 1e7)
        Applied-force sampling range in N (finite; may cross or include 0).
    beta_range : tuple[float, float], default (0.0, 0.9)
        Initial speed-ratio sampling range.  Must satisfy
        ``-1 < lo <= hi < 1`` (initial velocities stay subluminal).
        Common regimes: low-relativistic ``(0, 0.5]``, relativistic
        ``(0.5, 0.9]``, ultra-relativistic ``(0.9, 0.9999]``.
    time_range : tuple[float, float], default (0.0, 100.0)
        Observation-time sampling range in seconds (``t >= 0``).
    initial_position : float, default 0.0
        Initial position in m, shared by all rows.
    random_seed : int | None, default None
        Seed for the reproducible NumPy generator.  ``None`` draws from
        fresh entropy (non-reproducible, but still uses the modern
        Generator API and does not touch global state).

    Returns
    -------
    RelativisticDataset
        Columnar dataset: input columns ``mass, force, initial_velocity,
        initial_beta, time`` and ground-truth outputs ``position, velocity,
        beta, gamma, momentum, kinetic_energy, total_energy, proper_time``.

    Raises
    ------
    InvalidSimulationParameterError
        If the ranges, counts or seed are invalid.
    InvalidMassError
        If ``mass_range`` is not strictly positive.
    VelocityLimitError
        If ``beta_range`` touches ``|beta| >= 1``.

    Notes
    -----
    The outputs are the **ground truth from known physics**: this package
    generates data from equations; it does not "discover" physics.  A later
    ML experiment (external to this package) may study whether a model can
    learn this mapping and generalise toward the ultra-relativistic regime.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise InvalidSimulationParameterError(
            f"n_samples must be a positive integer; got {n_samples!r}."
        )
    n = int(n_samples)
    if n < 1:
        raise InvalidSimulationParameterError(f"n_samples must be >= 1; got {n}.")

    mass_lo, mass_hi = _validate_range("mass_range", mass_range)
    if mass_lo <= 0.0:
        raise InvalidSimulationParameterError(
            "mass_range must be strictly positive (m > 0 kg); got "
            f"({mass_lo}, {mass_hi})."
        )
    force_lo, force_hi = _validate_range("force_range", force_range)
    beta_lo, beta_hi = _validate_range("beta_range", beta_range)
    if not (beta_lo > -1.0 and beta_hi < 1.0):
        raise VelocityLimitError(
            "beta_range must satisfy -1 < lo <= hi < 1 strictly (initial "
            f"velocities of massive particles stay subluminal); got ({beta_lo}, {beta_hi})."
        )
    t_lo, t_hi = _validate_range("time_range", time_range, lower=0.0)

    x0 = float(initial_position)
    if not np.isfinite(x0):
        raise InvalidSimulationParameterError(f"initial_position must be finite; got {x0!r}.")

    if random_seed is not None and not isinstance(random_seed, (int, np.integer)):
        raise InvalidSimulationParameterError(
            f"random_seed must be an integer or None; got {random_seed!r}."
        )
    seed = int(random_seed) if random_seed is not None else None

    rng = np.random.default_rng(seed)
    mass = rng.uniform(mass_lo, mass_hi, n)
    force = rng.uniform(force_lo, force_hi, n)
    initial_beta = rng.uniform(beta_lo, beta_hi, n)
    time = rng.uniform(t_lo, t_hi, n)
    initial_velocity = initial_beta * C

    if np.any(np.abs(initial_beta) >= 1.0):  # defensive; range validation already guards
        raise VelocityLimitError(
            "sampled |initial_beta| reached 1.0; widen the margin in beta_range (|beta| < 1)."
        )

    state = analyze_constant_force(
        mass=mass,
        force=force,
        initial_velocity=initial_velocity,
        initial_position=x0 * np.ones(n),
        time=time,
    )

    return RelativisticDataset(
        n_samples=n,
        mass=mass,
        force=force,
        initial_velocity=initial_velocity,
        initial_beta=initial_beta,
        time=time,
        position=state["position"],
        velocity=state["velocity"],
        beta=state["beta"],
        gamma=state["gamma"],
        momentum=state["momentum"],
        kinetic_energy=state["kinetic_energy"],
        total_energy=state["total_energy"],
        proper_time=state["proper_time"],
        random_seed=seed,
        mass_range=(mass_lo, mass_hi),
        force_range=(force_lo, force_hi),
        beta_range=(beta_lo, beta_hi),
        time_range=(t_lo, t_hi),
        initial_position=x0,
    )