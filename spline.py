# spline.py
from __future__ import annotations
import numpy as np
from patsy import dmatrix

EMULATOR_N_TIMESTEPS = 200
EMULATOR_PRESSURE_SPLINE_BASE = 1.6  # tweak in UI


def generate_interior_knots_logspaced(
    n_knots: int,
    base: float,
    time: np.ndarray,
) -> np.ndarray:
    knots = np.logspace(0, 1, n_knots, base=base)
    knots -= knots[0]
    knots /= knots[-1]
    knots *= time[-1] - time[0]
    knots += time[0]
    return knots[1:-1]


def generate_spline_dmatrix(
    n_knots: int,
    base: float,
    time: np.ndarray = np.linspace(
        0, 1, num=EMULATOR_N_TIMESTEPS, endpoint=False
    ),
    include_intercept: bool = True,
) -> np.ndarray:
    knots = generate_interior_knots_logspaced(
        n_knots=n_knots, base=base, time=time
    )
    formula = (
        "cr(time, knots=knots)"
        if include_intercept
        else "cr(time, knots=knots) -1"
    )
    return np.asarray(dmatrix(formula, {"time": time, "knots": knots}))


def fit_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # least squares without regularisation
    return np.linalg.lstsq(X, y, rcond=None)[0]


def reconstruct_waveform(X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return X @ coeffs


def sample_waveforms(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    C = rng.normal(loc=mu, scale=sigma, size=(n_samples, mu.size))
    return C @ X.T
