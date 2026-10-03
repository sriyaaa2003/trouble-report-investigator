"""Bootstrap confidence intervals over queries (paired where two systems are compared)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float

    def fmt(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def bootstrap(values: NDArray[np.float64], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> Estimate:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return Estimate(float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_diff(a: NDArray[np.float64], b: NDArray[np.float64], n_boot: int = 2000, seed: int = 0) -> Estimate:
    """Mean of (a - b) with a bootstrap interval; resamples queries jointly, so the pairing is kept."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), n_boot, seed)
