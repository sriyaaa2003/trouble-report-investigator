"""Combine score matrices from different retrievers and apply the causal (time) mask."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Scores = NDArray[np.float32]
NEG = np.float32(-1e9)


def causal_mask(scores: Scores, corpus_created: NDArray[np.float64], query_created: NDArray[np.float64]) -> Scores:
    """Only documents created strictly before the query may be retrieved (no peeking at the future)."""
    out = scores.copy()
    out[corpus_created[None, :] >= query_created[:, None]] = NEG
    return out


def ranks(scores: Scores) -> NDArray[np.int32]:
    """0-based rank of every document for every query (best = 0)."""
    order = np.argsort(-scores, axis=1)
    r = np.empty_like(order, dtype=np.int32)
    rows = np.arange(scores.shape[0])[:, None]
    r[rows, order] = np.arange(scores.shape[1], dtype=np.int32)[None, :]
    return r


def rrf(mats: Sequence[Scores], k: int = 60, weights: Sequence[float] | None = None) -> Scores:
    """Reciprocal rank fusion. Masked entries (NEG) stay masked."""
    w = list(weights) if weights is not None else [1.0] * len(mats)
    out = np.zeros_like(mats[0], dtype=np.float32)
    for m, wi in zip(mats, w, strict=True):
        out += np.float32(wi) / (np.float32(k) + 1.0 + ranks(m).astype(np.float32))
    out[mats[0] <= NEG / 2] = NEG
    return out


def zfuse(mats: Sequence[Scores], weights: Sequence[float]) -> Scores:
    """Weighted sum of per-query z-scored scores over the eligible (non-masked) documents."""
    out = np.zeros_like(mats[0], dtype=np.float32)
    eligible = mats[0] > NEG / 2
    for m, w in zip(mats, weights, strict=True):
        z = np.zeros_like(m, dtype=np.float32)
        for i in range(m.shape[0]):
            e = eligible[i]
            if not e.any():
                continue
            v = m[i, e]
            sd = v.std()
            z[i, e] = (v - v.mean()) / (sd if sd > 1e-9 else 1.0)
        out += np.float32(w) * z
    out[~eligible] = NEG
    return out
