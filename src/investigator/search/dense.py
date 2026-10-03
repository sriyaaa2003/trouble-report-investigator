"""Dense (embedding) retrieval with an on-disk cache.

Bug-to-bug similarity is symmetric, so both sides are encoded the same way (passage encoder).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]
log = logging.getLogger(__name__)


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Matrix: ...


class FastEmbedEncoder:
    """Local ONNX embeddings (no API key). Default model is small and CPU-friendly."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding  # imported lazily: heavy

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def encode(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # Batch similar-length texts together: padding to the longest text in a batch dominates the cost
        # otherwise (about 2.5x faster here, identical vectors).
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ordered = np.asarray(list(self._model.passage_embed([texts[i] for i in order], batch_size=8)), dtype=np.float32)
        vecs = np.empty_like(ordered)
        vecs[np.asarray(order)] = ordered
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out: Matrix = np.asarray(vecs / norms, dtype=np.float32)
        return out


def _save(path: Path, known: dict[str, NDArray[np.float32]]) -> None:
    """Write the store atomically so an interrupted run never leaves a corrupt cache."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, keys=np.asarray(list(known.keys())), vecs=np.stack(list(known.values())))
    tmp.replace(path)


def encode_cached(
    encoder: TextEncoder, model_name: str, texts: Sequence[str], cache_dir: Path, chunk_size: int = 1000
) -> Matrix:
    """Encode `texts`, embedding only the ones never seen before (incremental, per-text cache).

    The store is one `.npz` per model holding sha256(text) -> vector. Progress is saved after every
    `chunk_size` texts, so an interrupted run resumes where it stopped and growing the dataset never
    re-embeds what was already computed.
    """
    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    safe = hashlib.sha256(model_name.encode()).hexdigest()[:12]
    path = cache_dir / f"emb_store_{safe}.npz"
    known: dict[str, NDArray[np.float32]] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            known = dict(zip(z["keys"].tolist(), z["vecs"], strict=True))
    # de-duplicate identical texts so each is embedded once
    uniq = {keys[i]: texts[i] for i in range(len(texts)) if keys[i] not in known}
    pending = list(uniq.items())
    cache_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        new = encoder.encode([t for _, t in chunk])
        known.update(zip([k for k, _ in chunk], new, strict=True))
        _save(path, known)
        log.info("embedded %d / %d new texts", min(start + chunk_size, len(pending)), len(pending))
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    out: Matrix = np.stack([known[k] for k in keys]).astype(np.float32)
    return out
