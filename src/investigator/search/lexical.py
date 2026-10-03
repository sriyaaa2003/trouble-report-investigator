"""BM25 lexical retrieval over a sparse term-document matrix."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

Scores = NDArray[np.float32]


class BM25:
    """Okapi BM25 with the usual idf variant that never goes negative.

    `scores(queries)` returns a dense (n_queries, n_docs) float32 matrix. Query terms are treated as a set.
    """

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        tfs: list[float] = []
        lengths = np.zeros(len(docs), dtype=np.float64)
        df: Counter[int] = Counter()
        for i, doc in enumerate(docs):
            counts = Counter(doc)
            lengths[i] = len(doc)
            for term, count in counts.items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                tfs.append(float(count))
                df[j] += 1
        n = len(docs)
        self.n_docs = n
        avgdl = float(lengths.mean()) if n else 1.0
        idf = np.zeros(len(self.vocab), dtype=np.float64)
        for j, d in df.items():
            idf[j] = math.log(1.0 + (n - d + 0.5) / (d + 0.5))
        tf = np.asarray(tfs)
        r = np.asarray(rows, dtype=np.int64)
        c = np.asarray(cols, dtype=np.int64)
        norm = k1 * (1.0 - b + b * lengths[r] / max(avgdl, 1e-9))
        weights = idf[c] * (tf * (k1 + 1.0)) / (tf + norm)
        self._w = sparse.csr_matrix((weights.astype(np.float32), (r, c)), shape=(n, len(self.vocab)))
        self._idf = idf

    def _query_matrix(self, queries: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        rows, cols = [], []
        for i, q in enumerate(queries):
            for term in set(q):
                j = self.vocab.get(term)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(queries), len(self.vocab)))

    def scores(self, queries: Sequence[Sequence[str]]) -> Scores:
        q = self._query_matrix(queries)
        out = (q @ self._w.T).toarray()
        return np.asarray(out, dtype=np.float32)

    def term_contributions(self, query: Sequence[str], doc_index: int, top: int = 8) -> list[tuple[str, float]]:
        """Which query terms contributed most to a document's score (used as human-readable evidence)."""
        row = self._w.getrow(doc_index)
        weights = dict(zip(row.indices.tolist(), row.data.tolist(), strict=True))
        inv = {j: t for t, j in self.vocab.items()}
        contrib = [(inv[j], weights[j]) for t in set(query) if (j := self.vocab.get(t)) is not None and j in weights]
        return sorted(contrib, key=lambda x: -x[1])[:top]
