"""Subsystem (component) predictors.

Three complementary members, combined by a weighted average of class probabilities:
  * TF-IDF + logistic regression: strong lexical baseline, learns which terms mark which subsystem.
  * Dense embedding + logistic regression: captures paraphrase and semantic similarity.
  * Retrieval vote: the components of the most similar historical reports (hybrid lexical+dense
    similarity), weighted by similarity. This is also what makes a prediction *explainable*: the evidence
    is a list of concrete past reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from investigator.search.fusion import NEG, zfuse
from investigator.search.lexical import BM25
from investigator.search.tokenize import tokenize

Probs = NDArray[np.float64]


def _align(proba: Probs, classes: NDArray[np.str_], all_classes: Sequence[str]) -> Probs:
    """Place a classifier's columns into the shared class order (missing classes get probability 0)."""
    out = np.zeros((proba.shape[0], len(all_classes)))
    pos = {c: i for i, c in enumerate(all_classes)}
    for j, c in enumerate(classes):
        out[:, pos[str(c)]] = proba[:, j]
    return out


@dataclass
class ComponentModel:
    """Fits on (texts, embeddings, labels); predicts class probabilities for new reports."""

    c_tfidf: float = 10.0
    c_dense: float = 10.0
    knn_k: int = 10
    knn_temperature: float = 0.5
    dense_weight_in_retrieval: float = 0.5
    classes: list[str] = field(default_factory=list)

    def fit(self, texts: Sequence[str], emb: NDArray[np.float32], labels: Sequence[str]) -> ComponentModel:
        if not labels:
            raise ValueError("cannot fit a component model without training reports")
        self.classes = sorted(set(labels))
        self._degenerate = len(self.classes) == 1  # a single class: nothing to discriminate, predict it
        self._labels = np.asarray([self.classes.index(label) for label in labels], dtype=np.int64)
        self._train_emb = emb
        self._train_tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25(self._train_tokens)
        self._tfidf = TfidfVectorizer(analyzer=tokenize, min_df=2, sublinear_tf=True, max_features=200_000)
        x = self._tfidf.fit_transform(texts)
        if not self._degenerate:
            self._lr_tfidf = LogisticRegression(C=self.c_tfidf, max_iter=400).fit(x, labels)
            self._lr_dense = LogisticRegression(C=self.c_dense, max_iter=400).fit(emb, labels)
        self._prior = np.bincount(self._labels, minlength=len(self.classes)) / len(self._labels)
        return self

    # ---- members --------------------------------------------------------------------------
    def proba_tfidf(self, texts: Sequence[str]) -> Probs:
        if self._degenerate:
            return np.ones((len(texts), 1))
        p = self._lr_tfidf.predict_proba(self._tfidf.transform(texts))
        return _align(p, self._lr_tfidf.classes_, self.classes)

    def proba_dense(self, emb: NDArray[np.float32]) -> Probs:
        if self._degenerate:
            return np.ones((emb.shape[0], 1))
        p = self._lr_dense.predict_proba(emb)
        return _align(p, self._lr_dense.classes_, self.classes)

    def similarity(self, texts: Sequence[str], emb: NDArray[np.float32]) -> NDArray[np.float32]:
        """Hybrid (z-scored dense + BM25) similarity of new reports to every training report."""
        sb = self._bm25.scores([tokenize(t) for t in texts])
        sd = (emb @ self._train_emb.T).astype(np.float32)
        w = self.dense_weight_in_retrieval
        sim: NDArray[np.float32] = zfuse([sd, sb], [w, 1 - w])
        return sim

    def proba_knn_from_similarity(self, sim: NDArray[np.float32]) -> Probs:
        k = min(self.knn_k, sim.shape[1])
        out = np.zeros((sim.shape[0], len(self.classes)))
        for i in range(sim.shape[0]):
            idx = np.argpartition(-sim[i], k - 1)[:k]
            idx = idx[sim[i, idx] > NEG / 2]
            if idx.size == 0:
                out[i] = self._prior
                continue
            z = sim[i, idx] / self.knn_temperature
            w = np.exp(z - z.max())
            for j, wj in zip(idx, w, strict=True):
                out[i, self._labels[j]] += wj
            out[i] /= out[i].sum()
        return out

    def proba_knn(self, texts: Sequence[str], emb: NDArray[np.float32]) -> Probs:
        return self.proba_knn_from_similarity(self.similarity(texts, emb))

    def members(self, texts: Sequence[str], emb: NDArray[np.float32]) -> dict[str, Probs]:
        return {
            "tfidf_lr": self.proba_tfidf(texts),
            "dense_lr": self.proba_dense(emb),
            "knn": self.proba_knn(texts, emb),
        }

    @staticmethod
    def combine(members: dict[str, Probs], weights: dict[str, float]) -> Probs:
        total = sum(weights.values())
        out = np.zeros_like(next(iter(members.values())), dtype=np.float64)
        for k, w in weights.items():
            out += w * members[k]
        result: Probs = out / total
        return result

    def prior_proba(self, n: int) -> Probs:
        return np.tile(self._prior, (n, 1))
