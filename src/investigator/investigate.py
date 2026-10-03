"""The investigation engine: new report in, structured evidence-backed analysis out.

Everything here is deterministic and traceable to concrete historical reports. An optional LLM step
(`hypothesis.py`) can be layered on top, but nothing below depends on it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from investigator.data.dataset import Bug, scrub
from investigator.models import ComponentModel
from investigator.search.dense import TextEncoder
from investigator.search.fusion import NEG, zfuse
from investigator.search.lexical import BM25
from investigator.search.tokenize import tokenize

_SYMBOL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+\b|\b[a-z]+[A-Z][A-Za-z0-9]+\b")


@dataclass
class SimilarCase:
    bug_id: int
    rank: int
    similarity_dense: float
    fused_score: float
    component: str
    resolution: str
    summary: str
    duplicate_of: int | None
    fix_note: str
    keywords: list[str]
    matched_terms: list[str]
    shared_symbols: list[str]


@dataclass
class Investigation:
    components: list[tuple[str, float]]
    similar_cases: list[SimilarCase]
    evidence: dict[str, Any]
    steps: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def symbols(text: str) -> set[str]:
    """Code-like identifiers (namespaced or camelCase): the strongest hint of where in the code a fault is."""
    return {m.group(0) for m in _SYMBOL.finditer(text)}


class Investigator:
    def __init__(
        self,
        bugs: Sequence[Bug],
        embeddings: NDArray[np.float32],
        model: ComponentModel,
        encoder: TextEncoder,
        weights: dict[str, float] | None = None,
        dense_weight: float = 0.5,
        top_k: int = 10,
    ) -> None:
        if len(bugs) != embeddings.shape[0]:
            raise ValueError("embeddings must align with bugs")
        self.bugs = list(bugs)
        self.emb = embeddings
        self.model = model
        self.encoder = encoder
        self.weights = weights or {"tfidf_lr": 1.0, "dense_lr": 1.0, "knn": 1.0}
        self.dense_weight = dense_weight
        self.top_k = top_k
        self._tokens = [tokenize(b.text()) for b in self.bugs]
        self._bm25 = BM25(self._tokens)
        self._symbols = [symbols(b.text()) for b in self.bugs]

    def investigate(
        self, summary: str, description: str = "", top_k: int | None = None, as_of: float | None = None
    ) -> Investigation:
        k = top_k or self.top_k
        text = scrub(f"{summary}\n{description}").strip()
        if not text:
            raise ValueError("a report needs a summary or a description")
        qvec = self.encoder.encode([text])
        qtok = tokenize(text)
        sd = (qvec @ self.emb.T).astype(np.float32)
        sb = self._bm25.scores([qtok])
        if as_of is not None:  # reproduce what the engineer could have seen at that time
            created = np.asarray([b.created for b in self.bugs])
            sd[:, created >= as_of] = NEG
            sb[:, created >= as_of] = NEG
        fused = zfuse([sd, sb], [self.dense_weight, 1 - self.dense_weight])
        if as_of is not None:
            fused[:, sd[0] <= NEG / 2] = NEG
        order = np.argsort(-fused[0])[:k]
        order = np.asarray([i for i in order if fused[0, i] > NEG / 2], dtype=np.int64)

        # --- subsystem prediction (ensemble) ---
        members = self.model.members([text], qvec)
        proba = ComponentModel.combine(members, self.weights)[0]
        top_c = np.argsort(-proba)[:5]
        components = [(self.model.classes[int(i)], float(proba[i])) for i in top_c]

        # --- similar cases with evidence ---
        qsyms = symbols(text)
        cases: list[SimilarCase] = []
        for r, idx in enumerate(order, start=1):
            b = self.bugs[int(idx)]
            terms = [t for t, _ in self._bm25.term_contributions(qtok, int(idx), top=6)]
            cases.append(
                SimilarCase(
                    bug_id=b.id,
                    rank=r,
                    similarity_dense=float(sd[0, idx]),
                    fused_score=float(fused[0, idx]),
                    component=b.component,
                    resolution=b.resolution,
                    summary=b.summary,
                    duplicate_of=b.dupe_of,
                    fix_note=b.fix_note,
                    keywords=list(b.keywords),
                    matched_terms=terms,
                    shared_symbols=sorted(qsyms & self._symbols[int(idx)])[:6],
                )
            )
        evidence = self._evidence(cases)
        return Investigation(
            components, cases, evidence, self._steps(components, cases, evidence), warnings=self._warnings(cases)
        )

    # ---- evidence and recommendations ---------------------------------------------------------
    @staticmethod
    def _evidence(cases: Sequence[SimilarCase]) -> dict[str, Any]:
        comp = Counter(c.component for c in cases)
        res = Counter(c.resolution for c in cases)
        kw = Counter(k for c in cases for k in c.keywords)
        syms = Counter(s for c in cases for s in c.shared_symbols)
        return {
            "neighbour_components": comp.most_common(),
            "neighbour_resolutions": res.most_common(),
            "recurring_keywords": [(k, n) for k, n in kw.most_common(6) if n >= 2],
            "shared_symbols": syms.most_common(6),
            "n_neighbours": len(cases),
        }

    @staticmethod
    def _steps(
        components: Sequence[tuple[str, float]], cases: Sequence[SimilarCase], evidence: dict[str, Any]
    ) -> list[str]:
        steps: list[str] = []
        if not cases:
            return [
                "No similar historical report was found. Treat this as a new class of problem and collect more data."
            ]
        top = cases[0]
        steps.append(
            f"Read #{top.bug_id} first (most similar, dense cosine {top.similarity_dense:.2f}): "
            f"'{top.summary[:90]}' [{top.component}, {top.resolution}]."
            + (f" It was closed as a duplicate of #{top.duplicate_of}." if top.duplicate_of else "")
        )
        name, p = components[0]
        votes = dict(evidence["neighbour_components"]).get(name, 0)
        steps.append(
            f"Start triage in '{name}' (model probability {p:.2f}; "
            f"{votes} of {len(cases)} similar reports are filed there)."
            + (f" Second candidate: '{components[1][0]}' ({components[1][1]:.2f})." if len(components) > 1 else "")
        )
        fixes = [c for c in cases if c.fix_note][:2]
        for c in fixes:
            steps.append(f"Inspect the change that resolved similar report #{c.bug_id}: {c.fix_note[:140].strip()!r}")
        if evidence["shared_symbols"]:
            syms = ", ".join(s for s, _ in evidence["shared_symbols"][:4])
            steps.append(f"Look at code involving: {syms} (these appear in both this report and similar ones).")
        if evidence["recurring_keywords"]:
            kws = ", ".join(f"{k} ({n})" for k, n in evidence["recurring_keywords"][:4])
            steps.append(f"Recurring tags among similar reports: {kws}; check whether the same procedure applies.")
        return steps

    @staticmethod
    def _warnings(cases: Sequence[SimilarCase]) -> list[str]:
        out = []
        if cases and cases[0].similarity_dense < 0.6:
            out.append(
                "Even the closest historical report is only loosely similar; treat component and steps as weak hints."
            )
        return out
