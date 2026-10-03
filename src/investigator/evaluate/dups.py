"""Duplicate-report retrieval benchmark.

Question: given a new report that was later marked a duplicate, does the system find an earlier report
of the same issue? This is the "find similar historical Trouble Reports" capability, with a ground truth
produced by human triagers.

Protocol (no leakage):
  * Query = a bug resolved as DUPLICATE. Correct answers = earlier bugs in its duplicate group
    (the master it points at, the master's other duplicates, chained masters).
  * Corpus = every bug created BEFORE the query (a system never sees the future).
  * References to other bugs ("bug 1234567", Bugzilla URLs) are scrubbed from all text.
  * Hyper-parameters (fusion weight) are chosen on queries from the development years and every number
    is reported on the held-out test year.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from investigator.data.dataset import Bug, duplicate_groups
from investigator.evaluate.stats import Estimate, bootstrap, paired_diff
from investigator.search.fusion import NEG, causal_mask, rrf, zfuse
from investigator.search.lexical import BM25
from investigator.search.tokenize import tokenize

CUTOFFS = (1, 5, 10, 20)


@dataclass(frozen=True)
class DupQuery:
    index: int  # index into the bug list
    targets: tuple[int, ...]  # indices of correct (earlier) bugs


def build_queries(bugs: Sequence[Bug]) -> list[DupQuery]:
    group = duplicate_groups(bugs)
    members: dict[int, list[int]] = {}
    for i, b in enumerate(bugs):
        members.setdefault(group[b.id], []).append(i)
    out: list[DupQuery] = []
    for i, b in enumerate(bugs):
        if b.resolution != "DUPLICATE" or b.dupe_of is None:
            continue
        targets = tuple(j for j in members[group[b.id]] if j != i and bugs[j].created < b.created)
        if targets:
            out.append(DupQuery(i, targets))
    return out


def _first_correct_rank(scores: NDArray[np.float32], targets: Sequence[int]) -> int:
    best = float(np.max(scores[np.asarray(list(targets), dtype=np.int64)]))
    if best <= NEG / 2:  # target was masked out (should not happen: targets are earlier)
        return 10**9
    return int((scores > best).sum())


def _metrics(ranks: NDArray[np.int64]) -> dict[str, NDArray[np.float64]]:
    out = {f"recall@{k}": (ranks < k).astype(np.float64) for k in CUTOFFS}
    out["mrr"] = 1.0 / (ranks.astype(np.float64) + 1.0)
    return out


@dataclass
class DupResults:
    n_queries: int
    n_corpus: int
    per_system: dict[str, dict[str, NDArray[np.float64]]]
    chosen: dict[str, float]
    seconds: float

    def table(self, baseline: str = "bm25") -> str:
        keys = [f"recall@{k}" for k in CUTOFFS] + ["mrr"]
        lines = ["| System | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for name, m in self.per_system.items():
            cells = [bootstrap(m[k]).fmt() for k in keys]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", f"Paired difference vs `{baseline}` (95% bootstrap CI over queries):", ""]
        lines += ["| System | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        base = self.per_system[baseline]
        for name, m in self.per_system.items():
            if name == baseline:
                continue
            cells = [paired_diff(m[k], base[k]).fmt() for k in keys]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def estimates(self) -> dict[str, dict[str, Estimate]]:
        return {n: {k: bootstrap(v) for k, v in m.items()} for n, m in self.per_system.items()}


def evaluate(
    bugs: Sequence[Bug],
    dense: NDArray[np.float32],
    test_from: datetime,
    dev_from: datetime,
    weights: Sequence[float] = (0.3, 0.5, 0.7, 0.85),
    chunk: int = 400,
    include_description: bool = True,
) -> DupResults:
    """Run BM25, dense, RRF and tuned z-score fusion; select fusion weight on dev, report on test."""
    t0 = time.time()
    created = np.asarray([b.created for b in bugs], dtype=np.float64)
    docs = [tokenize(b.text(include_description)) for b in bugs]
    bm25 = BM25(docs)
    queries = build_queries(bugs)
    test_ts, dev_ts = test_from.replace(tzinfo=UTC).timestamp(), dev_from.replace(tzinfo=UTC).timestamp()
    dev_q = [q for q in queries if dev_ts <= created[q.index] < test_ts]
    test_q = [q for q in queries if created[q.index] >= test_ts]

    def run(qs: Sequence[DupQuery], ws: Sequence[float]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {"bm25": [], "dense": [], "rrf": [], **{f"zfuse_w{w}": [] for w in ws}}
        for s in range(0, len(qs), chunk):
            part = qs[s : s + chunk]
            idx = [q.index for q in part]
            qc = created[idx]
            sb = causal_mask(bm25.scores([docs[i] for i in idx]), created, qc)
            sd = causal_mask((dense[idx] @ dense.T).astype(np.float32), created, qc)
            # a query must never retrieve itself
            for r, i in enumerate(idx):
                sb[r, i] = NEG
                sd[r, i] = NEG
            mats = {"bm25": sb, "dense": sd, "rrf": rrf([sb, sd])}
            for w in ws:
                mats[f"zfuse_w{w}"] = zfuse([sd, sb], [w, 1 - w])
            for name, mat in mats.items():
                out[name] += [_first_correct_rank(mat[r], q.targets) for r, q in enumerate(part)]
        return out

    dev_ranks = run(dev_q, weights)
    best_w = max(weights, key=lambda w: float(np.mean(1.0 / (np.asarray(dev_ranks[f"zfuse_w{w}"]) + 1.0))))
    test_ranks = run(test_q, [best_w])
    systems = {
        "bm25": "bm25",
        "dense": "dense",
        "rrf": "hybrid (RRF)",
        f"zfuse_w{best_w}": f"hybrid (z-score, w_dense={best_w})",
    }
    per = {label: _metrics(np.asarray(test_ranks[key], dtype=np.int64)) for key, label in systems.items()}
    # Order: baseline first.
    ordered = {"bm25": per["bm25"], **{k: v for k, v in per.items() if k != "bm25"}}
    return DupResults(
        len(test_q), len(bugs), ordered, {"w_dense": best_w, "dev_queries": float(len(dev_q))}, time.time() - t0
    )
