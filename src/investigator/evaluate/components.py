"""Subsystem-prediction benchmark with a strict time split.

Train on reports created up to the end of the development period, test on the held-out final year.
Ensemble weights are chosen on the development year only (model refit on earlier data), then the model is
refit on everything before the test year and evaluated once on the test year.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from investigator.data.dataset import Bug
from investigator.evaluate.stats import bootstrap, paired_diff
from investigator.models import ComponentModel, Probs


def _topk_hits(proba: Probs, y: NDArray[np.int64], k: int) -> NDArray[np.float64]:
    top = np.argsort(-proba, axis=1)[:, :k]
    hits: NDArray[np.float64] = (top == y[:, None]).any(axis=1).astype(np.float64)
    return hits


@dataclass
class ComponentResults:
    n_train: int
    n_test: int
    n_classes: int
    coverage: float  # share of test reports whose component is among the modelled classes
    per_system: dict[str, dict[str, NDArray[np.float64]]]
    macro_f1: dict[str, float]
    parent_top1: dict[str, float]  # right broad area (text before ':'), e.g. 'Graphics: WebRender' ~ 'Graphics'
    chosen_weights: dict[str, float]
    seconds: float

    def table(self, baseline: str = "tfidf_lr") -> str:
        lines = ["| System | top-1 accuracy | top-3 accuracy | parent-area top-1 | macro-F1 |", "|---|---|---|---|---|"]
        for name, m in self.per_system.items():
            t1, t3 = bootstrap(m["top1"]).fmt(), bootstrap(m["top3"]).fmt()
            lines.append(f"| {name} | {t1} | {t3} | {self.parent_top1[name]:.3f} | {self.macro_f1[name]:.3f} |")
        lines += [
            "",
            f"Paired difference vs `{baseline}` (95% bootstrap CI over test reports):",
            "",
            "| System | top-1 | top-3 |",
            "|---|---|---|",
        ]
        base = self.per_system[baseline]
        for name, m in self.per_system.items():
            if name != baseline:
                d1, d3 = paired_diff(m["top1"], base["top1"]), paired_diff(m["top3"], base["top3"])
                lines.append(f"| {name} | {d1.fmt()} | {d3.fmt()} |")
        return "\n".join(lines)


def _split(bugs: Sequence[Bug], lo: float | None, hi: float | None) -> list[int]:
    return [i for i, b in enumerate(bugs) if (lo is None or b.created >= lo) and (hi is None or b.created < hi)]


def evaluate(
    bugs: Sequence[Bug],
    emb: NDArray[np.float32],
    dev_from: datetime,
    test_from: datetime,
    min_train: int = 40,
    grid: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
) -> ComponentResults:
    t0 = time.time()
    dev_ts, test_ts = dev_from.replace(tzinfo=UTC).timestamp(), test_from.replace(tzinfo=UTC).timestamp()
    texts = [b.text() for b in bugs]
    labels = np.asarray([b.component for b in bugs])

    def fit(train_idx: list[int]) -> ComponentModel:
        counts: dict[str, int] = {}
        for i in train_idx:
            counts[labels[i]] = counts.get(labels[i], 0) + 1
        keep = [i for i in train_idx if counts[labels[i]] >= min_train]
        return ComponentModel().fit([texts[i] for i in keep], emb[keep], [str(labels[i]) for i in keep])

    def evaluate_members(
        model: ComponentModel, idx: list[int]
    ) -> tuple[dict[str, Probs], NDArray[np.int64], list[int]]:
        known = set(model.classes)
        in_cls = [i for i in idx if labels[i] in known]
        y = np.asarray([model.classes.index(labels[i]) for i in in_cls], dtype=np.int64)
        return model.members([texts[i] for i in in_cls], emb[in_cls]), y, in_cls

    # --- choose ensemble weights on the dev year (train strictly before it) ---
    dev_model = fit(_split(bugs, None, dev_ts))
    dev_members, dev_y, _ = evaluate_members(dev_model, _split(bugs, dev_ts, test_ts))
    best, best_acc = {"tfidf_lr": 1.0, "dense_lr": 0.0, "knn": 0.0}, -1.0
    for wt, wd, wk in itertools.product(grid, grid, grid):
        if wt + wd + wk == 0:
            continue
        p = ComponentModel.combine(dev_members, {"tfidf_lr": wt, "dense_lr": wd, "knn": wk})
        acc = float(_topk_hits(p, dev_y, 1).mean())
        if acc > best_acc:
            best, best_acc = {"tfidf_lr": wt, "dense_lr": wd, "knn": wk}, acc

    # --- refit on everything before the test year; evaluate once on the test year ---
    train_idx = _split(bugs, None, test_ts)
    model = fit(train_idx)
    test_idx_all = _split(bugs, test_ts, None)
    members, y, kept = evaluate_members(model, test_idx_all)
    members["ensemble"] = ComponentModel.combine(members, best)
    members["majority"] = model.prior_proba(len(y))

    per: dict[str, dict[str, NDArray[np.float64]]] = {}
    f1: dict[str, float] = {}
    parent: dict[str, float] = {}
    parent_of = [c.split(":")[0] for c in model.classes]
    for name in ["majority", "tfidf_lr", "dense_lr", "knn", "ensemble"]:
        p = members[name]
        per[name] = {"top1": _topk_hits(p, y, 1), "top3": _topk_hits(p, y, 3)}
        pred = p.argmax(axis=1)
        parent[name] = float(np.mean([parent_of[a] == parent_of[b] for a, b in zip(pred, y, strict=True)]))
        f1[name] = float(
            f1_score(y, p.argmax(axis=1), average="macro", labels=list(range(len(model.classes))), zero_division=0)
        )
    return ComponentResults(
        len(train_idx),
        len(kept),
        len(model.classes),
        len(kept) / max(1, len(test_idx_all)),
        per,
        f1,
        parent,
        best,
        time.time() - t0,
    )
