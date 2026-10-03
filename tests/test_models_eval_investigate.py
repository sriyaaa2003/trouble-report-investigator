from __future__ import annotations

import random
from datetime import UTC, datetime

import numpy as np
import pytest

from investigator.data.dataset import Bug
from investigator.evaluate import components as comp_eval
from investigator.evaluate import dups as dup_eval
from investigator.investigate import Investigator, symbols
from investigator.models import ComponentModel
from tests.conftest import DAY, T0, HashEncoder, make_bug, synthetic_bugs


def _embed(bugs: list[Bug], enc: HashEncoder) -> np.ndarray:
    return enc.encode([b.text() for b in bugs])


# ---- duplicate-retrieval protocol --------------------------------------------------------
def test_build_queries_only_uses_earlier_targets(bugs: list[Bug]) -> None:
    qs = dup_eval.build_queries(bugs)
    assert qs, "synthetic corpus should contain duplicates"
    for q in qs:
        assert bugs[q.index].resolution == "DUPLICATE"
        assert all(bugs[t].created < bugs[q.index].created for t in q.targets)
        assert q.index not in q.targets


def test_a_duplicate_of_a_later_bug_is_not_a_valid_query() -> None:
    rng = random.Random(0)
    early_dup = Bug(1, T0, "Graphics", "DUPLICATE", 2, "gpu render", "gpu render paint")  # points forward in time
    late_master = make_bug(2, "Graphics", rng, T0 + DAY, id=2)
    assert dup_eval.build_queries([early_dup, late_master]) == []


def test_dup_evaluation_finds_planted_duplicates_and_reports_all_systems(bugs: list[Bug], encoder: HashEncoder) -> None:
    emb = _embed(bugs, encoder)
    res = dup_eval.evaluate(
        bugs,
        emb,
        test_from=datetime.fromtimestamp(T0 + 120 * DAY, UTC),
        dev_from=datetime.fromtimestamp(T0 + 60 * DAY, UTC),
        weights=(0.3, 0.7),
        chunk=16,
    )
    assert res.n_queries > 5 and res.chosen["w_dense"] in (0.3, 0.7)
    assert set(res.per_system) >= {"bm25", "dense", "hybrid (RRF)"}
    # planted duplicates share most of their words with their master, so retrieval must beat chance by far
    chance = 10 / len(bugs)
    for name, m in res.per_system.items():
        assert m["recall@10"].mean() > 5 * chance, name
    assert "| bm25 |" in res.table() and "Paired difference" in res.table()


def test_no_peeking_at_the_future(encoder: HashEncoder) -> None:
    """Even a perfect textual match must not be returned if it was created after the query."""
    rng = random.Random(1)
    bugs = [make_bug(i, "Graphics", rng, T0 + i * DAY) for i in range(30)]
    master = bugs[5]
    # a duplicate created BEFORE its master (an adversarial ordering) has the master as a "future" target
    dup = Bug(900, T0 + 2 * DAY + 1, "Graphics", "DUPLICATE", master.id, master.summary, master.description)
    corpus = sorted([*bugs, dup], key=lambda b: (b.created, b.id))
    assert dup_eval.build_queries(corpus) == []


# ---- component model ---------------------------------------------------------------------
def test_component_model_learns_topic_vocabulary(encoder: HashEncoder) -> None:
    bugs = synthetic_bugs(60)
    train, test = bugs[:180], bugs[180:]
    model = ComponentModel().fit([b.text() for b in train], _embed(train, encoder), [b.component for b in train])
    members = model.members([b.text() for b in test], _embed(test, encoder))
    truth = np.array([model.classes.index(b.component) for b in test])
    for name, p in members.items():
        assert p.shape == (len(test), len(model.classes))
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), name
        assert (p.argmax(axis=1) == truth).mean() > 0.9, name
    combined = ComponentModel.combine(members, {"tfidf_lr": 1, "dense_lr": 1, "knn": 1})
    assert (combined.argmax(axis=1) == truth).mean() > 0.9


def test_component_evaluation_runs_with_time_split_and_reports_baselines(encoder: HashEncoder) -> None:
    bugs = synthetic_bugs(80)
    emb = _embed(bugs, encoder)
    res = comp_eval.evaluate(
        bugs,
        emb,
        dev_from=datetime.fromtimestamp(T0 + 200 * DAY, UTC),
        test_from=datetime.fromtimestamp(T0 + 260 * DAY, UTC),
        min_train=10,
        grid=(0.0, 1.0),
    )
    assert res.n_test > 20 and res.n_classes == 4 and 0.9 <= res.coverage <= 1.0
    assert set(res.per_system) == {"majority", "tfidf_lr", "dense_lr", "knn", "ensemble"}
    assert res.per_system["tfidf_lr"]["top1"].mean() > res.per_system["majority"]["top1"].mean() + 0.3
    assert all(m["top3"].mean() >= m["top1"].mean() for m in res.per_system.values())
    assert "Paired difference" in res.table()


# ---- investigator ------------------------------------------------------------------------
@pytest.fixture
def investigator(bugs: list[Bug], encoder: HashEncoder) -> Investigator:
    emb = _embed(bugs, encoder)
    model = ComponentModel().fit([b.text() for b in bugs], emb, [b.component for b in bugs])
    return Investigator(bugs, emb, model, encoder, top_k=5)


def test_investigation_returns_components_cases_evidence_and_steps(investigator: Investigator) -> None:
    r = investigator.investigate("gpu render compositor texture shader", "webrender frame paint layer crash")
    assert r.components[0][0] == "Graphics" and abs(sum(p for _, p in r.components) - 1.0) < 0.5
    assert len(r.similar_cases) == 5 and all(c.component for c in r.similar_cases)
    assert (
        r.similar_cases[0].rank == 1 and r.similar_cases[0].similarity_dense >= r.similar_cases[-1].similarity_dense - 1
    )
    assert r.evidence["n_neighbours"] == 5 and dict(r.evidence["neighbour_components"]).get("Graphics", 0) >= 3
    assert r.steps and f"#{r.similar_cases[0].bug_id}" in r.steps[0]
    assert any("Graphics" in s for s in r.steps)
    d = r.to_dict()
    assert {"components", "similar_cases", "evidence", "steps", "warnings"} <= set(d)


def test_investigation_rejects_empty_input(investigator: Investigator) -> None:
    with pytest.raises(ValueError):
        investigator.investigate("   ", "")


def test_as_of_hides_reports_created_after_that_time(investigator: Investigator, bugs: list[Bug]) -> None:
    cutoff = bugs[40].created
    r = investigator.investigate("gpu render compositor texture shader", "webrender", as_of=cutoff)
    by_id = {b.id: b for b in bugs}
    assert r.similar_cases and all(by_id[c.bug_id].created < cutoff for c in r.similar_cases)


def test_weak_matches_produce_a_warning(investigator: Investigator) -> None:
    r = investigator.investigate("zebra quantum xylophone", "entirely unrelated vocabulary here")
    assert r.warnings


def test_symbols_extracts_code_identifiers() -> None:
    s = symbols("crash in mozilla::dom::ContentParent::RecvFoo and nsHttpChannel but not plain words")
    assert "mozilla::dom::ContentParent::RecvFoo" in s and "nsHttpChannel" in s and "plain" not in s


def test_fix_notes_and_shared_symbols_become_investigation_steps(encoder: HashEncoder) -> None:
    rng = random.Random(4)
    bugs = [make_bug(i, "Graphics", rng, T0 + i * DAY) for i in range(12)]
    bugs[3] = Bug(
        bugs[3].id,
        bugs[3].created,
        "Graphics",
        "FIXED",
        None,
        "crash in gfx::WebRenderBridge::Flush",
        "gfx::WebRenderBridge::Flush null deref render",
        fix_note="Pushed by dev: guard null bridge",
    )
    emb = _embed(bugs, encoder)
    model = ComponentModel().fit([b.text() for b in bugs], emb, [b.component for b in bugs])
    inv = Investigator(bugs, emb, model, encoder, top_k=3)
    r = inv.investigate("crash in gfx::WebRenderBridge::Flush", "null deref render")
    assert r.similar_cases[0].bug_id == bugs[3].id
    assert r.similar_cases[0].shared_symbols == ["gfx::WebRenderBridge::Flush"]
    assert any("guard null bridge" in s for s in r.steps) and any("WebRenderBridge" in s for s in r.steps)


def test_investigator_rejects_misaligned_embeddings(bugs: list[Bug], encoder: HashEncoder) -> None:
    emb = _embed(bugs, encoder)
    model = ComponentModel().fit([b.text() for b in bugs], emb, [b.component for b in bugs])
    with pytest.raises(ValueError):
        Investigator(bugs, emb[:-1], model, encoder)


def test_component_model_handles_a_single_class_and_rejects_empty_training(encoder: HashEncoder) -> None:
    rng = random.Random(2)
    bugs = [make_bug(i, "Graphics", rng, T0 + i * DAY) for i in range(8)]
    m = ComponentModel().fit([b.text() for b in bugs], _embed(bugs, encoder), [b.component for b in bugs])
    p = m.members(["gpu render"], encoder.encode(["gpu render"]))
    assert m.classes == ["Graphics"] and all(v.shape == (1, 1) and v[0, 0] == pytest.approx(1.0) for v in p.values())
    with pytest.raises(ValueError):
        ComponentModel().fit([], np.zeros((0, 4), dtype=np.float32), [])
