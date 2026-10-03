from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest

from investigator.data.dataset import Bug, duplicate_groups, load_bugs, scrub
from investigator.data.fetch import (
    Client,
    clean_description,
    extract_fix_note,
    fetch_texts,
    month_windows,
    select_ids,
)
from investigator.evaluate.stats import bootstrap, paired_diff
from investigator.search.fusion import NEG, causal_mask, ranks, rrf, zfuse
from investigator.search.lexical import BM25
from investigator.search.tokenize import tokenize


# ---- tokenizer ---------------------------------------------------------------------------
def test_tokenizer_keeps_full_identifier_and_its_parts() -> None:
    toks = tokenize("Crash in mozilla::dom::ContentParent::RecvFoo during startup")
    assert "mozilla::dom::contentparent::recvfoo" in toks
    for part in ("mozilla", "dom", "contentparent", "content", "parent", "recvfoo", "recv", "foo", "crash", "startup"):
        assert part in toks


def test_tokenizer_drops_stopwords_numbers_and_short_tokens() -> None:
    toks = tokenize("The 12345 of a is x")
    assert toks == []


def test_tokenizer_handles_snake_and_camel_case() -> None:
    toks = tokenize("nsHttpChannel and get_user_agent")
    assert {"nshttpchannel", "get", "user", "agent"} <= set(toks)


# ---- scrubbing ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "see bug 1234567 for details",
        "Bug #1234567",
        "https://bugzilla.mozilla.org/show_bug.cgi?id=1234567",
        "bugs 1234567",
    ],
)
def test_scrub_removes_bug_references(text: str) -> None:
    out = scrub(text)
    assert "1234567" not in out and "BUGREF" in out


def test_scrub_keeps_ordinary_numbers() -> None:
    assert scrub("version 115 crashes in 2 threads") == "version 115 crashes in 2 threads"


# ---- BM25 --------------------------------------------------------------------------------
def test_bm25_ranks_rare_term_match_first_and_ignores_unknown_terms() -> None:
    docs = [["common", "word", "alpha"], ["common", "word", "rare"], ["common", "other"]]
    bm = BM25(docs)
    s = bm.scores([["rare", "common"], ["unknownterm"]])
    assert s.shape == (2, 3) and int(np.argmax(s[0])) == 1
    assert np.all(s[1] == 0)


def test_bm25_term_contributions_explain_a_match() -> None:
    bm = BM25([["alpha", "beta"], ["gamma", "delta"]])
    top = bm.term_contributions(["alpha", "gamma", "missing"], 0)
    assert [t for t, _ in top] == ["alpha"]


def test_bm25_length_normalisation_prefers_shorter_doc_for_same_tf() -> None:
    bm = BM25([["x", "y"], ["x"] + ["filler"] * 50])
    s = bm.scores([["x"]])[0]
    assert s[0] > s[1]


# ---- fusion ------------------------------------------------------------------------------
def test_causal_mask_hides_documents_created_at_or_after_the_query() -> None:
    scores = np.ones((2, 4), dtype=np.float32)
    corpus = np.array([1.0, 2.0, 3.0, 4.0])
    out = causal_mask(scores, corpus, np.array([3.0, 5.0]))
    assert list(out[0] > NEG / 2) == [True, True, False, False]
    assert list(out[1] > NEG / 2) == [True, True, True, True]


def test_ranks_and_rrf_prefer_documents_both_retrievers_like() -> None:
    a = np.array([[0.9, 0.8, 0.1, 0.05]], dtype=np.float32)  # ranks: d0=0 d1=1 d2=2 d3=3
    b = np.array([[0.1, 0.9, 0.8, 0.7]], dtype=np.float32)  # ranks: d1=0 d2=1 d3=2 d0=3
    assert list(ranks(a)[0]) == [0, 1, 2, 3]
    fused = rrf([a, b])
    assert int(np.argmax(fused[0])) == 1  # 2nd in one list and 1st in the other beats 1st and last


def test_fusion_keeps_masked_documents_masked() -> None:
    a = np.array([[0.9, NEG, 0.2]], dtype=np.float32)
    b = np.array([[0.1, NEG, 0.8]], dtype=np.float32)
    assert rrf([a, b])[0, 1] <= NEG / 2
    assert zfuse([a, b], [0.5, 0.5])[0, 1] <= NEG / 2


# ---- statistics --------------------------------------------------------------------------
def test_bootstrap_interval_brackets_the_mean_and_paired_diff_detects_a_gap() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.1, 400)
    est = bootstrap(a)
    assert est.low < est.mean < est.high
    d = paired_diff(a, a - 0.2)
    assert d.low > 0.15 and d.high < 0.25
    with pytest.raises(ValueError):
        paired_diff(a, a[:-1])


# ---- dataset -----------------------------------------------------------------------------
def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_bugs_joins_meta_and_texts_and_skips_unusable(tmp_path: Path) -> None:
    meta = [
        {
            "id": 1,
            "product": "Core",
            "component": "Graphics",
            "resolution": "FIXED",
            "dupe_of": None,
            "summary": "s1",
            "creation_time": "2023-01-02T00:00:00Z",
            "keywords": ["crash"],
        },
        {
            "id": 2,
            "product": "Core",
            "component": "Networking",
            "resolution": "DUPLICATE",
            "dupe_of": 1,
            "summary": "s2",
            "creation_time": "2023-02-02T00:00:00Z",
        },
        {
            "id": 3,
            "product": "Firefox",
            "component": "X",
            "resolution": "FIXED",
            "summary": "other product",
            "creation_time": "2023-03-02T00:00:00Z",
        },
    ]
    _write(tmp_path / "meta.jsonl", meta)
    _write(tmp_path / "masters_meta.jsonl", [])
    _write(
        tmp_path / "texts.jsonl",
        [
            {"id": 1, "ok": True, "description": "d1", "fix_note": "Pushed by x"},
            {"id": 2, "ok": True, "description": "d2"},
            {"id": 3, "ok": True, "description": "d3"},
            {"id": 4, "ok": False},
            {"id": 99, "ok": True, "description": "no meta"},
        ],
    )
    bugs = load_bugs(tmp_path)
    assert [b.id for b in bugs] == [1, 2]
    assert bugs[0].keywords == ("crash",) and bugs[0].fix_note == "Pushed by x" and bugs[1].dupe_of == 1
    assert bugs[0].created < bugs[1].created


def test_bug_text_scrubs_references_and_truncates() -> None:
    b = Bug(1, 0.0, "C", "FIXED", None, "title", "see bug 1234567 " + "x" * 5000)
    assert "1234567" not in b.text() and len(b.text()) <= 4000
    assert b.text(include_description=False) == "title"


def test_duplicate_groups_follow_chains() -> None:
    mk = lambda i, d: Bug(i, float(i), "C", "DUPLICATE" if d else "FIXED", d, "s", "d")  # noqa: E731
    bugs = [mk(1, None), mk(2, 1), mk(3, 2), mk(4, None), mk(5, 99)]  # 99 is outside the data set
    g = duplicate_groups(bugs)
    assert g[1] == g[2] == g[3] and g[4] != g[1] and g[5] == 5


# ---- fetching helpers --------------------------------------------------------------------
def test_month_windows_cover_years_without_gaps() -> None:
    w = month_windows(2023, 2024)
    assert len(w) == 24 and w[0] == ("2023-01-01", "2023-02-01") and w[11] == ("2023-12-01", "2024-01-01")
    assert all(w[i][1] == w[i + 1][0] for i in range(23))


def test_clean_description_and_fix_note() -> None:
    raw = "Created attachment 1\nUser Agent: x\nBuild ID: 1\nSteps to reproduce:\n\n\n\n1. open"
    assert clean_description(raw) == "Steps to reproduce:\n\n1. open"
    comments = [
        {"text": "desc"},
        {"text": "needs info"},
        {"text": "Pushed by dev@x https://hg.mozilla.org/integration/autoland/rev/abc"},
    ]
    assert extract_fix_note(comments).startswith("Pushed by dev@x")
    assert extract_fix_note(comments[:2]) == ""


def test_select_ids_is_seeded_and_returns_masters() -> None:
    meta = [{"id": i, "resolution": "DUPLICATE", "dupe_of": 1000 + i} for i in range(50)] + [
        {"id": 100 + i, "resolution": "FIXED"} for i in range(50)
    ]
    a = select_ids(meta, 10, 20, seed=1)
    b = select_ids(meta, 10, 20, seed=1)
    c = select_ids(meta, 10, 20, seed=2)
    assert a == b and a != c
    ids, masters = a
    assert len(ids) == 30 and len(masters) == 10 and all(m >= 1000 for m in masters)


async def test_client_retries_transient_errors_and_skips_restricted_bugs() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if "/bug/1/" in str(req.url):
            return httpx.Response(401)
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            json={"bugs": {"2": {"comments": [{"text": "Created attachment\nreal text"}, {"text": "Pushed by bot"}]}}},
        )

    client = Client(2, transport=httpx.MockTransport(handler), backoff_s=0.0)
    try:
        assert await client.get("/bug/1/comment") is None  # restricted
        r = await client.get("/bug/2/comment")
        assert r is not None and r.status_code == 200
    finally:
        await client.aclose()


async def test_fetch_texts_is_resumable_and_records_failures(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        bug_id = int(str(req.url).split("/bug/")[1].split("/")[0])
        if bug_id == 3:
            return httpx.Response(401)
        return httpx.Response(
            200, json={"bugs": {str(bug_id): {"comments": [{"text": f"desc {bug_id}"}, {"text": "Pushed by bot"}]}}}
        )

    out = tmp_path / "texts.jsonl"
    client = Client(2, transport=httpx.MockTransport(handler), backoff_s=0.0)
    try:
        await fetch_texts(client, [1, 2, 3], out)
        first = out.read_text(encoding="utf-8").splitlines()
        await fetch_texts(client, [1, 2, 3, 4], out)  # resume: only id 4 is new
    finally:
        await client.aclose()
    recs = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(first) == 3 and len(recs) == 4
    by_id = {r["id"]: r for r in recs}
    assert by_id[1]["ok"] and by_id[1]["fix_note"].startswith("Pushed by") and not by_id[3]["ok"]


def test_datetime_import_used() -> None:
    assert datetime(2024, 1, 1, tzinfo=UTC).timestamp() > 0


def test_embedding_cache_is_incremental_and_deduplicates(tmp_path: Path) -> None:
    from investigator.search.dense import encode_cached
    from tests.conftest import HashEncoder

    class Counting(HashEncoder):
        def __init__(self) -> None:
            self.encoded: list[str] = []

        def encode(self, texts):  # type: ignore[no-untyped-def]
            self.encoded += list(texts)
            return super().encode(texts)

    enc = Counting()
    a = encode_cached(enc, "m", ["alpha beta", "gamma delta", "alpha beta"], tmp_path)
    assert a.shape[0] == 3 and enc.encoded == ["alpha beta", "gamma delta"]  # duplicate text encoded once
    b = encode_cached(enc, "m", ["gamma delta", "epsilon zeta", "alpha beta"], tmp_path)
    assert enc.encoded[2:] == ["epsilon zeta"]  # only the new text was embedded
    assert np.allclose(b[0], a[1]) and np.allclose(b[2], a[0])
    assert encode_cached(enc, "other-model", ["alpha beta"], tmp_path).shape[0] == 1 and len(enc.encoded) == 4


def test_embedding_cache_resumes_after_an_interruption(tmp_path: Path) -> None:
    from investigator.search.dense import encode_cached
    from tests.conftest import HashEncoder

    class Flaky(HashEncoder):
        def __init__(self) -> None:
            self.calls = 0
            self.encoded: list[str] = []

        def encode(self, texts):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("machine went to sleep")
            self.encoded += list(texts)
            return super().encode(texts)

    texts = [f"word{i} common" for i in range(10)]
    enc = Flaky()
    with pytest.raises(RuntimeError):
        encode_cached(enc, "m", texts, tmp_path, chunk_size=3)
    assert len(enc.encoded) == 6  # two chunks were completed and saved before the crash
    enc.encoded.clear()
    out = encode_cached(enc, "m", texts, tmp_path, chunk_size=3)
    assert out.shape[0] == 10 and len(enc.encoded) == 4  # only the unfinished texts are embedded


@pytest.mark.parametrize(
    "note,expected",
    [
        (
            "Pushed by dev@x.org:\nhttps://hg.mozilla.org/integration/autoland/rev/abc  Fix the thing",
            "Pushed by dev@x.org: https://hg.mozilla.org/integration/autoland/rev/abc Fix the thing",
        ),
        ("Backed out changeset abc for causing bustage. Pushed by sheriff@x.org", ""),
        ("The [Bugbug](https://github.com/mozilla/bugbug/) bot thinks this bug should belong to 'X'", ""),
        (
            "Pushed by dev@x.org: fixes both the crash and the hang https://hg.mozilla.org/integration/autoland/rev/abc",
            "Pushed by dev@x.org: fixes both the crash and the hang https://hg.mozilla.org/integration/autoland/rev/abc",
        ),
        ("needs more info from the reporter", ""),
        ("", ""),
    ],
)
def test_usable_fix_note_only_accepts_landed_changes(note: str, expected: str) -> None:
    from investigator.data.dataset import usable_fix_note

    assert usable_fix_note(note) == expected


def test_fix_notes_are_dropped_for_reports_that_were_not_fixed(tmp_path: Path) -> None:
    base = {"product": "Core", "component": "A", "summary": "s"}
    meta = [
        {**base, "id": 1, "resolution": "FIXED", "creation_time": "2023-01-02T00:00:00Z"},
        {**base, "id": 2, "resolution": "WORKSFORME", "creation_time": "2023-01-03T00:00:00Z"},
    ]
    note = "Pushed by dev@x.org: https://hg.mozilla.org/integration/autoland/rev/abc"
    _write(tmp_path / "meta.jsonl", meta)
    rows = [
        {"id": 1, "ok": True, "description": "d", "fix_note": note},
        {"id": 2, "ok": True, "description": "d", "fix_note": note},
    ]
    _write(tmp_path / "texts.jsonl", rows)
    by_id = {b.id: b for b in load_bugs(tmp_path)}
    assert by_id[1].fix_note.startswith("Pushed by") and by_id[2].fix_note == ""
