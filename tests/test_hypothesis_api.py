from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from investigator.api import create_app
from investigator.data.dataset import Bug
from investigator.hypothesis import Hypothesiser
from investigator.investigate import Investigator
from investigator.llm import LLMError, LLMResult
from investigator.models import ComponentModel
from investigator.settings import Settings
from tests.conftest import HashEncoder, ScriptedLLM, ok_json


@pytest.fixture
def investigator(bugs: list[Bug], encoder: HashEncoder) -> Investigator:
    emb = encoder.encode([b.text() for b in bugs])
    model = ComponentModel().fit([b.text() for b in bugs], emb, [b.component for b in bugs])
    return Investigator(bugs, emb, model, encoder, top_k=5)


QUERY = ("gpu render compositor texture", "shader webrender frame paint")


# ---- hypotheses --------------------------------------------------------------------------
async def test_grounded_hypothesis_maps_case_numbers_to_bug_ids(investigator: Investigator) -> None:
    llm = ScriptedLLM(lambda n: LLMResult(ok_json(), 120, 40))
    inv = investigator.investigate(*QUERY)
    res = await Hypothesiser(llm).run(*QUERY, inv)
    assert len(res.hypotheses) == 1 and res.hypotheses[0].evidence_bug_ids == [inv.similar_cases[0].bug_id]
    assert res.hypotheses[0].confidence == "medium" and (res.input_tokens, res.output_tokens) == (120, 40)
    prompt = llm.prompts[0]
    assert "[1] #" in prompt and QUERY[0] in prompt


async def test_ungrounded_or_invalid_hypotheses_are_dropped(investigator: Investigator) -> None:
    payload = {
        "summary": "x",
        "hypotheses": [
            {"statement": "invented cause with no support", "evidence_cases": [], "confidence": "high"},
            {"statement": "cites a case that does not exist", "evidence_cases": [99], "confidence": "high"},
            {"statement": "valid and grounded", "evidence_cases": [2, 2, 99], "confidence": "low"},
        ],
    }
    llm = ScriptedLLM(lambda n: LLMResult(json.dumps(payload), 1, 1))
    inv = investigator.investigate(*QUERY)
    res = await Hypothesiser(llm).run(*QUERY, inv)
    assert [h.statement for h in res.hypotheses] == ["valid and grounded"]
    assert res.hypotheses[0].evidence_bug_ids == [inv.similar_cases[1].bug_id]
    assert len(res.warnings) == 2


@pytest.mark.parametrize(
    "raw",
    [
        "no json at all",
        "{broken",
        '{"hypotheses": [{"statement": "x", "confidence": "certain", "evidence_cases": [1]}]}',
    ],
)
async def test_invalid_model_output_yields_no_hypotheses(investigator: Investigator, raw: str) -> None:
    llm = ScriptedLLM(lambda n: LLMResult(raw, 1, 1))
    res = await Hypothesiser(llm).run(*QUERY, investigator.investigate(*QUERY))
    assert res.hypotheses == [] and res.warnings


async def test_llm_failure_degrades_gracefully(investigator: Investigator) -> None:
    llm = ScriptedLLM(lambda n: LLMError("down", retryable=True))
    res = await Hypothesiser(llm).run(*QUERY, investigator.investigate(*QUERY))
    assert res.hypotheses == [] and "unavailable" in res.warnings[0]


async def test_more_than_three_hypotheses_are_capped(investigator: Investigator) -> None:
    hyps = [{"statement": f"hypothesis {i}", "evidence_cases": [1], "confidence": "low"} for i in range(6)]
    llm = ScriptedLLM(lambda n: LLMResult(json.dumps({"summary": "", "hypotheses": hyps}), 1, 1))
    res = await Hypothesiser(llm).run(*QUERY, investigator.investigate(*QUERY))
    assert len(res.hypotheses) == 3


# ---- HTTP API ----------------------------------------------------------------------------
def _client(investigator: Investigator, llm: ScriptedLLM | None, key: str = "") -> TestClient:
    s = Settings(api_key=key, _env_file=None)  # type: ignore[call-arg]
    return TestClient(create_app(s, investigator_factory=lambda _: investigator, llm=llm))


def test_api_investigate_and_case_lookup(investigator: Investigator) -> None:
    with _client(investigator, None) as c:
        assert c.get("/health").json() == {
            "status": "ok",
            "indexed_reports": len(investigator.bugs),
            "llm_configured": False,
        }
        r = c.post("/investigate", json={"summary": QUERY[0], "description": QUERY[1], "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["components"][0][0] == "Graphics" and len(body["similar_cases"]) == 3 and body["steps"]
        bug_id = body["similar_cases"][0]["bug_id"]
        assert c.get(f"/cases/{bug_id}").json()["id"] == bug_id
        assert c.get("/cases/1").status_code == 404


def test_api_validation_and_hypothesis_requires_llm(investigator: Investigator) -> None:
    with _client(investigator, None) as c:
        assert c.post("/investigate", json={"summary": ""}).status_code == 422
        assert c.post("/investigate", json={"summary": "x", "top_k": 0}).status_code == 422
        assert c.post("/investigate", json={"summary": "   "}).status_code == 422
        r = c.post("/investigate", json={"summary": QUERY[0], "hypotheses": True})
        assert r.status_code == 503


def test_api_returns_hypotheses_when_llm_configured(investigator: Investigator) -> None:
    with _client(investigator, ScriptedLLM()) as c:
        r = c.post("/investigate", json={"summary": QUERY[0], "description": QUERY[1], "hypotheses": True})
        h = r.json()["hypotheses"]
        assert r.status_code == 200 and len(h["items"]) == 1 and h["items"][0]["evidence_bug_ids"]


def test_api_key_is_enforced_when_configured(investigator: Investigator) -> None:
    with _client(investigator, None, key="secret") as c:
        assert c.post("/investigate", json={"summary": "x"}).status_code == 401
        assert c.post("/investigate", json={"summary": "x"}, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/investigate", json={"summary": QUERY[0]}, headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/health").status_code == 200
