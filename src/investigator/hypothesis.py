"""Optional LLM step: root-cause hypotheses grounded in the retrieved cases.

The LLM is never trusted: its output must parse as the expected JSON, every cited case number must exist
in the prompt, hypotheses without valid evidence are dropped, and confidence must be one of three labels.
The retrieval, component prediction and evidence do not depend on this step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from investigator.investigate import Investigation
from investigator.llm import Completer, LLMError


class _Hypothesis(BaseModel):
    statement: str = Field(min_length=3)
    evidence_cases: list[int] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    how_to_check: str = ""


class _Payload(BaseModel):
    summary: str = ""
    hypotheses: list[_Hypothesis] = Field(default_factory=list)


@dataclass(frozen=True)
class Hypothesis:
    statement: str
    evidence_bug_ids: list[int]
    confidence: str
    how_to_check: str


@dataclass(frozen=True)
class HypothesisResult:
    summary: str
    hypotheses: list[Hypothesis]
    warnings: list[str]
    input_tokens: int = 0
    output_tokens: int = 0


class Hypothesiser:
    def __init__(self, llm: Completer, max_cases: int = 8) -> None:
        pkg = resources.files("investigator.prompts")
        self._llm = llm
        self._system = (pkg / "hypothesis_system.txt").read_text(encoding="utf-8").strip()
        self._user = (pkg / "hypothesis_user.txt").read_text(encoding="utf-8")
        self._max_cases = max_cases

    async def run(self, summary: str, description: str, inv: Investigation) -> HypothesisResult:
        cases = inv.similar_cases[: self._max_cases]
        if not cases:
            return HypothesisResult("No similar historical cases, so no grounded hypothesis is possible.", [], [])
        block = "\n".join(
            f"[{i}] #{c.bug_id} ({c.component}, {c.resolution}) {c.summary}"
            + (f" | fix note: {c.fix_note[:200]}" if c.fix_note else "")
            for i, c in enumerate(cases, 1)
        )
        comps = ", ".join(f"{n} ({p:.2f})" for n, p in inv.components[:3])
        user = self._user.format(summary=summary, description=description[:3000], components=comps, cases=block)
        try:
            res = await self._llm.complete(system=self._system, user=user)
        except LLMError as e:
            return HypothesisResult("", [], [f"LLM unavailable: {e}"])
        payload = self._parse(res.text)
        if payload is None:
            return HypothesisResult(
                "", [], ["model output was not valid JSON; no hypotheses returned"], res.input_tokens, res.output_tokens
            )
        warnings: list[str] = []
        out: list[Hypothesis] = []
        for h in payload.hypotheses[:3]:
            valid = sorted({n for n in h.evidence_cases if 1 <= n <= len(cases)})
            if not valid:
                warnings.append(f"dropped an ungrounded hypothesis: {h.statement[:60]!r}")
                continue
            out.append(Hypothesis(h.statement, [cases[n - 1].bug_id for n in valid], h.confidence, h.how_to_check))
        return HypothesisResult(payload.summary, out, warnings, res.input_tokens, res.output_tokens)

    @staticmethod
    def _parse(raw: str) -> _Payload | None:
        a, b = raw.find("{"), raw.rfind("}")
        if a == -1 or b <= a:
            return None
        try:
            return _Payload.model_validate(json.loads(raw[a : b + 1]))
        except (json.JSONDecodeError, ValidationError):
            return None
