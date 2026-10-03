"""HTTP API.

Note: no `from __future__ import annotations` here, because FastAPI must resolve annotations of
dependencies defined inside `create_app` at definition time.
"""

import hmac
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from investigator.hypothesis import Hypothesiser
from investigator.investigate import Investigator
from investigator.llm import Completer, LLMService
from investigator.settings import Settings

log = logging.getLogger(__name__)


class InvestigateBody(BaseModel):
    summary: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=20000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    hypotheses: bool = False


def create_app(
    settings: Settings | None = None,
    investigator_factory: Callable[[Settings], Investigator] | None = None,
    llm: Completer | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from investigator.build import load_investigator

        app.state.inv = (investigator_factory or load_investigator)(settings)
        http: httpx.AsyncClient | None = None
        completer = llm
        if completer is None and settings.llm_kind:
            http = httpx.AsyncClient()
            completer = LLMService(settings, http)
        app.state.hyp = Hypothesiser(completer) if completer else None
        try:
            yield
        finally:
            if http is not None:
                await http.aclose()

    app = FastAPI(title="trouble-report-investigator", version="0.1.0", lifespan=lifespan)

    def auth(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if settings.api_key and not (x_api_key and hmac.compare_digest(x_api_key.encode(), settings.api_key.encode())):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        inv: Investigator = app.state.inv
        return {"status": "ok", "indexed_reports": len(inv.bugs), "llm_configured": app.state.hyp is not None}

    @app.post("/investigate", dependencies=[Depends(auth)])
    async def investigate(body: InvestigateBody) -> dict[str, Any]:
        inv: Investigator = app.state.inv
        try:
            result = inv.investigate(body.summary, body.description, body.top_k)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        out = result.to_dict()
        if body.hypotheses:
            hyp: Hypothesiser | None = app.state.hyp
            if hyp is None:
                raise HTTPException(
                    status_code=503, detail="no LLM configured; set INVESTIGATOR_LLM_* or omit hypotheses"
                )
            h = await hyp.run(body.summary, body.description, result)
            out["hypotheses"] = {
                "summary": h.summary,
                "items": [asdict(x) for x in h.hypotheses],
                "warnings": h.warnings,
                "tokens": {"input": h.input_tokens, "output": h.output_tokens},
            }
        return out

    @app.get("/cases/{bug_id}", dependencies=[Depends(auth)])
    async def case(bug_id: int) -> dict[str, Any]:
        inv: Investigator = app.state.inv
        for b in inv.bugs:
            if b.id == bug_id:
                return {
                    "id": b.id,
                    "component": b.component,
                    "resolution": b.resolution,
                    "summary": b.summary,
                    "description": b.description,
                    "fix_note": b.fix_note,
                    "duplicate_of": b.dupe_of,
                    "keywords": list(b.keywords),
                }
        raise HTTPException(status_code=404, detail="case not found")

    return app
