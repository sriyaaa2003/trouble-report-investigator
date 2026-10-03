"""Download public Mozilla Bugzilla (product Core) bug reports through the official REST API.

Polite by design: identifying User-Agent, bounded concurrency, Retry-After honoured, everything
cached on disk so a run can be stopped and resumed. Restricted (security) bugs answer 401/403 and are
skipped. The data is NOT redistributed with this repository; each user fetches it themselves.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

BASE = "https://bugzilla.mozilla.org/rest"
USER_AGENT = "trouble-report-investigator-research/0.1 (portfolio project; polite client)"
META_FIELDS = "id,summary,component,product,resolution,dupe_of,creation_time,severity,keywords,status,cf_crash_signature"
_FIX_MARKERS = ("Pushed by", "hg.mozilla.org/integration", "hg.mozilla.org/mozilla-central", "mozilla-firefox/firefox/commit")


def month_windows(start_year: int, end_year: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            a = date(y, m, 1)
            b = date(y + (m == 12), (m % 12) + 1, 1)
            out.append((a.isoformat(), b.isoformat()))
    return out


class Client:
    def __init__(self, concurrency: int = 6, transport: httpx.AsyncBaseTransport | None = None, backoff_s: float = 1.5) -> None:
        self._http = httpx.AsyncClient(timeout=60, headers={"User-Agent": USER_AGENT}, transport=transport)
        self._backoff_s = backoff_s
        self._sem = asyncio.Semaphore(concurrency)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response | None:
        """GET with retry/backoff. Returns None for 401/403/404 (restricted or missing bugs)."""
        for attempt in range(6):
            async with self._sem:
                try:
                    r = await self._http.get(f"{BASE}{path}", params=params)
                except (httpx.TransportError, httpx.TimeoutException):
                    r = None
            if r is not None:
                if r.status_code == 200:
                    return r
                if r.status_code in (401, 403, 404):
                    return None
                if r.status_code not in (429, 500, 502, 503, 504):
                    log.warning("unexpected status %s for %s", r.status_code, path)
                    return None
                wait = float(r.headers.get("retry-after", 0)) or 0.0
            else:
                wait = 0.0
            await asyncio.sleep(max(wait, self._backoff_s * (2**attempt)) + (random.random() if self._backoff_s else 0.0))  # noqa: S311
        log.warning("giving up on %s", path)
        return None


async def fetch_meta(client: Client, start_year: int, end_year: int, out: Path) -> list[dict[str, Any]]:
    """Metadata for resolved (FIXED or DUPLICATE) Core bugs created in the given years."""
    if out.exists():
        return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]

    async def window(resolution: str, a: str, b: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            params = {
                "product": "Core", "resolution": resolution, "limit": 500, "offset": offset,
                "include_fields": META_FIELDS,
                "f1": "creation_ts", "o1": "greaterthaneq", "v1": a,
                "f2": "creation_ts", "o2": "lessthan", "v2": b,
            }
            r = await client.get("/bug", params)
            if r is None:
                break
            bugs = r.json().get("bugs", [])
            rows += bugs
            if len(bugs) < 500:
                break
            offset += 500
        return rows

    tasks = [window(res, a, b) for (a, b) in month_windows(start_year, end_year) for res in ("FIXED", "DUPLICATE")]
    results = await asyncio.gather(*tasks)
    seen: dict[int, dict[str, Any]] = {}
    for rows in results:
        for row in rows:
            seen[int(row["id"])] = row
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in seen.values()), encoding="utf-8")
    log.info("metadata: %d bugs", len(seen))
    return list(seen.values())


async def fetch_meta_by_ids(client: Client, ids: Iterable[int]) -> list[dict[str, Any]]:
    ids = sorted(set(ids))
    out: list[dict[str, Any]] = []

    async def batch(chunk: list[int]) -> None:
        r = await client.get("/bug", {"id": ",".join(map(str, chunk)), "include_fields": META_FIELDS})
        if r is not None:
            out.extend(r.json().get("bugs", []))

    await asyncio.gather(*[batch(ids[i : i + 100]) for i in range(0, len(ids), 100)])
    return out


def clean_description(text: str, limit: int = 6000) -> str:
    """Drop boilerplate that carries no diagnostic signal (attachment notices, user-agent lines)."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Created attachment") or s.startswith("User Agent:") or s.startswith("Build ID:"):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()[:limit]


def extract_fix_note(comments: list[dict[str, Any]], limit: int = 400) -> str:
    """First later comment that records a landed change (changeset URL or the 'Pushed by' bot note)."""
    for c in comments[1:]:
        text = str(c.get("text", ""))
        if any(m in text for m in _FIX_MARKERS):
            return text.strip()[:limit]
    return ""


async def fetch_texts(client: Client, ids: list[int], out: Path, progress_every: int = 500) -> None:
    """Fetch description + fix note for each id, appending to a JSONL cache (resumable)."""
    done: set[int] = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    todo = [i for i in ids if i not in done]
    log.info("texts: %d cached, %d to fetch", len(done), len(todo))
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    counter = {"n": 0}

    with out.open("a", encoding="utf-8") as f:

        async def one(bug_id: int) -> None:
            r = await client.get(f"/bug/{bug_id}/comment")
            rec: dict[str, Any] = {"id": bug_id, "ok": False}
            if r is not None:
                comments = r.json().get("bugs", {}).get(str(bug_id), {}).get("comments", [])
                if comments:
                    rec = {
                        "id": bug_id, "ok": True,
                        "description": clean_description(str(comments[0].get("text", ""))),
                        "fix_note": extract_fix_note(comments),
                        "n_comments": len(comments),
                    }
            async with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                counter["n"] += 1
                if counter["n"] % progress_every == 0:
                    log.info("texts fetched: %d / %d", counter["n"], len(todo))

        await asyncio.gather(*[one(i) for i in todo])


def select_ids(
    meta: list[dict[str, Any]], max_dups: int, max_fixed: int, seed: int
) -> tuple[list[int], list[int]]:
    """Seeded sample: duplicates (with their master ids) and fixed bugs."""
    rng = random.Random(seed)  # noqa: S311
    dups = [m for m in meta if m.get("resolution") == "DUPLICATE" and m.get("dupe_of")]
    fixed = [m for m in meta if m.get("resolution") == "FIXED"]
    rng.shuffle(dups)
    rng.shuffle(fixed)
    dups, fixed = dups[:max_dups], fixed[:max_fixed]
    return [int(m["id"]) for m in dups] + [int(m["id"]) for m in fixed], [int(m["dupe_of"]) for m in dups]


async def run_fetch(
    out_dir: Path, start_year: int, end_year: int, max_dups: int, max_fixed: int, seed: int, concurrency: int
) -> None:
    client = Client(concurrency)
    try:
        meta = await fetch_meta(client, start_year, end_year, out_dir / "meta.jsonl")
        ids, master_ids = select_ids(meta, max_dups, max_fixed, seed)
        have = {int(m["id"]) for m in meta}
        masters_meta_path = out_dir / "masters_meta.jsonl"
        if masters_meta_path.exists():
            masters_meta = [json.loads(x) for x in masters_meta_path.read_text(encoding="utf-8").splitlines() if x]
        else:
            masters_meta = await fetch_meta_by_ids(client, [m for m in master_ids if m not in have])
            masters_meta_path.write_text("\n".join(json.dumps(m) for m in masters_meta), encoding="utf-8")
        all_ids = sorted(set(ids) | set(master_ids))
        log.info("selected %d bugs (%d dup masters outside the window)", len(all_ids), len(masters_meta))
        await fetch_texts(client, all_ids, out_dir / "texts.jsonl")
    finally:
        await client.aclose()
