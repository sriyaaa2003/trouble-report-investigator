"""Bug dataset: loading, text scrubbing and duplicate groups."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# References to other bugs would let a model "cheat" on duplicate detection by reading the answer.
_BUGREF = re.compile(
    r"(?:https?://bugzilla\.mozilla\.org/show_bug\.cgi\?id=\d+|\bbug\s*#?\s*\d{5,8}\b|\bbugs?\s+\d{5,8}\b)", re.IGNORECASE
)


def scrub(text: str) -> str:
    return _BUGREF.sub("BUGREF", text)


# A comment only counts as "the fix" if it records a landed change and is not a back-out or a bot notice.
_LANDED = re.compile(r"Pushed by|hg\.mozilla\.org/(?:integration|mozilla-central)|mozilla-firefox/firefox/commit", re.I)
_NOT_A_FIX = re.compile(r"back(?:ed)?[ -]?out|bugbug|\bbot\b", re.I)


def usable_fix_note(note: str) -> str:
    """Return a cleaned one-line fix note, or '' if the comment is not evidence of a landed fix."""
    if not note or not _LANDED.search(note) or _NOT_A_FIX.search(note):
        return ""
    return re.sub(r"\s+", " ", note).strip()


@dataclass(frozen=True)
class Bug:
    id: int
    created: float  # epoch seconds
    component: str
    resolution: str
    dupe_of: int | None
    summary: str
    description: str
    fix_note: str = ""
    severity: str = ""
    keywords: tuple[str, ...] = ()
    crash_signature: str = ""
    product: str = "Core"
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def text(self, include_description: bool = True, max_chars: int = 4000) -> str:
        parts = [self.summary]
        if include_description and self.description:
            parts.append(self.description)
        if self.crash_signature:
            parts.append(self.crash_signature)
        return scrub("\n".join(parts))[:max_chars]


def _epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC).timestamp()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_bugs(directory: Path) -> list[Bug]:
    """Join metadata with fetched texts; keep Core bugs whose description was fetched."""
    meta: dict[int, dict[str, Any]] = {}
    for rec in _read_jsonl(directory / "meta.jsonl") + _read_jsonl(directory / "masters_meta.jsonl"):
        meta[int(rec["id"])] = rec
    bugs: list[Bug] = []
    for t in _read_jsonl(directory / "texts.jsonl"):
        if not t.get("ok"):
            continue
        m = meta.get(int(t["id"]))
        if m is None or m.get("product") != "Core":
            continue
        bugs.append(
            Bug(
                id=int(m["id"]),
                created=_epoch(m["creation_time"]),
                component=str(m["component"]),
                resolution=str(m.get("resolution", "")),
                dupe_of=int(m["dupe_of"]) if m.get("dupe_of") else None,
                summary=str(m.get("summary", "")),
                description=str(t.get("description", "")),
                fix_note=usable_fix_note(str(t.get("fix_note", "")))
                if str(m.get("resolution", "")) == "FIXED"
                else "",
                severity=str(m.get("severity", "")),
                keywords=tuple(m.get("keywords") or ()),
                crash_signature=str(m.get("cf_crash_signature") or ""),
            )
        )
    bugs.sort(key=lambda b: (b.created, b.id))
    return bugs


def duplicate_groups(bugs: Sequence[Bug]) -> dict[int, int]:
    """Union-find over `dupe_of` edges between bugs present in the dataset: bug id -> group root id."""
    ids = {b.id for b in bugs}
    parent: dict[int, int] = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in bugs:
        if b.dupe_of is not None and b.dupe_of in ids:
            ra, rb = find(b.id), find(b.dupe_of)
            if ra != rb:
                parent[ra] = rb
    return {i: find(i) for i in ids}
