"""Test doubles and synthetic corpora. Production code never imports from tests."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from investigator.data.dataset import Bug
from investigator.llm import LLMResult

DAY = 86400.0
T0 = 1_600_000_000.0  # arbitrary epoch origin for synthetic data

TOPICS: dict[str, list[str]] = {
    "Graphics": "gpu render compositor texture shader webrender frame paint layer".split(),
    "Networking": "socket http request tls dns proxy cookie connection channel".split(),
    "JavaScript Engine": "script jit bytecode garbage collector interpreter wasm heap realm".split(),
    "Audio/Video": "media decoder codec playback stream audio video buffer demuxer".split(),
}
COMMON = "crash fails when opening page after update expected result seen problem".split()


class HashEncoder:
    """Deterministic bag-of-words encoder (test double): cosine tracks lexical overlap."""

    dim = 96

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0  # noqa: S324
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return np.asarray(out / n, dtype=np.float32)


def make_bug(i: int, topic: str, rng: random.Random, created: float, **kw) -> Bug:  # type: ignore[no-untyped-def]
    words = rng.sample(TOPICS[topic], 5) + rng.sample(COMMON, 3)
    return Bug(
        id=kw.pop("id", 1000 + i),
        created=created,
        component=topic,
        resolution=kw.pop("resolution", "FIXED"),
        dupe_of=kw.pop("dupe_of", None),
        summary=" ".join(words[:4]),
        description=" ".join(words),
        fix_note=kw.pop("fix_note", ""),
        keywords=kw.pop("keywords", ()),
        **kw,
    )


def synthetic_bugs(n_per_topic: int = 40, seed: int = 3) -> list[Bug]:
    """Bugs spread over time; every 5th bug is a near-copy duplicate of an earlier bug of the same topic."""
    rng = random.Random(seed)
    bugs: list[Bug] = []
    topics = list(TOPICS)
    for i in range(n_per_topic * len(topics)):
        topic = topics[i % len(topics)]
        created = T0 + i * DAY
        earlier = [b for b in bugs if b.component == topic and b.resolution == "FIXED"]
        if i % 5 == 4 and earlier:
            master = rng.choice(earlier)
            words = master.description.split()
            rng.shuffle(words)
            bugs.append(
                Bug(
                    id=1000 + i,
                    created=created,
                    component=topic,
                    resolution="DUPLICATE",
                    dupe_of=master.id,
                    summary=" ".join(words[:4]),
                    description=" ".join(words[:7]),
                )
            )
        else:
            bugs.append(
                make_bug(i, topic, rng, created, fix_note=f"Pushed by dev: fix for {topic} {i}" if i % 3 == 0 else "")
            )
    return bugs


@pytest.fixture
def bugs() -> list[Bug]:
    return synthetic_bugs()


@pytest.fixture
def encoder() -> HashEncoder:
    return HashEncoder()


def ok_json(**kw) -> str:  # type: ignore[no-untyped-def]
    payload = {
        "summary": "s",
        "hypotheses": [
            {
                "statement": "memory corruption in shader cache",
                "evidence_cases": [1],
                "confidence": "medium",
                "how_to_check": "run with sanitiser",
            }
        ],
    }
    payload.update(kw)
    return json.dumps(payload)


class ScriptedLLM:
    def __init__(self, behaviour: Callable[[int], LLMResult | Exception] | None = None) -> None:
        self.prompts: list[str] = []
        self._behaviour = behaviour or (lambda n: LLMResult(ok_json(), 100, 20))

    async def complete(self, *, system: str, user: str) -> LLMResult:
        self.prompts.append(user)
        out = self._behaviour(len(self.prompts))
        if isinstance(out, Exception):
            raise out
        return out
