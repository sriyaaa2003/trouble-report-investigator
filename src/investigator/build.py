"""Build and persist the artifacts the investigator needs (embeddings + fitted component model)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from investigator.data.dataset import Bug, load_bugs
from investigator.investigate import Investigator
from investigator.models import ComponentModel
from investigator.search.dense import FastEmbedEncoder, TextEncoder, encode_cached
from investigator.settings import Settings

# bge-small truncates at 512 tokens anyway; capping the text keeps embedding time bounded.
EMBED_CHARS = 1500
DEFAULT_WEIGHTS = {"tfidf_lr": 1.0, "dense_lr": 1.0, "knn": 1.0}


def embed_bugs(bugs: list[Bug], encoder: TextEncoder, model_name: str, cache_dir: Path) -> np.ndarray:
    texts = [b.text(max_chars=EMBED_CHARS) for b in bugs]
    return encode_cached(encoder, model_name, texts, cache_dir)


def build_artifacts(settings: Settings, weights_path: Path | None = None, min_train: int = 40) -> Path:
    bugs = load_bugs(settings.data_dir)
    if not bugs:
        raise SystemExit(f"no bugs found in {settings.data_dir}; run `investigator fetch` first")
    encoder = FastEmbedEncoder(
        settings.embedding_model, str(settings.embedding_cache_dir) if settings.embedding_cache_dir else None
    )
    emb = embed_bugs(bugs, encoder, settings.embedding_model, settings.artifacts_dir)

    counts: dict[str, int] = {}
    for b in bugs:
        counts[b.component] = counts.get(b.component, 0) + 1
    keep = [i for i, b in enumerate(bugs) if counts[b.component] >= min_train]
    model = ComponentModel().fit([bugs[i].text() for i in keep], emb[keep], [bugs[i].component for i in keep])

    weights = dict(DEFAULT_WEIGHTS)
    if weights_path and weights_path.exists():
        weights = json.loads(weights_path.read_text(encoding="utf-8")).get("chosen_weights", weights)

    out = settings.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    (out / "bug_ids.json").write_text(json.dumps([b.id for b in bugs]), encoding="utf-8")
    joblib.dump({"model": model, "weights": weights}, out / "component_model.joblib")
    return out


def load_investigator(settings: Settings) -> Investigator:
    out = settings.artifacts_dir
    bugs = load_bugs(settings.data_dir)
    ids = json.loads((out / "bug_ids.json").read_text(encoding="utf-8"))
    if [b.id for b in bugs] != ids:
        raise RuntimeError("artifacts do not match the data directory; run `investigator build` again")
    emb = np.load(out / "embeddings.npy")
    saved = joblib.load(out / "component_model.joblib")
    encoder = FastEmbedEncoder(
        settings.embedding_model, str(settings.embedding_cache_dir) if settings.embedding_cache_dir else None
    )
    return Investigator(bugs, emb, saved["model"], encoder, weights=saved["weights"], top_k=settings.top_k)
