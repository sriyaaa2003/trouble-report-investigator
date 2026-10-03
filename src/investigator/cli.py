"""Command line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from investigator.settings import Settings


def _cmd_fetch(args: argparse.Namespace) -> int:
    from investigator.data.fetch import run_fetch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(
        run_fetch(
            Path(args.out), args.start_year, args.end_year, args.max_dups, args.max_fixed, args.seed, args.concurrency
        )
    )
    return 0


def _prepare(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from investigator.build import embed_bugs
    from investigator.data.dataset import load_bugs
    from investigator.search.dense import FastEmbedEncoder

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = Settings()
    bugs = load_bugs(Path(args.data))
    if not bugs:
        raise SystemExit(f"no bugs in {args.data}; run `investigator fetch` first")
    encoder = FastEmbedEncoder(s.embedding_model)
    emb = embed_bugs(bugs, encoder, s.embedding_model, s.artifacts_dir)
    return bugs, emb


def _cmd_eval_dups(args: argparse.Namespace) -> int:
    from investigator.evaluate.dups import evaluate

    bugs, emb = _prepare(args)
    res = evaluate(bugs, emb, datetime.fromisoformat(args.test_from), datetime.fromisoformat(args.dev_from))
    print(f"corpus={res.n_corpus} bugs, test queries={res.n_queries}, dev queries={int(res.chosen['dev_queries'])}")
    print(f"chosen on dev: {res.chosen}\n")
    print(res.table())
    if args.out:
        est = {n: {k: [v.mean, v.low, v.high] for k, v in m.items()} for n, m in res.estimates().items()}
        Path(args.out).write_text(
            json.dumps(
                {"n_test_queries": res.n_queries, "n_corpus": res.n_corpus, "chosen": res.chosen, "estimates": est},
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


def _cmd_eval_components(args: argparse.Namespace) -> int:
    from investigator.evaluate.components import evaluate

    bugs, emb = _prepare(args)
    res = evaluate(bugs, emb, datetime.fromisoformat(args.dev_from), datetime.fromisoformat(args.test_from))
    print(
        f"train={res.n_train}, test={res.n_test} (coverage {res.coverage:.1%} of test reports), classes={res.n_classes}"
    )
    print(f"ensemble weights chosen on dev: {res.chosen_weights}\n")
    print(res.table())
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "chosen_weights": res.chosen_weights,
                    "n_train": res.n_train,
                    "n_test": res.n_test,
                    "coverage": res.coverage,
                    "n_classes": res.n_classes,
                    "macro_f1": res.macro_f1,
                    "parent_top1": res.parent_top1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from investigator.build import build_artifacts

    out = build_artifacts(Settings(), Path(args.weights) if args.weights else None)
    print(f"artifacts written to {out}")
    return 0


def _cmd_investigate(args: argparse.Namespace) -> int:
    from investigator.build import load_investigator

    inv = load_investigator(Settings())
    result = inv.investigate(args.summary, args.description or "")
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    print("Likely subsystems:")
    for name, p in result.components[:3]:
        print(f"  {p:5.2f}  {name}")
    print("\nSimilar historical reports:")
    for c in result.similar_cases[:5]:
        print(f"  #{c.bug_id}  cos={c.similarity_dense:.2f}  [{c.component} / {c.resolution}]  {c.summary[:80]}")
        if c.matched_terms:
            print(f"      matched terms: {', '.join(c.matched_terms)}")
    print("\nSuggested investigation:")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step}")
    for w in result.warnings:
        print(f"\n! {w}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from investigator.api import create_app

    uvicorn.run(create_app(Settings()), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="investigator", description="Trouble-report investigation toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download public Mozilla Core bug reports (cached, resumable)")
    f.add_argument("--out", default="data/mozilla")
    f.add_argument("--start-year", type=int, default=2021)
    f.add_argument("--end-year", type=int, default=2024)
    f.add_argument("--max-dups", type=int, default=3000, help="duplicate bugs to sample")
    f.add_argument("--max-fixed", type=int, default=9000, help="fixed bugs to sample")
    f.add_argument("--seed", type=int, default=13)
    f.add_argument("--concurrency", type=int, default=6)
    f.set_defaults(fn=_cmd_fetch)

    for name, fn, help_ in (
        ("eval-dups", _cmd_eval_dups, "benchmark similar-report (duplicate) retrieval"),
        ("eval-components", _cmd_eval_components, "benchmark affected-subsystem prediction"),
    ):
        e = sub.add_parser(name, help=help_)
        e.add_argument("--data", default="data/mozilla")
        e.add_argument(
            "--dev-from", default="2023-01-01", help="start of the development period (hyper-parameter tuning)"
        )
        e.add_argument("--test-from", default="2024-01-01", help="start of the held-out test period")
        e.add_argument("--out", default=None, help="write summary JSON here")
        e.set_defaults(fn=fn)

    b = sub.add_parser("build", help="embed reports and fit the component model for investigate/serve")
    b.add_argument("--weights", default="results/components.json", help="ensemble weights from eval-components")
    b.set_defaults(fn=_cmd_build)

    i = sub.add_parser("investigate", help="analyse a new report")
    i.add_argument("summary")
    i.add_argument("--description", default="")
    i.add_argument("--json", action="store_true")
    i.set_defaults(fn=_cmd_investigate)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_cmd_serve)

    args = p.parse_args(argv)
    code: int = args.fn(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
