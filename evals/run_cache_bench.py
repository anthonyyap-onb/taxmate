"""Caching before/after benchmark: python -m evals.run_cache_bench --run NAME

Measures, through the same pipeline as the app (response cache ON here, unlike the other runners):
1. Document index at startup: embedding the corpus (cold) vs loading the saved vectors (warm).
2. Per question: a cache miss, an exact repeat, and a paraphrase (semantic hit or miss), with latency and cost.
3. The same question under two different profiles: the second must MISS.
4. Similarity of paraphrase pairs vs near-miss pairs (look alike, need different answers), and whether the
   salient-fact check keeps the near misses apart. This is what the semantic threshold is tuned on.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
import time
import uuid

import numpy as np

from evals.common import ROOT, run_dir
from taxmate import tracing, usage
from taxmate.cache import corpus_fingerprint, load_or_build_index, salient_tokens
from taxmate.config import load_settings, missing_keys
from taxmate.corpus import load_chunks
from taxmate.live import build_live_deps
from taxmate.pipeline import run_turn
from taxmate.profile import TaxProfile
from taxmate.retrieval import make_embedder


def bench_index(s) -> dict:
    embed = make_embedder(tracing.make_openai_client(s), s.embed_model)
    chunks = load_chunks()
    fp = corpus_fingerprint(chunks, s.embed_model)
    with tempfile.TemporaryDirectory() as d:
        with usage.track() as u:
            t0 = time.perf_counter()
            _, how1 = load_or_build_index(chunks, embed, fp, d)
            cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, how2 = load_or_build_index(chunks, embed, fp, d)
        warm = time.perf_counter() - t0
    return {"n_chunks": len(chunks), "cold": how1, "cold_ms": int(cold * 1000), "warm": how2,
            "warm_ms": round(warm * 1000, 1), **{f"cold_{k}": v for k, v in u.summary().items()}}


def turn_row(label: str, qid: str, r) -> dict:
    rec = r.record
    return {"case": label, "id": qid, "kind": r.kind, "cache": rec["cache"], "similarity": rec["cache_similarity"],
            "latency_ms": rec["latency_ms"]["total"], "generation_ms": rec["latency_ms"]["generation"],
            "retrieval_ms": rec["latency_ms"]["retrieval"], "guard_profile_ms":
                rec["latency_ms"]["guard"] + rec["latency_ms"]["profile"],
            "prompt_tokens": rec["tokens"]["prompt"], "completion_tokens": rec["tokens"]["completion"],
            "cost_usd": rec["est_cost_usd"]}


def cosine(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    s = load_settings()
    if missing_keys(s):
        raise SystemExit("Fill in .env first.")
    tracing.init(s)
    spec = json.loads((ROOT / "evals" / "cache_bench.json").read_text(encoding="utf-8"))
    index = bench_index(s)

    deps, _ = build_live_deps(s, response_cache=True)
    rows = []
    with tracing.trace_context(session_id=f"{args.run}", tags=[f"run:{args.run}", "cache-bench"]):
        for p in spec["pairs"]:
            for label, text in (("miss", p["question"]), ("exact_repeat", p["question"]),
                                ("paraphrase", p["paraphrase"])):
                rows.append(turn_row(label, p["id"], run_turn(text, [], TaxProfile(), deps, str(uuid.uuid4()))))
        for i, p in enumerate(spec["profile_pairs"], start=1):
            for label, prof in (("profile_a", p["profile_a"]), ("different_profile", p["profile_b"])):
                rows.append(turn_row(label, f"P{i}", run_turn(p["question"], [], TaxProfile(**prof), deps,
                                                             str(uuid.uuid4()))))
    tracing.flush()

    embed = make_embedder(tracing.make_openai_client(s), s.embed_model)
    sims = []
    for p in spec["pairs"]:
        a, b = embed([p["question"], p["paraphrase"]])
        sims.append({"type": "paraphrase", "a": p["question"], "b": p["paraphrase"], "cosine": round(cosine(a, b), 4),
                     "same_salient": salient_tokens(p["question"]) == salient_tokens(p["paraphrase"])})
    for p in spec["near_misses"]:
        a, b = embed([p["a"], p["b"]])
        sims.append({"type": "near_miss", "a": p["a"], "b": p["b"], "cosine": round(cosine(a, b), 4),
                     "same_salient": salient_tokens(p["a"]) == salient_tokens(p["b"])})

    out = run_dir(args.run)
    for name, data in (("turns.csv", rows), ("similarity.csv", sims)):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    def agg(case: str) -> dict:
        rs = [r for r in rows if r["case"] == case]
        return {"n": len(rs), "median_latency_ms": statistics.median(r["latency_ms"] for r in rs),
                "mean_cost_usd": round(statistics.mean(r["cost_usd"] or 0 for r in rs), 6),
                "cache_modes": {m: sum(r["cache"] == m for r in rs) for m in {r["cache"] for r in rs}}}

    summary = {"run": args.run, "semantic_threshold": s.semantic_threshold, "index": index,
               **{c: agg(c) for c in ("miss", "exact_repeat", "paraphrase", "profile_a", "different_profile")},
               "paraphrase_cosines": sorted(x["cosine"] for x in sims if x["type"] == "paraphrase"),
               "near_miss_cosines": sorted(x["cosine"] for x in sims if x["type"] == "near_miss"),
               "near_misses_blocked_by_salient_check": sum(not x["same_salient"] for x in sims
                                                           if x["type"] == "near_miss"),
               "n_near_misses": len(spec["near_misses"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
