"""RAGAS evaluation of the golden set through the real pipeline.
python -m evals.run_ragas --run NAME --variant {baseline,deliberate}

Uses the RAGAS 0.4 per-sample metrics API (ragas.metrics.collections) with gpt-4o-mini as judge."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import uuid
from statistics import mean

from evals.common import init_live, load_jsonl, run_dir
from taxmate import tracing
from taxmate.pipeline import run_turn
from taxmate.profile import TaxProfile
from taxmate.prompts import PROMPT_VERSION

JUDGE_MODEL = "gpt-4o-mini"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def build_metrics(api_key: str, embed_model: str) -> dict:
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecisionWithReference, ContextRecall,
                                           Faithfulness)

    client = AsyncOpenAI(api_key=api_key)
    llm = llm_factory(JUDGE_MODEL, client=client)
    emb = OpenAIEmbeddings(client=client, model=embed_model)
    return {"faithfulness": Faithfulness(llm=llm), "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
            "context_precision": ContextPrecisionWithReference(llm=llm), "context_recall": ContextRecall(llm=llm)}


async def score_sample(metrics: dict, q: str, response: str, contexts: list[str], reference: str) -> dict:
    calls = {
        "faithfulness": metrics["faithfulness"].ascore(user_input=q, response=response, retrieved_contexts=contexts),
        "answer_relevancy": metrics["answer_relevancy"].ascore(user_input=q, response=response),
        "context_precision": metrics["context_precision"].ascore(user_input=q, reference=reference,
                                                                 retrieved_contexts=contexts),
        "context_recall": metrics["context_recall"].ascore(user_input=q, retrieved_contexts=contexts,
                                                           reference=reference),
    }
    results = await asyncio.gather(*calls.values(), return_exceptions=True)
    out = {}
    for name, res in zip(calls, results):
        out[name] = None if isinstance(res, Exception) else float(res.value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", default="deliberate", choices=["baseline", "deliberate"])
    ap.add_argument("--only", nargs="*", help="limit to these golden ids")
    args = ap.parse_args()
    deps, settings = init_live()
    metrics = build_metrics(settings.openai_api_key, settings.embed_model)
    golden = [g for g in load_jsonl("evals/golden_set.jsonl") if not args.only or g["id"] in args.only]

    rows = []
    for g in golden:
        with tracing.trace_context(session_id=f"{args.run}-{g['id']}", tags=[f"run:{args.run}", "ragas", g["id"]]):
            r = run_turn(g["question"], [], TaxProfile(**g["profile"]), deps, str(uuid.uuid4()), args.variant)
        contexts = [h["text"] for h in r.record["retrieved"]]
        row = {"id": g["id"], "in_corpus": g["in_corpus"], "kind": r.kind, "citation_check": r.record["citation_check"],
               "top_score": r.record["top_score"], "langfuse_trace_id": r.record["langfuse_trace_id"],
               "question": g["question"], "response": r.text, "reference": g["reference_answer"],
               "retrieved_chunk_ids": " | ".join(h["chunk_id"] for h in r.record["retrieved"])}
        if g["in_corpus"] and contexts:
            row.update(asyncio.run(score_sample(metrics, g["question"], r.text, contexts, g["reference_answer"])))
        else:
            row.update({m: None for m in METRICS})
        rows.append(row)
        print(f"{g['id']}: {r.kind} " + " ".join(f"{m}={row[m]:.2f}" for m in METRICS if row[m] is not None))
    tracing.flush()

    d = run_dir(args.run)
    with (d / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    inc = [r for r in rows if r["in_corpus"]]
    ooc = [r for r in rows if not r["in_corpus"]]
    summary = {
        "run": args.run, "variant": args.variant, "prompt_version": PROMPT_VERSION, "judge_model": JUDGE_MODEL,
        "ragas_version": __import__("ragas").__version__, "threshold": deps.threshold, "k": deps.k,
        "n_in_corpus": len(inc),
        **{m: round(mean(v for r in inc if (v := r[m]) is not None), 3) if any(r[m] is not None for r in inc) else None
           for m in METRICS},
        "in_corpus_fallback_rate": round(sum(r["kind"] == "fallback" for r in inc) / len(inc), 3) if inc else None,
        "out_of_corpus_fallback_rate": round(sum(r["kind"] == "fallback" for r in ooc) / len(ooc), 3) if ooc else None,
        "threshold_env": os.environ.get("TAXMATE_THRESHOLD"),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
