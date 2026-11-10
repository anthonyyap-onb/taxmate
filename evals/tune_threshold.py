"""Record top retrieval scores for the golden set to choose the fallback threshold.
python -m evals.tune_threshold --run NAME"""
from __future__ import annotations

import argparse
import csv

from evals.common import init_live, load_jsonl, run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    deps, _ = init_live()
    rows = []
    for g in load_jsonl("evals/golden_set.jsonl"):
        hits = deps.retrieve(g["question"], deps.k)
        top = hits[0].score if hits else 0.0
        ref_hit = any(h.chunk.doc_id == g["reference_doc_id"] for h in hits)
        rows.append({"id": g["id"], "in_corpus": g["in_corpus"], "top_score": round(top, 4),
                     "top_chunk": hits[0].chunk.chunk_id if hits else "", "reference_doc_in_top_k": ref_hit})
    out = run_dir(args.run) / "scores.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    inc = sorted(r["top_score"] for r in rows if r["in_corpus"])
    ooc = sorted(r["top_score"] for r in rows if not r["in_corpus"])
    suggested = inc[0] - 0.005
    fallback_rate = sum(s < suggested for s in ooc) / len(ooc) if ooc else 0.0
    print(f"In-corpus top scores: min={inc[0]:.3f} median={inc[len(inc) // 2]:.3f} max={inc[-1]:.3f}")
    print(f"Out-of-corpus top scores: {[round(s, 3) for s in ooc]}")
    print(f"Suggested threshold (keeps every in-corpus question): {suggested:.3f}; "
          f"out-of-corpus fallback rate at that value: {fallback_rate:.0%}")
    print(f"Reference doc retrieved in top-{deps.k}: "
          f"{sum(r['reference_doc_in_top_k'] for r in rows if r['in_corpus'])}/{len(inc)} -> {out}")


if __name__ == "__main__":
    main()
