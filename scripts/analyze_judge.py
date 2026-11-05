"""Summarise LLM-judge scores for the thesis (means, SDs, paired bootstrap CIs).

scripts/judge_domain_expert.py saves the ratings (300 for the three original systems, 200 more
for the two GPT-4o + RAG systems) and a short summary that only
compares the fine-tuned model with the base model. The thesis also reports
standard deviations and GPT-4o minus fine-tuned; this script computes all of them
from the saved ratings, with the same bootstrap procedure (5,000 resamples, seed 0).
No API calls, standard library only.

Usage:
    python scripts/analyze_judge.py                                   # v2 run
    python scripts/analyze_judge.py --scores results/domain_expert/judge_scores.json
Output: judge_analysis.json next to the scores file.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

BASE, OURS, GPT = "Llama-3.1-8B (base)", "Llama-3.1-8B + QLoRA (ours)", "GPT-4o (reference)"
RAG, RAG600 = "GPT-4o + RAG", "GPT-4o + RAG (600 chars)"      # added by scripts/run_rag_baseline.py
METRICS = ("faithfulness", "clinical_correctness")


def bootstrap_ci(diffs, n=5000, seed=0):                    # identical to judge_domain_expert.py
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/domain_expert_v2/judge_scores.json"))
    args = ap.parse_args()
    data = json.loads(args.scores.read_text(encoding="utf-8"))
    sc = data["scores"]
    idx = sorted({int(k.split("|", 1)[0]) for k in sc})
    systems = [s for s in (BASE, OURS, GPT, RAG, RAG600) if any(k.endswith("|" + s) for k in sc)]
    out = {"judge_model": data.get("judge_model"), "n_ratings": len(sc), "systems": {}, "paired": {}}
    for s in systems:
        rows = [sc[f"{i}|{s}"] for i in idx if f"{i}|{s}" in sc]
        out["systems"][s] = {"n": len(rows), **{
            m: {"mean": round(statistics.mean(r[m] for r in rows), 2),
                "sd": round(statistics.stdev(r[m] for r in rows), 2)} for m in METRICS}}
    for a, b in ((OURS, BASE), (GPT, OURS), (GPT, BASE), (RAG, GPT), (RAG, OURS), (RAG600, RAG)):
        if a in systems and b in systems:
            for m in METRICS:
                d = [sc[f"{i}|{a}"][m] - sc[f"{i}|{b}"][m] for i in idx if f"{i}|{a}" in sc and f"{i}|{b}" in sc]
                out["paired"][f"{a} minus {b}: {m}"] = {"n": len(d), "mean_difference": round(statistics.mean(d), 2),
                                                        "bootstrap_95ci": bootstrap_ci(d)}
    dst = args.scores.parent / "judge_analysis.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
