"""Extra statistics for the GPT-4o + RAG answers (scripts/run_rag_baseline.py).

Reads results/domain_expert_v2/{evaluation_report.json, rag_retrieval.json, judge_scores.json}
and writes rag_analysis.json next to them. Standard library only, no API calls.

* citations: share of answers with at least one [Reference N]; share whose every N is
  one of the five retrieved references (1-5); answers citing a number that was not given
* "evidence insufficient" answers: the model says the references do not cover the question
* judge scores split by whether the question's own source chunk was among the five
  retrieved chunks, with the paired difference to GPT-4o without retrieval
  (bootstrap 95 % CI, 5,000 resamples, seed 0 -- as in the other scripts)

Usage:  python scripts/analyze_rag.py
"""
from __future__ import annotations

import json
import random
import re
import statistics
from pathlib import Path

FOLDER = Path("results/domain_expert_v2")
GPT, RAG, RAG600 = "GPT-4o (reference)", "GPT-4o + RAG", "GPT-4o + RAG (600 chars)"
METRICS = ("faithfulness", "clinical_correctness")
REF_RE = re.compile(r"\[Reference\s*(\d+)\]")
INSUFFICIENT_RE = re.compile(
    r"(do(?:es)?n['’]?t|do(?:es)? not|did not)\s+(?:provide|mention|contain|specify|include|address|cover|describe|discuss)"
    r"|not (?:provided|mentioned|specified|covered|addressed|available) in the|insufficient|no (?:specific )?information",
    re.I)


def bootstrap_ci(diffs, n=5000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)]


def main():
    rep = json.loads((FOLDER / "evaluation_report.json").read_text(encoding="utf-8"))
    ret = json.loads((FOLDER / "rag_retrieval.json").read_text(encoding="utf-8"))
    sc = json.loads((FOLDER / "judge_scores.json").read_text(encoding="utf-8"))["scores"]
    items = rep["indomain_answers"]
    hit = [r["gold_rank"] is not None for r in ret["per_question"]]
    out = {"n": len(items), "gold_chunk_in_top5": sum(hit),
           "gold_or_neighbour_in_top5": ret["gold_or_neighbour_in_top5"], "systems": {}}

    for s in (RAG, RAG600, GPT):
        if s not in items[0]:
            continue
        texts = [it[s] for it in items]
        nums = [[int(x) for x in REF_RE.findall(t)] for t in texts]
        d = {"cites_any_reference": sum(bool(n) for n in nums),
             "all_cited_numbers_valid_1_to_5": sum(bool(n) and all(1 <= x <= 5 for x in n) for n in nums),
             "cites_a_number_outside_1_to_5": sum(any(x < 1 or x > 5 for x in n) for n in nums),
             "says_evidence_insufficient": sum(bool(INSUFFICIENT_RE.search(t)) for t in texts),
             "says_evidence_insufficient_when_gold_missed": sum(
                 bool(INSUFFICIENT_RE.search(t)) for t, h in zip(texts, hit) if not h)}
        if all(f"{i}|{s}" in sc for i in range(len(items))):
            for group, keep in (("gold_retrieved", True), ("gold_not_retrieved", False)):
                idx = [i for i, h in enumerate(hit) if h == keep]
                g = {"n": len(idx)}
                for m in METRICS:
                    g[m] = round(statistics.mean(sc[f"{i}|{s}"][m] for i in idx), 2) if idx else None
                    if s != GPT and idx:
                        diff = [sc[f"{i}|{s}"][m] - sc[f"{i}|{GPT}"][m] for i in idx]
                        g[f"{m}_minus_gpt4o_no_rag"] = {"mean_difference": round(statistics.mean(diff), 2),
                                                        "bootstrap_95ci": bootstrap_ci(diff)}
                d[group] = g
        out["systems"][s] = d

    (FOLDER / "rag_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
