"""Recompute every domain-expert statistic reported in the thesis from saved answers.

Input : an evaluation_report.json written by the Colab notebook
        (default results/domain_expert_v2/evaluation_report.json)
Output: <same folder>/analysis.json  and  <same folder>/flagged_citations.csv

Computed, per system (base Llama, fine-tuned Llama, GPT-4o):
  * ROUGE-L F1 vs. the reference answer (rouge_score, stemming) - as in the notebook
  * semantic similarity (all-MiniLM-L6-v2 cosine)               - as in the notebook
  * source-marker rate (same regex as the notebook), average length in words
  * bare "[Reference N]" markers (no reference was ever given to the models)
  * invented bibliographic references, by an explicit rule:
      (a) a quoted title after "textbook"/"book", e.g.  textbook "Williams Hematology"
      (b) a reference-list entry, e.g.  Lichtman, M. A. (2018)   or   ... et al.
    Every flagged answer is written to flagged_citations.csv - CHECK THEM BY HAND
    and report the hand-checked number.
Paired statistics (same questions for all systems):
  * 95 % bootstrap CI of the mean ROUGE-L difference (5,000 resamples, seed 0,
    same procedure as scripts/judge_domain_expert.py)
  * number of questions on which the fine-tuned model has the higher ROUGE-L
  * MedMCQA: accuracy, unparsable answers, discordant pairs, exact McNemar p
  * which book the fine-tuned model names in its source line, and whether it matches
    the book named in the reference answer

Usage:
    python scripts/analyze_domain_expert.py
    python scripts/analyze_domain_expert.py --report results/domain_expert/evaluation_report.json --no-semantic
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from pathlib import Path

BASE, OURS, GPT = "Llama-3.1-8B (base)", "Llama-3.1-8B + QLoRA (ours)", "GPT-4o (reference)"
CITE_RE = re.compile(r"source:|\[reference|haematology\.pdf", re.I)          # notebook's source-marker rule
BARE_REF_RE = re.compile(r"\[Reference\s*\d+\]")
TITLE_RE = re.compile(r"(?:textbook|book)\s*(?:titled\s*|called\s*|,\s*)?[\"“‘'][A-Z][^\"”’']{2,80}[\"”’']")
REFLIST_RE = re.compile(r"\b[A-Z][a-z]+,\s+[A-Z]\.(?:\s*[A-Z]\.)*(?:,|\s*\(|\s+&|\s+and)|\bet al\.")
BOOK_RE = re.compile(r"(essentials_haematology|consie_haematology)\.pdf", re.I)


def bootstrap_ci(diffs, n=5000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)]


def letter(t):                                                                 # notebook's MCQ parser
    m = re.search(r"\b([ABCD])\b", t.upper())
    return m.group(1) if m else None


def mcnemar_exact(b, c):
    from scipy.stats import binomtest
    return float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=Path("results/domain_expert_v2/evaluation_report.json"))
    ap.add_argument("--no-semantic", action="store_true", help="skip the embedding-similarity metric")
    args = ap.parse_args()
    rep = json.loads(args.report.read_text(encoding="utf-8"))
    items = rep["indomain_answers"]
    systems = [s for s in (BASE, OURS, GPT) if s in items[0]]
    golds = [it["gold"] for it in items]

    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = {s: [rs.score(g, it[s])["rougeL"].fmeasure for g, it in zip(golds, items)] for s in systems}

    sem = {}
    if not args.no_semantic:
        from sentence_transformers import SentenceTransformer, util
        emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        ge = emb.encode(golds, convert_to_tensor=True, normalize_embeddings=True)
        for s in systems:
            pe = emb.encode([it[s] for it in items], convert_to_tensor=True, normalize_embeddings=True)
            sem[s] = util.cos_sim(pe, ge).diagonal().cpu().tolist()

    flagged_rows, per_system = [], {}
    for s in systems:
        texts = [it[s] for it in items]
        inv = []
        for i, t in enumerate(texts):
            m = TITLE_RE.search(t) or REFLIST_RE.search(t)
            if m:
                inv.append(i)
                a, b = max(0, m.start() - 40), min(len(t), m.end() + 60)
                flagged_rows.append([s, i, t[a:b].replace("\n", " ")])
        per_system[s] = {
            "n": len(texts),
            "rouge_l": round(statistics.mean(rouge[s]), 3),
            "semantic_sim": round(statistics.mean(sem[s]), 3) if s in sem else None,
            "source_marker_pct": round(100 * sum(bool(CITE_RE.search(t)) for t in texts) / len(texts), 1),
            "avg_words": round(statistics.mean(len(t.split()) for t in texts), 1),
            "bare_reference_marker": sum(bool(BARE_REF_RE.search(t)) for t in texts),
            "invented_reference_flagged": len(inv),
        }

    paired = {}
    if OURS in rouge:
        for other in (BASE, GPT):
            if other in rouge:
                d = [a - b for a, b in zip(rouge[OURS], rouge[other])]
                paired[f"rouge_l ours - {other}"] = {
                    "mean_diff": round(statistics.mean(d), 3), "bootstrap_95ci": bootstrap_ci(d),
                    "ours_higher_on": sum(x > 0 for x in d), "of": len(d)}

    books = {}
    if OURS in systems:
        named = [BOOK_RE.search(it[OURS]) for it in items]
        gold_book = [BOOK_RE.search(g) for g in golds]
        books = {"ours_names_book1": sum(bool(m) and m.group(1).lower().startswith("essentials") for m in named),
                 "ours_names_book2": sum(bool(m) and m.group(1).lower().startswith("consie") for m in named),
                 "reference_answers_from_book2": sum(bool(m) and m.group(1).lower().startswith("consie") for m in gold_book),
                 "book2_questions_labelled_book2_by_ours": sum(
                     bool(g) and g.group(1).lower().startswith("consie") and bool(n) and n.group(1).lower().startswith("consie")
                     for g, n in zip(gold_book, named))}

    mcq = {}
    if rep.get("mcq_answers"):
        ma = rep["mcq_answers"]
        pb = [letter(x["base"]) for x in ma]; pf = [letter(x["ft"]) for x in ma]; g = [x["gold"] for x in ma]
        cb = [p == t for p, t in zip(pb, g)]; cf = [p == t for p, t in zip(pf, g)]
        b = sum(x and not y for x, y in zip(cb, cf)); c = sum(y and not x for x, y in zip(cb, cf))
        mcq = {"n": len(ma), "acc_base_pct": round(100 * sum(cb) / len(ma), 1), "acc_ft_pct": round(100 * sum(cf) / len(ma), 1),
               "unparsable_base": pb.count(None), "unparsable_ft": pf.count(None),
               "base_only_correct": b, "ft_only_correct": c, "mcnemar_exact_p": round(mcnemar_exact(b, c), 4)}

    reference = {"n": len(golds), "avg_words": round(statistics.mean(len(g.split()) for g in golds), 1),
                 "source_marker_pct": round(100 * sum(bool(CITE_RE.search(g)) for g in golds) / len(golds), 1)}
    out = {"report": str(args.report), "config": rep.get("config"), "systems": per_system,
           "reference_answers": reference,
           "paired_rouge": paired, "source_line_books": books, "medmcqa": mcq,
           "invented_reference_rule": "quoted title after textbook/book, or reference-list entry (Surname, X. / et al.); flagged answers listed in flagged_citations.csv and must be checked by hand"}
    folder = args.report.parent
    (folder / "analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    with (folder / "flagged_citations.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["system", "question_index", "snippet"]); w.writerows(flagged_rows)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
