"""Score the saved domain-expert answers with an LLM judge (GPT-4o-mini).

Replaces the judge step of ``domain_expert_evaluation.ipynb``, which failed
because the prompt template contained literal JSON braces that
``str.format`` tried to fill in. This script needs no GPU: it reads the
answers already saved in ``results/domain_expert/evaluation_report.json``.

For every held-out question and every system (base Llama, fine-tuned Llama,
GPT-4o) the judge sees the question, the textbook reference answer and ONE
model answer -- never the system's name -- and returns two 1-5 scores:

* faithfulness          -- agrees with the reference; no unsupported claims
* clinical_correctness  -- medically accurate

Usage (from the repository root, with OPENAI_API_KEY in .env):

    python scripts/judge_domain_expert.py
    python scripts/judge_domain_expert.py --limit 5      # quick test
    python scripts/judge_domain_expert.py --dry-run      # no API calls

Scores are cached in ``judge_scores.json``; re-running only scores missing
items, so an interrupted run can simply be restarted.
Cost: ~300 short calls to gpt-4o-mini, well under 1 USD.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "results" / "domain_expert" / "evaluation_report.json"
DEFAULT_OUT = ROOT / "results" / "domain_expert" / "judge_scores.json"

JUDGE_INSTRUCTIONS = (
    "You are grading an answer written by a hematology assistant.\n"
    "Compare the MODEL ANSWER with the GOLD ANSWER, which was written from a "
    "hematology textbook passage.\n"
    "Give two integer scores from 1 (worst) to 5 (best):\n"
    "- faithfulness: agrees with the gold answer and adds no unsupported or "
    "contradicting claims;\n"
    "- clinical_correctness: the medical content is accurate.\n"
    "A source line such as 'Source: file.pdf' or '[Reference 1]' earns no "
    "credit by itself. Length does not matter.\n"
    'Return ONLY a JSON object: {"faithfulness": <int>, '
    '"clinical_correctness": <int>, "rationale": "<one sentence>"}'
)


def build_prompt(question: str, gold: str, answer: str) -> str:
    # Plain concatenation: no str.format, so JSON braces are safe.
    return (
        JUDGE_INSTRUCTIONS
        + "\n\nQUESTION:\n" + question
        + "\n\nGOLD ANSWER:\n" + gold
        + "\n\nMODEL ANSWER:\n" + answer
    )


def load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def bootstrap_ci(diffs: List[float], n: int = 5000, seed: int = 0) -> List[float]:
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(n)
    )
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N questions")
    ap.add_argument("--dry-run", action="store_true", help="build prompts, make no API calls")
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = report["indomain_answers"]
    if args.limit:
        items = items[: args.limit]
    systems = [k for k in items[0] if k not in ("q", "gold")]
    print(f"{len(items)} questions x {len(systems)} systems: {systems}")

    cache: Dict[str, Any] = {}
    if args.out.exists():
        cache = json.loads(args.out.read_text(encoding="utf-8")).get("scores", {})

    if args.dry_run:
        print(build_prompt(items[0]["q"], items[0]["gold"], items[0][systems[0]])[:1200])
        print(f"\n[dry run] would make up to {len(items) * len(systems)} calls.")
        return

    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (put it in .env or the environment).")
    from openai import OpenAI  # type: ignore
    client = OpenAI()

    def save() -> None:
        args.out.write_text(json.dumps({"judge_model": args.model, "scores": cache}, indent=1), encoding="utf-8")

    for i, it in enumerate(items):
        for s in systems:
            key = f"{i}|{s}"
            if key in cache:
                continue
            prompt = build_prompt(it["q"], it["gold"], it[s])
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model=args.model, temperature=0,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": prompt}],
                    )
                    d = json.loads(r.choices[0].message.content)
                    cache[key] = {
                        "faithfulness": int(d["faithfulness"]),
                        "clinical_correctness": int(d["clinical_correctness"]),
                        "rationale": str(d.get("rationale", ""))[:300],
                    }
                    break
                except Exception as exc:  # noqa: BLE001 - retry any API/parse error
                    print(f"  [{key}] attempt {attempt + 1} failed: {exc}")
                    time.sleep(2 * (attempt + 1))
        if (i + 1) % 10 == 0:
            save()
            print(f"  {i + 1}/{len(items)} questions scored")
    save()

    # ---- summary
    summary: Dict[str, Any] = {}
    for s in systems:
        rows = [cache[f"{i}|{s}"] for i in range(len(items)) if f"{i}|{s}" in cache]
        summary[s] = {
            "n": len(rows),
            "faithfulness": round(statistics.mean(r["faithfulness"] for r in rows), 2),
            "clinical_correctness": round(statistics.mean(r["clinical_correctness"] for r in rows), 2),
        }
    base, ft = systems[0], systems[1]
    for metric in ("faithfulness", "clinical_correctness"):
        diffs = [
            cache[f"{i}|{ft}"][metric] - cache[f"{i}|{base}"][metric]
            for i in range(len(items))
            if f"{i}|{ft}" in cache and f"{i}|{base}" in cache
        ]
        if diffs:
            summary[f"{ft} minus {base}: {metric}"] = {
                "mean_difference": round(statistics.mean(diffs), 3),
                "bootstrap_95ci": bootstrap_ci(diffs),
            }
    data = json.loads(args.out.read_text(encoding="utf-8"))
    data["summary"] = summary
    args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
