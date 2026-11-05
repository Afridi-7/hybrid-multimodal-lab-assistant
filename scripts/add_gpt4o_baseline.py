"""Add the GPT-4o reference answers to a saved evaluation_report.json.

The Colab notebook answers the held-out questions with GPT-4o too, but in the v2
run (Nov 2026) that step failed inside Colab, so the report contains only the two
Llama systems. GPT-4o needs no GPU, so this script adds its answers afterwards with
EXACTLY the notebook's settings: model gpt-4o, temperature 0, max_tokens 300, the
same system prompt as in training, the same questions in the same order.

Usage (repo root, OPENAI_API_KEY in .env):
    python scripts/add_gpt4o_baseline.py
    python scripts/add_gpt4o_baseline.py --report results/domain_expert_v2/evaluation_report.json
Cost: 100 short gpt-4o calls, well under 1 USD. Re-running skips answers already present.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPT = "GPT-4o (reference)"
# Identical to SYSTEM_PROMPT in the notebook and in scripts/build_finetune_dataset.py.
SYSTEM_PROMPT = ("You are a clinical hematology expert. Always ground answers in the provided textbook reference. "
                 "Use cautious, evidence-based language. When evidence is insufficient, say so explicitly. "
                 "Cite using [Reference N] when paraphrasing the source.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=ROOT / "results/domain_expert_v2/evaluation_report.json")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (put it in .env).")
    from openai import OpenAI
    client = OpenAI()

    rep = json.loads(args.report.read_text(encoding="utf-8"))
    items = rep["indomain_answers"]
    todo = [i for i, it in enumerate(items) if not it.get(GPT)]
    print(f"{len(items)} questions, {len(todo)} without a GPT-4o answer")
    for n, i in enumerate(todo, 1):
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o", temperature=0, max_tokens=300,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": items[i]["q"]}])
                items[i][GPT] = r.choices[0].message.content.strip()
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  retry {attempt + 1} for question {i}: {exc}")
                time.sleep(2 ** attempt)
        else:
            sys.exit(f"giving up at question {i}; re-run to continue (finished answers are kept)")
        if n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}")
            args.report.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    rep.setdefault("notes", []).append(
        f"{GPT} answers added on {dt.date.today().isoformat()} by scripts/add_gpt4o_baseline.py "
        "(gpt-4o, temperature 0, max_tokens 300, training system prompt) because the Colab step failed.")
    args.report.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", args.report)


if __name__ == "__main__":
    main()
