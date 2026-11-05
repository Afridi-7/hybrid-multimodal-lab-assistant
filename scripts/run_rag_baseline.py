"""Answer the held-out questions with GPT-4o + retrieval (RAG): does retrieval help?

The thesis compares base Llama, fine-tuned Llama and GPT-4o *without* retrieval.
This script adds the missing condition, GPT-4o *with* the system's own retrieval,
to results/domain_expert_v2/evaluation_report.json (existing answers are not touched):

  "GPT-4o + RAG"                 top-5 chunks from the app's retriever (ClinicalRetriever:
                                 bge-small-en-v1.5 + query instruction, ChromaDB index in
                                 data/chroma_db), FULL chunk text, formatted exactly as in
                                 linear mode (ClinicalRetriever.format_retrieved_context).
  "GPT-4o + RAG (600 chars)"     the same five chunks, each cut to 600 characters, exactly as
                                 the ReAct agent's query_knowledge_base tool shows them
                                 (src/rag/agent.py::_format_chunks).

Everything else is identical to scripts/add_gpt4o_baseline.py (the no-RAG GPT-4o answers):
model gpt-4o, temperature 0, max_tokens 300, the training system prompt, the same 100
questions in the same order. The question itself is the retrieval query.

It also writes results/domain_expert_v2/rag_retrieval.json: for every question the retrieved
(source, chunk_id, score) and whether the question's own source chunk (test_split.jsonl) was
among the five. Only ids and scores are saved, no textbook text.

Usage (repo root, project venv, OPENAI_API_KEY in .env, data/pdfs + data/chroma_db present):
    python scripts/run_rag_baseline.py
Cost: 200 short gpt-4o calls (about 1 USD). Re-running skips answers already present.
Then:  python scripts/judge_domain_expert.py --report results/domain_expert_v2/evaluation_report.json --out results/domain_expert_v2/judge_scores.json
       python scripts/analyze_domain_expert.py ; python scripts/analyze_judge.py ; python scripts/analyze_rag.py
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
sys.path.insert(0, str(ROOT))

RAG = "GPT-4o + RAG"
RAG600 = "GPT-4o + RAG (600 chars)"
# Identical to scripts/add_gpt4o_baseline.py, the notebook and scripts/build_finetune_dataset.py.
SYSTEM_PROMPT = ("You are a clinical hematology expert. Always ground answers in the provided textbook reference. "
                 "Use cautious, evidence-based language. When evidence is insufficient, say so explicitly. "
                 "Cite using [Reference N] when paraphrasing the source.")


def user_message(context: str, question: str) -> str:
    return f"Textbook references:\n\n{context}\n\nQuestion: {question}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=ROOT / "results/domain_expert_v2/evaluation_report.json")
    ap.add_argument("--test-split", type=Path, default=ROOT / "results/domain_expert_v2/test_split.jsonl")
    args = ap.parse_args()
    os.chdir(ROOT)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (put it in .env).")

    from src.config.config_loader import load_config
    from src.rag.agent import _format_chunks
    from src.rag.retriever import ClinicalRetriever

    config = load_config(str(ROOT / "config.yaml"))
    retriever = ClinicalRetriever(config)
    retriever.build_index()                      # reuses data/chroma_db when it is up to date
    print("embedding model:", retriever.embedding_model_name, "| top_k:", retriever.top_k,
          "| internet:", retriever.internet_enabled)

    rep = json.loads(args.report.read_text(encoding="utf-8"))
    items = rep["indomain_answers"]
    tests = [json.loads(line) for line in args.test_split.open(encoding="utf-8") if line.strip()]
    by_q = {next(m["content"] for m in t["messages"] if m["role"] == "user"): t for t in tests}

    from openai import OpenAI
    client = OpenAI()

    def ask(content: str) -> str:
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o", temperature=0, max_tokens=300,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": content}])
                return r.choices[0].message.content.strip()
            except Exception as exc:  # noqa: BLE001
                print(f"  retry {attempt + 1}: {exc}")
                time.sleep(2 ** attempt)
        raise RuntimeError("OpenAI call failed 4 times; re-run to continue (finished answers are kept)")

    retrieval = []
    for i, it in enumerate(items):
        chunks = retriever.retrieve(it["q"])
        gold = by_q[it["q"]]
        gold_key = (gold["source"], int(gold["chunk_id"]))
        got = [(c.get("source"), int(c.get("chunk_id"))) for c in chunks]
        neighbours = {(gold_key[0], gold_key[1] + d) for d in (-1, 0, 1)}
        retrieval.append({
            "question_index": i,
            "gold": f"{gold_key[0]}#{gold_key[1]}",
            "retrieved": [{"source": s, "chunk_id": c, "score": round(float(ch.get("score", 0.0)), 4)}
                          for (s, c), ch in zip(got, chunks)],
            "gold_rank": (got.index(gold_key) + 1) if gold_key in got else None,
            "gold_or_neighbour_retrieved": bool(neighbours & set(got)),
        })
        if not it.get(RAG):
            it[RAG] = ask(user_message(retriever.format_retrieved_context(chunks), it["q"]))
        if not it.get(RAG600):
            it[RAG600] = ask(user_message(_format_chunks(chunks, 0), it["q"]))
        if (i + 1) % 10 == 0 or i + 1 == len(items):
            print(f"  {i + 1}/{len(items)}")
            args.report.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")

    hits = sum(r["gold_rank"] is not None for r in retrieval)
    out = {"retriever": {"embedding_model": retriever.embedding_model_name,
                         "query_instruction": retriever.query_instruction, "top_k": retriever.top_k,
                         "internet_enabled": retriever.internet_enabled},
           "n": len(retrieval), "gold_chunk_in_top5": hits,
           "gold_or_neighbour_in_top5": sum(r["gold_or_neighbour_retrieved"] for r in retrieval),
           "per_question": retrieval}
    (args.report.parent / "rag_retrieval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    rep.setdefault("notes", []).append(
        f"{RAG} and {RAG600} answers added on {dt.date.today().isoformat()} by scripts/run_rag_baseline.py "
        "(gpt-4o, temperature 0, max_tokens 300, training system prompt; top-5 chunks from the app's "
        "retriever; full chunks as in linear mode / 600-character chunks as in the agent tool).")
    args.report.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"source chunk among the 5 retrieved: {hits}/{len(retrieval)}")
    print("saved", args.report)


if __name__ == "__main__":
    main()
