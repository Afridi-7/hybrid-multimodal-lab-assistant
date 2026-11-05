"""Retrieval evaluation: does a question find the chunk it was written from?

Every record in data/finetune/train_v2.jsonl stores the textbook chunk its
question was generated from (chunk_key = "<source file>#<chunk index>"). That
gives a known "correct" chunk per question, so retrieval can be measured:

  hit@1 / hit@5   share of questions whose source chunk is ranked 1st / in the top 5
                  (the system retrieves the top 5, see rag.retrieval.top_k)
  MRR@10          mean reciprocal rank of the source chunk (0 if not in the top 10)
  hit@5_or_neighbour  source chunk OR its direct neighbour (same book, index +-1) in the
                  top 5. Neighbouring chunks share 40 words, so a question about the
                  shared words can be answered by either chunk.

Each embedding model is run exactly as SentenceTransformer runs it in the app,
including its input limit (all-MiniLM-L6-v2 reads 256 tokens, bge-small 512).
The vector index in data/chroma_db is NOT changed; chunk texts are only read.

"Late" questions: questions whose answer matches the part of the chunk after its
first 150 words better than the beginning. If truncation hurts, the 256-token
model should do worst on these.

Paired comparison of hit@5 between two systems: exact McNemar test.

Usage (project venv, internet needed once to download the models):
    python scripts/eval_retrieval.py
Output:
    results/retrieval_eval/retrieval_eval.json      (summary)
    results/retrieval_eval/per_question_ranks.csv    (rank of the source chunk per question)
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np

DB = Path("data/chroma_db/chroma.sqlite3")
DATA = Path("data/finetune/train_v2.jsonl")
OUT_DIR = Path("results/retrieval_eval")
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# name -> (model id, query prefix)
SYSTEMS = {
    "minilm_l6_v2 (256 tok, current)": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge_small_v1.5 (512 tok)": ("BAAI/bge-small-en-v1.5", ""),
    "bge_small_v1.5 (512 tok) + query instruction": ("BAAI/bge-small-en-v1.5", BGE_INSTRUCTION),
}
TOP_K = 5
MRR_DEPTH = 10
LATE_WORDS = 150


def load_chunks(db: Path) -> List[Dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select id, key, string_value, int_value from embedding_metadata "
        "where key in ('chroma:document', 'source', 'chunk_id')").fetchall()
    by_id: Dict[int, Dict] = {}
    for rid, key, sval, ival in rows:
        by_id.setdefault(rid, {})[key] = sval if sval is not None else ival
    chunks = []
    for rid in sorted(by_id):
        d = by_id[rid]
        chunks.append({"key": f"{d['source']}#{d['chunk_id']}", "text": d["chroma:document"]})
    return chunks


def load_questions(path: Path) -> List[Dict]:
    out = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        key = r.get("chunk_key")
        if not key or key.startswith("external#"):   # no textbook chunk to find
            continue
        q = next(m["content"] for m in r["messages"] if m["role"] == "user")
        a = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        a = re.sub(r"\n+Source:.*$", "", a.strip())
        out.append({"q": q, "a": a, "key": key})
    return out


def late_flags(questions: List[Dict], chunk_text: Dict[str, str]) -> np.ndarray:
    """True if the answer overlaps more with the chunk's tail than with its head."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    heads, tails = [], []
    for q in questions:
        words = chunk_text[q["key"]].split()
        heads.append(" ".join(words[:LATE_WORDS]))
        tails.append(" ".join(words[LATE_WORDS:]) or "-")
    answers = [q["a"] for q in questions]
    vec = TfidfVectorizer(stop_words="english").fit(heads + tails + answers)
    A, H, T = vec.transform(answers), vec.transform(heads), vec.transform(tails)
    sim_h = np.asarray(A.multiply(H).sum(1)).ravel()
    sim_t = np.asarray(A.multiply(T).sum(1)).ravel()
    return sim_t > sim_h


def encode(model, texts: List[str]) -> np.ndarray:
    e = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def neighbours(key: str) -> set:
    src, idx = key.rsplit("#", 1)
    return {key, f"{src}#{int(idx) - 1}", f"{src}#{int(idx) + 1}"}


def ranks_for(model, prefix: str, chunks: List[Dict], questions: List[Dict]):
    C = encode(model, [c["text"] for c in chunks])
    Q = encode(model, [prefix + q["q"] for q in questions])
    S = Q @ C.T
    pos = {c["key"]: i for i, c in enumerate(chunks)}
    target = np.array([pos[q["key"]] for q in questions])
    tscore = S[np.arange(len(questions)), target]
    ranks = 1 + (S > tscore[:, None]).sum(1)          # 1 = best
    top = np.argsort(-S, axis=1)[:, :TOP_K]
    near = np.array([bool(neighbours(q["key"]) & {chunks[j]["key"] for j in top[i]})
                     for i, q in enumerate(questions)])
    return ranks, near


def summarise(r: np.ndarray, near: np.ndarray) -> Dict:
    if len(r) == 0:
        return {"n": 0}
    return {"n": int(len(r)),
            "hit@1": round(float(np.mean(r <= 1)), 4),
            f"hit@{TOP_K}": round(float(np.mean(r <= TOP_K)), 4),
            f"hit@{TOP_K}_or_neighbour": round(float(np.mean(near)), 4),
            f"mrr@{MRR_DEPTH}": round(float(np.mean(np.where(r <= MRR_DEPTH, 1.0 / r, 0.0))), 4)}


def mcnemar(a_hit: np.ndarray, b_hit: np.ndarray) -> Dict:
    from scipy.stats import binomtest
    b = int(np.sum(a_hit & ~b_hit)); c = int(np.sum(~a_hit & b_hit))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"only_first": b, "only_second": c, "p_exact": round(float(p), 6)}


def main(model_loader=None) -> Dict:
    if model_loader is None:
        from sentence_transformers import SentenceTransformer
        model_loader = SentenceTransformer
    chunks = load_chunks(DB)
    chunk_text = {c["key"]: c["text"] for c in chunks}
    questions = [q for q in load_questions(DATA) if q["key"] in chunk_text]
    late = late_flags(questions, chunk_text)
    print(f"{len(chunks)} chunks, {len(questions)} questions ({int(late.sum())} 'late')")

    cache, ranks, nears, info = {}, {}, {}, {}
    for name, (model_id, prefix) in SYSTEMS.items():
        if model_id not in cache:
            cache[model_id] = model_loader(model_id)
        m = cache[model_id]
        info[name] = {"model": model_id, "query_prefix": prefix,
                      "max_seq_length": int(getattr(m, "max_seq_length", 0) or 0)}
        ranks[name], nears[name] = ranks_for(m, prefix, chunks, questions)
        print(name, summarise(ranks[name], nears[name]))

    names = list(SYSTEMS)
    res = {
        "data": str(DATA), "chunks": len(chunks), "questions": len(questions),
        "late_questions": int(late.sum()), "late_rule": f"answer overlaps more with words after the first {LATE_WORDS}",
        "systems": {n: {**info[n], "all": summarise(ranks[n], nears[n]),
                        "late": summarise(ranks[n][late], nears[n][late]),
                        "early": summarise(ranks[n][~late], nears[n][~late])}
                    for n in names},
        "mcnemar_hit@5_vs_current": {n: mcnemar(ranks[names[0]] <= TOP_K, ranks[n] <= TOP_K) for n in names[1:]},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval_eval.json").write_text(json.dumps(res, indent=1))
    with (OUT_DIR / "per_question_ranks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_key", "late", *names, "question"])
        for i, q in enumerate(questions):
            w.writerow([q["key"], int(late[i]), *[int(ranks[n][i]) for n in names], q["q"]])
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
