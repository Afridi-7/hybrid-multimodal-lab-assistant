"""Build a supervised fine-tuning dataset for the domain-expert hematology LLM.

Pipeline:
  1. Pull every chunk from the existing Chroma collection (1k+ textbook passages).
  2. For each chunk, ask GPT-4o to generate two grounded (question, answer) pairs.
  3. Optionally append MedQA-USMLE hematology items.
  4. Optionally append a hand-curated gold file (data/finetune/gold.jsonl).
  5. Write a single JSONL file in OpenAI/HF chat format. Every record also stores the
     chunk it was generated from, so the train/test split can be made by chunk:
        {"messages": [{"role": "system", ...},
                      {"role": "user", ...},
                      {"role": "assistant", ...}],
         "chunk_key": "essentials_haematology.pdf#123", "source": "...",
         "chunk_id": 123, "pair_index": 0}

  Note (Nov 2026): the first dataset (data/finetune/train.jsonl) did not store the
  chunk, and records were collected in completion order and shuffled, so records
  2i and 2i+1 were NOT from the same chunk. See scripts/check_split_leakage.py.
  Since then the default output is data/finetune/train_v2.jsonl, records carry
  chunk_key, and MedQA augmentation is off by default (it was not used before
  either: the first file contained only the 2 x 1,049 chunk pairs).

Run:
  python scripts/build_finetune_dataset.py \
      --chroma-dir data/chroma_db \
      --out data/finetune/train_v2.jsonl \
      --pairs-per-chunk 2 \
      --max-chunks 0          # 0 = no cap, otherwise sample N chunks for cheap dry runs

Cost (GPT-4o-mini default): ~$2 for the full 1 049 chunks @ 2 pairs each.
Set ``--model gpt-4o`` for higher quality (~$8–10).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Force UTF-8 stdout on Windows so progress arrows / unicode prints don't crash.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Avoid torch/transformers imports — keep this script light.
SYSTEM_PROMPT = (
    "You are a clinical hematology expert. Always ground answers in the "
    "provided textbook reference. Use cautious, evidence-based language. "
    "When evidence is insufficient, say so explicitly. Cite using "
    "[Reference N] when paraphrasing the source."
)

QA_GENERATION_INSTRUCTION = """You will receive ONE textbook passage about hematology.
Produce {n} *grounded* training examples in JSON. Each example must have:
  - "question": a realistic clinician/student question that this passage can answer.
  - "answer":   a 2–5 sentence answer grounded ONLY in the passage. Hedge when uncertain.
Vary the question style: definitions, mechanisms, differentials, lab interpretation,
"what would you expect on a smear if...", patient-vignette style.
Return STRICT JSON: {{"pairs": [{{"question": "...", "answer": "..."}}, ...]}}
Do not invent facts beyond the passage.

PASSAGE:
\"\"\"
{passage}
\"\"\""""


# ---------------------------------------------------------------------------
# Chroma extraction
# ---------------------------------------------------------------------------

def load_chunks_from_chroma(chroma_dir: Path, collection: str) -> List[Dict[str, Any]]:
    """Return [{id, text, metadata}, ...] for every chunk in the collection."""
    import chromadb  # local import: heavy

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        col = client.get_collection(collection)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Could not open Chroma collection '{collection}' at {chroma_dir}: {exc}"
        ) from exc

    # Pull everything in pages (Chroma get() supports include + limit/offset).
    all_chunks: List[Dict[str, Any]] = []
    page = 1000
    offset = 0
    while True:
        batch = col.get(limit=page, offset=offset, include=["documents", "metadatas"])
        ids = batch.get("ids") or []
        if not ids:
            break
        for cid, doc, meta in zip(ids, batch["documents"], batch["metadatas"] or []):
            if doc and len(doc.strip()) >= 80:  # skip near-empty chunks
                meta = meta or {}
                src = meta.get("source") or meta.get("file") or "textbook"
                # Stable identity of a chunk: source file + per-file chunk index.
                # (Chroma's own ids depend on build order; this does not.)
                key = f"{src}#{meta.get('chunk_id', cid)}"
                all_chunks.append({"id": cid, "key": key, "text": doc, "metadata": meta})
        if len(ids) < page:
            break
        offset += page
    return all_chunks


# ---------------------------------------------------------------------------
# Synthetic Q&A generation
# ---------------------------------------------------------------------------

def generate_pairs_for_chunk(
    client: Any,
    model: str,
    passage: str,
    n_pairs: int,
    max_retries: int = 3,
) -> List[Dict[str, str]]:
    prompt = QA_GENERATION_INSTRUCTION.format(n=n_pairs, passage=passage[:3500])
    delay = 1.5
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            pairs = data.get("pairs") or []
            cleaned = [
                {"question": p["question"].strip(), "answer": p["answer"].strip()}
                for p in pairs
                if isinstance(p, dict) and p.get("question") and p.get("answer")
            ]
            return cleaned[:n_pairs]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(delay)
            delay *= 2
    print(f"    [warn] giving up on chunk: {last_err}", file=sys.stderr)
    return []


def to_chat_record(question: str, answer: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# ---------------------------------------------------------------------------
# Optional extras
# ---------------------------------------------------------------------------

def maybe_load_medqa_hematology(limit: int) -> List[Dict[str, Any]]:
    """Try to pull a MedQA hematology subset; skip cleanly if datasets isn't available."""
    if limit <= 0:
        return []
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        print("[info] datasets not installed — skipping MedQA augment", file=sys.stderr)
        return []

    keywords = (
        "anemia", "leukemia", "lymphoma", "hemoglobin", "platelet", "neutrophil",
        "lymphocyte", "monocyte", "eosinophil", "basophil", "blood smear",
        "thrombocyt", "myeloid", "erythrocyt",
    )

    try:
        ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split="train")
    except Exception as exc:  # noqa: BLE001
        print(f"[info] MedQA load failed ({exc}) — skipping", file=sys.stderr)
        return []

    out: List[Dict[str, Any]] = []
    for ex in ds:
        q = (ex.get("question") or "").lower()
        if not any(k in q for k in keywords):
            continue
        opts = ex.get("options") or []
        answer_idx = ex.get("answer_idx")
        answer_txt = ex.get("answer") or ""
        if not answer_txt:
            continue
        opts_txt = "\n".join(f"  {o.get('key')}. {o.get('value')}" for o in opts)
        prompt = f"{ex['question']}\n\nOptions:\n{opts_txt}".strip()
        rationale = (
            f"The correct answer is **{answer_idx}. {answer_txt}**. "
            "Reasoning is grounded in standard hematology references; consult "
            "current guidelines and expert review for clinical decisions."
        )
        out.append(to_chat_record(prompt, rationale))
        if len(out) >= limit:
            break
    return out


def load_gold(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Accept either {messages:[…]} or {question, answer}
        if "messages" in obj:
            out.append(obj)
        elif obj.get("question") and obj.get("answer"):
            out.append(to_chat_record(obj["question"], obj["answer"]))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-dir", default="data/chroma_db", type=Path)
    parser.add_argument("--collection", default="hematology_knowledge")
    parser.add_argument("--out", default="data/finetune/train_v2.jsonl", type=Path)
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model (gpt-4o-mini ≈ $2 / gpt-4o ≈ $10 for full corpus).")
    parser.add_argument("--pairs-per-chunk", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Cap number of chunks (0 = use all). Useful for dry runs.")
    parser.add_argument("--medqa-limit", type=int, default=0,
                        help="Append N MedQA items (default 0 = off; they carry no chunk).")
    parser.add_argument("--gold", default="data/finetune/gold.jsonl", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=20,
                        help="Concurrent OpenAI requests (gpt-4o-mini handles 20+ easily).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip OpenAI calls; only print chunk count + emit MedQA/gold.")
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Read OPENAI_API_KEY from the repo's .env (same as scripts/judge_domain_expert.py).
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    print(f"[1/4] Loading Chroma chunks from {args.chroma_dir} …")
    chunks = load_chunks_from_chroma(args.chroma_dir, args.collection)
    print(f"      → {len(chunks)} chunks loaded")

    if args.max_chunks and args.max_chunks < len(chunks):
        chunks = random.sample(chunks, args.max_chunks)
        print(f"      → sampled down to {len(chunks)} for this run")

    records: List[Dict[str, Any]] = []

    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set in environment.")
        from openai import OpenAI  # type: ignore
        client = OpenAI()

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _process(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
            pairs = generate_pairs_for_chunk(
                client, args.model, chunk["text"], args.pairs_per_chunk
            )
            src = chunk["metadata"].get("source") or chunk["metadata"].get("file") or "textbook"
            local: List[Dict[str, Any]] = []
            for j, p in enumerate(pairs):
                grounded_answer = p["answer"].rstrip()
                if "[Reference" not in grounded_answer:
                    grounded_answer += f"\n\nSource: {src}."
                rec = to_chat_record(p["question"], grounded_answer)
                rec["chunk_key"] = chunk["key"]
                rec["source"] = src
                rec["chunk_id"] = chunk["metadata"].get("chunk_id")
                rec["pair_index"] = j
                local.append(rec)
            return local

        print(f"[2/4] Generating {args.pairs_per_chunk} pairs per chunk via {args.model} "
              f"with {args.workers} parallel workers …")
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process, c) for c in chunks]
            for fut in as_completed(futures):
                try:
                    records.extend(fut.result())
                except Exception as exc:  # noqa: BLE001
                    print(f"      [warn] chunk failed: {exc}", file=sys.stderr)
                done += 1
                if done % 25 == 0 or done == len(chunks):
                    print(f"      {done}/{len(chunks)} chunks → {len(records)} pairs so far")
        # Results arrive in completion order; sort them back into chunk order so the
        # file is reproducible apart from the generator's own randomness.
        order = {c["key"]: i for i, c in enumerate(chunks)}
        records.sort(key=lambda r: (order[r["chunk_key"]], r["pair_index"]))
        per_chunk: Dict[str, int] = {}
        for r in records:
            per_chunk[r["chunk_key"]] = per_chunk.get(r["chunk_key"], 0) + 1
        short = [k for k in order if per_chunk.get(k, 0) < args.pairs_per_chunk]
        print(f"      chunks with fewer than {args.pairs_per_chunk} pairs: {len(short)}")
    else:
        print("[2/4] --dry-run set; skipping OpenAI calls")

    print(f"[3/4] Augmenting with MedQA (limit={args.medqa_limit}) …")
    medqa = maybe_load_medqa_hematology(args.medqa_limit)
    print(f"      → {len(medqa)} MedQA items")

    gold = load_gold(args.gold)
    print(f"      → {len(gold)} hand-curated gold items from {args.gold}")

    # External items have no textbook chunk; give each its own key so a
    # chunk-level split still treats them as independent groups.
    for i, r in enumerate(medqa + gold):
        r.setdefault("chunk_key", f"external#{i}")
    all_records = records + medqa + gold
    # Shuffling is harmless now that every record stores its chunk_key:
    # the split must group by chunk_key, never by position in the file.
    random.shuffle(all_records)
    print(f"[4/4] Writing {len(all_records)} records → {args.out}")
    with args.out.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
