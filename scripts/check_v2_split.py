"""Rebuild the corrected (v2) train/test split exactly as the Colab notebook does and
report its properties for the thesis.

Split: 90/10 over blocks of 10 consecutive chunks of the same book (seed 42).
Checks: sizes, books, whether it equals the saved split files, how many test chunks
share their 40-word overlap with a neighbouring training chunk, and whether any
question text occurs on both sides.

Usage:  python scripts/check_v2_split.py
Output: results/domain_expert_v2/split_check.json
"""
import json
import random
from collections import Counter
from pathlib import Path

DATA = Path("data/finetune/train_v2.jsonl")
RUN = Path("results/domain_expert_v2")
SEED, BLOCK = 42, 10

recs = [json.loads(l) for l in DATA.open(encoding="utf-8") if l.strip()]
block_of = lambda r: f"{r['source']}#{r['chunk_id'] // BLOCK}"
groups = sorted({block_of(r) for r in recs})
random.Random(SEED).shuffle(groups)
test_groups = set(groups[: int(0.10 * len(groups))])
train = [r for r in recs if block_of(r) not in test_groups]
test = [r for r in recs if block_of(r) in test_groups]

def neighbours(key):
    s, i = key.rsplit("#", 1)
    return {f"{s}#{int(i) - 1}", f"{s}#{int(i) + 1}"}

train_keys = {r["chunk_key"] for r in train}
test_keys = {r["chunk_key"] for r in test}
train_q = {r["messages"][1]["content"] for r in train}
saved_test = [json.loads(l) for l in (RUN / "test_split.jsonl").open(encoding="utf-8")]
res = {
    "records": len(recs), "chunks": len({r["chunk_key"] for r in recs}),
    "blocks": len(groups), "test_blocks": len(test_groups),
    "train": len(train), "test": len(test),
    "test_by_source": dict(Counter(r["source"] for r in test)),
    "test_chunks": len(test_keys),
    "test_chunks_sharing_overlap_with_a_training_chunk": sum(bool(neighbours(k) & train_keys) for k in test_keys),
    "chunks_on_both_sides": len(train_keys & test_keys),
    "test_questions_whose_text_also_in_training": sum(r["messages"][1]["content"] in train_q for r in test),
    "equals_saved_test_split": [r["messages"][1]["content"] for r in test] == [r["messages"][1]["content"] for r in saved_test],
    "avg_question_words": round(sum(len(r["messages"][1]["content"].split()) for r in recs) / len(recs), 1),
    "avg_answer_words": round(sum(len(r["messages"][2]["content"].split()) for r in recs) / len(recs), 1),
}
(RUN / "split_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
