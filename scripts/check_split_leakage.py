"""Check whether the original 90/10 split of the Q&A data was really chunk-level.

The Colab notebook assumed that records 2i and 2i+1 of data/finetune/train.jsonl
come from the same textbook chunk. scripts/build_finetune_dataset.py, however,
collects results in completion order and then calls random.shuffle(all_records),
so that assumption does not hold. This script measures it on the data:

1. Each Q&A record is matched to its most similar chunk (TF-IDF, question + answer
   against the 1,049 chunk texts stored in ChromaDB).
2. It reports how often records 2i and 2i+1 map to the same chunk (vs. random pairs).
3. It rebuilds the notebook's split (seed 42), checks it equals the saved
   results/domain_expert/test_split.jsonl, and counts test records whose source
   chunk also produced a training record (leakage).

The chunk matching is approximate, so the counts are estimates.
Usage:  python scripts/check_split_leakage.py
Output: results/domain_expert/old_split_leakage_check.json
"""
import json, random, re, sqlite3
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DB = Path("data/chroma_db/chroma.sqlite3")
DATA = Path("data/finetune/train.jsonl")
SAVED_TEST = Path("results/domain_expert/test_split.jsonl")
OUT = Path("results/domain_expert/old_split_leakage_check.json")
SEED = 42

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
chunks = [r[0] for r in con.execute(
    "select string_value from embedding_metadata where key='chroma:document' order by id")]
recs = [json.loads(l) for l in DATA.open(encoding="utf-8") if l.strip()]

def qa_text(r):
    q = r["messages"][1]["content"]
    a = re.sub(r"\n+Source:.*$", "", r["messages"][2]["content"].strip())
    return q + " " + a

texts = [qa_text(r) for r in recs]
vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english").fit(chunks + texts)
S = linear_kernel(vec.transform(texts), vec.transform(chunks))
best = S.argmax(1)

n_pairs = len(recs) // 2
same_pair = np.mean([best[2 * i] == best[2 * i + 1] for i in range(n_pairs)])
perm = np.random.default_rng(0).permutation(len(recs))
same_rand = np.mean([best[perm[2 * i]] == best[perm[2 * i + 1]] for i in range(n_pairs)])

groups = sorted({i // 2 for i in range(len(recs))})
random.Random(SEED).shuffle(groups)
test_groups = set(groups[: int(0.10 * len(groups))])
test_idx = [i for i in range(len(recs)) if i // 2 in test_groups]
train_idx = [i for i in range(len(recs)) if i // 2 not in test_groups]
saved = [json.loads(l)["messages"][1]["content"] for l in SAVED_TEST.open(encoding="utf-8")]
matches_saved = [recs[i]["messages"][1]["content"] for i in test_idx] == saved

train_chunks = set(best[train_idx])
leak = [i for i in test_idx if best[i] in train_chunks]
leak100 = [i for i in test_idx[:100] if best[i] in train_chunks]

res = {
    "records": len(recs), "chunks": len(chunks),
    "pairs_2i_2i1_same_chunk_pct": round(100 * float(same_pair), 1),
    "random_pairing_baseline_pct": round(100 * float(same_rand), 1),
    "reconstructed_split_matches_saved_test_split": bool(matches_saved),
    "test_records": len(test_idx),
    "test_records_with_source_chunk_in_training": len(leak),
    "first100_test_records_with_source_chunk_in_training": len(leak100),
    "note": "Chunk matching by TF-IDF is approximate; counts are estimates.",
}
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
