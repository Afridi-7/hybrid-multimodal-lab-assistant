"""Count word-piece tokens per knowledge-base chunk (thesis Section 3.1).

Reads the 1,049 chunks straight from the ChromaDB SQLite file and tokenises them
with the all-MiniLM-L6-v2 tokenizer, whose input limit is 256 tokens. BAAI/bge-small-en-v1.5
uses the same (bert-base-uncased) vocabulary and reads 512 tokens, so the counts for 384 and
512 tell how many chunks a longer-input embedder would still truncate.
Usage (project venv):  python scripts/check_chunk_tokens.py
Output: results/kb_chunk_token_stats.json
"""
import json, sqlite3
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

DB = Path("data/chroma_db/chroma.sqlite3")
OUT = Path("results/kb_chunk_token_stats.json")
LIMIT = 256

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
docs = [r[0] for r in con.execute(
    "select string_value from embedding_metadata where key='chroma:document' order by id")]
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
lens = np.array([len(tok(d)["input_ids"]) for d in docs])   # includes [CLS] and [SEP]
over = lens > LIMIT
stats = {
    "chunks": len(docs),
    "tokens_mean": round(float(lens.mean()), 1),
    "tokens_median": float(np.median(lens)),
    "tokens_min": int(lens.min()),
    "tokens_max": int(lens.max()),
    "over_256": int(over.sum()),
    "over_256_pct": round(100 * float(over.mean()), 1),
    "share_of_all_tokens_cut_pct": round(100 * float(np.clip(lens - LIMIT, 0, None).sum() / lens.sum()), 1),
    "over_384": int((lens > 384).sum()),
    "over_512": int((lens > 512).sum()),
}
OUT.write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
