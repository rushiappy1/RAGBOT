import json
import glob
import pickle
from rank_bm25 import BM25Okapi
from pathlib import Path
import os

# Change to repo root
repo_root = Path(__file__).parent.parent
os.chdir(repo_root)

# Load all chunks
all_chunks = []
for jsonl_path in sorted(glob.glob("data/chunks/*.jsonl")):
    print(f"Loading {jsonl_path}...")
    for line in open(jsonl_path, encoding="utf-8"):
        chunk = json.loads(line)
        all_chunks.append(chunk)

print(f"Loaded {len(all_chunks)} chunks")

# Build BM25 index
corpus = [chunk["text"] for chunk in all_chunks]
tokenized_corpus = [doc.lower().split() for doc in corpus]

print("Building BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)

# Save index + chunk metadata
index_data = {
    "bm25": bm25,
    "chunks": all_chunks
}

output_path = Path("data/bm25_index.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump(index_data, f)

print(f"✅ BM25 index saved to {output_path}")
print(f"   Corpus size: {len(all_chunks)} chunks")
