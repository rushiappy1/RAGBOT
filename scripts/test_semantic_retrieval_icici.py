# scripts/test_semantic_retrieval_icici.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load ICICI chunks
chunks = [json.loads(line) for line in open("data/chunks/ICICI_Lombard_Motor_test.jsonl", encoding="utf-8")]
texts = [c['text'] for c in chunks]

# Embed and build FAISS index
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding ICICI chunks...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
index = faiss.IndexFlatIP(384)
index.add(embeddings.astype('float32'))

# 5 hard claim prompts
queries = [
    "What is the minimum age requirement for insuring a private car?",
    "How do I transfer my No Claim Bonus from another insurer?",
    "Are add-on covers available with third party only policy?",
    "What is the own damage deductible amount for this policy?",
    "When is a vehicle declared as constructive total loss or salvage?"
]

print("\n" + "="*80)
for i, query in enumerate(queries, 1):
    print(f"\nQuery {i}: {query}")
    print("-" * 80)
    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    D, I = index.search(q_emb, 3)
    
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        print(f"\n[Rank {rank}] Score: {score:.3f} | Chunk: {chunks[idx]['chunk_id']}")
        print(f"Tokens: {chunks[idx]['approx_tokens']}")
        print(f"Text preview:\n{chunks[idx]['text'][:400]}...\n")
