import json
import pickle
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

repo_root = Path(__file__).parent.parent
os.chdir(repo_root)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load BM25 index
with open("data/bm25_index.pkl", "rb") as f:
    index_data = pickle.load(f)
bm25 = index_data["bm25"]
chunks = index_data["chunks"]

model = SentenceTransformer(MODEL_NAME)

conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, database=DB_NAME,
    user=DB_USER, password=DB_PASSWORD
)
cursor = conn.cursor()

def hybrid_search(query, top_k=10, bm25_weight=0.4, vector_weight=0.6):
    """Hybrid retrieval"""
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()
    cursor.execute("""
        SELECT c.chunk_id, 1 - (ce.embedding <=> %s::vector) as similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
    """, (q_emb,))
    vector_results = {row[0]: row[1] for row in cursor.fetchall()}
    
    combined_scores = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        hybrid_score = (bm25_weight * bm25_scores[i]) + (vector_weight * vector_results.get(chunk_id, 0.0))
        combined_scores.append({"chunk_id": chunk_id, "score": hybrid_score, "text": chunk["text"]})
    
    combined_scores.sort(key=lambda x: x["score"], reverse=True)
    return combined_scores[:top_k]

# Load eval queries
with open("data/eval_queries_v0.json") as f:
    eval_data = json.load(f)

print("="*80)
print("RETRIEVAL EVALUATION - EVAL SET V0")
print("="*80 + "\n")

results = []

for q in eval_data["queries"]:
    query_id = q["id"]
    query_text = q["query"]
    bucket = q["bucket"]
    
    print(f"{query_id} [{bucket}]: {query_text[:80]}...")
    
    retrieved = hybrid_search(query_text, top_k=5)
    
    print("  Top-5 chunks:")
    for rank, result in enumerate(retrieved, 1):
        print(f"    [{rank}] {result['chunk_id']} | score={result['score']:.3f}")
        print(f"        {result['text'][:120]}...")
    print()
    
    results.append({
        "query_id": query_id,
        "query": query_text,
        "bucket": bucket,
        "retrieved_chunks": [r["chunk_id"] for r in retrieved]
    })

# Save results for ground truth annotation
with open("data/eval_results_v0.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation results saved to data/eval_results_v0.json")
print("Next: Manually annotate ground truth chunk IDs in eval_queries_v0.json")

cursor.close()
conn.close()
