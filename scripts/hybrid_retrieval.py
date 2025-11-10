import pickle
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from pathlib import Path

# Change to repo root
repo_root = Path(__file__).parent.parent
os.chdir(repo_root)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load BM25 index
print("Loading BM25 index...")
with open("data/bm25_index.pkl", "rb") as f:
    index_data = pickle.load(f)
bm25 = index_data["bm25"]
chunks = index_data["chunks"]
print(f"✅ Loaded {len(chunks)} chunks\n")

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Connect to DB
conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, database=DB_NAME,
    user=DB_USER, password=DB_PASSWORD
)
cursor = conn.cursor()

def hybrid_search(query, top_k=10, bm25_weight=0.4, vector_weight=0.6):
    """
    Hybrid retrieval: BM25 + vector search with weighted fusion
    """
    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)  # normalize
    
    # Vector search
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()
    cursor.execute("""
        SELECT 
            c.chunk_id,
            1 - (ce.embedding <=> %s::vector) as similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
    """, (q_emb,))
    vector_results = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Combine scores
    combined_scores = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        bm25_score = bm25_scores[i]
        vector_score = vector_results.get(chunk_id, 0.0)
        
        # Weighted fusion
        hybrid_score = (bm25_weight * bm25_score) + (vector_weight * vector_score)
        
        combined_scores.append({
            "chunk_id": chunk_id,
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "bm25_score": bm25_score,
            "vector_score": vector_score,
            "hybrid_score": hybrid_score
        })
    
    # Sort by hybrid score
    combined_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return combined_scores[:top_k]

# Test queries
queries = [
    "What is the No Claim Bonus for 2 consecutive claim-free years?",
    "What is the minimum age requirement for insuring a private car?",
    "Are add-on covers available with third party only policy?"
]

print("="*80)
print("HYBRID RETRIEVAL TEST (BM25 + Vector)")
print("="*80 + "\n")

for query in queries:
    print(f"Query: {query}")
    print("-" * 80)
    
    results = hybrid_search(query, top_k=5)
    
    for rank, result in enumerate(results, 1):
        print(f"[{rank}] {result['doc_id']} | hybrid={result['hybrid_score']:.3f} | bm25={result['bm25_score']:.3f} | vec={result['vector_score']:.3f}")
        print(f"    {result['text'][:150]}...\n")
    print()

cursor.close()
conn.close()
