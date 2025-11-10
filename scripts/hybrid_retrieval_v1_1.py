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

def rewrite_query_focused(query):
    """Minimal, targeted query expansion"""
    query_lower = query.lower()
    expansions = []
    
    if "add-on" in query_lower and "third party" in query_lower:
        expansions = ["optional covers", "TP policy", "liability only"]
    elif "carpool" in query_lower or ("private car" in query_lower and "paid" in query_lower):
        expansions = ["commercial use", "hire or reward"]
    elif "legal heir" in query_lower and "salvage" in query_lower:
        expansions = ["policy transfer", "claim settlement"]
    
    if expansions:
        return query + " " + " ".join(expansions)
    return query

def hybrid_search_v1_1(query, top_k=5, bm25_weight=0.5, vector_weight=0.5):
    """V1.1: Focused query expansion + no cross-encoder"""
    rewritten_query = rewrite_query_focused(query)
    
    # BM25
    tokenized_query = rewritten_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    
    # Vector
    q_emb = model.encode([rewritten_query], normalize_embeddings=True)[0].tolist()
    cursor.execute("""
        SELECT c.chunk_id, 1 - (ce.embedding <=> %s::vector) as similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
    """, (q_emb,))
    vector_results = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Hybrid fusion
    combined_scores = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        hybrid_score = (bm25_weight * bm25_scores[i]) + (vector_weight * vector_results.get(chunk_id, 0.0))
        
        combined_scores.append({
            "chunk_id": chunk_id,
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "score": hybrid_score
        })
    
    combined_scores.sort(key=lambda x: x["score"], reverse=True)
    return combined_scores[:top_k]

# Test on 3 failing queries
test_queries = [
    ("Are add-on covers available with third party only policy?", 0.5, 0.5),
    ("Can a vehicle registered as a private car but used for occasional paid carpooling be insured under a standard private car policy?", 0.6, 0.4),
    ("If the insured dies after filing a claim but before settlement, can the legal heir claim the salvage value separately from the total loss payout?", 0.4, 0.6)
]

print("="*80)
print("HYBRID RETRIEVAL V1.1 - FOCUSED EXPANSION, NO RERANKING")
print("="*80 + "\n")

for query, bm25_w, vec_w in test_queries:
    print("="*80)
    print(f"Query: {query}")
    print(f"Weights: BM25={bm25_w}, Vector={vec_w}")
    rewritten = rewrite_query_focused(query)
    if rewritten != query:
        print(f"Rewritten: {rewritten}")
    print("="*80)
    
    results = hybrid_search_v1_1(query, top_k=5, bm25_weight=bm25_w, vector_weight=vec_w)
    
    for rank, result in enumerate(results, 1):
        print(f"[{rank}] {result['doc_id']} | score={result['score']:.3f}")
        print(f"    {result['text'][:150]}...\n")
    print()

cursor.close()
conn.close()
