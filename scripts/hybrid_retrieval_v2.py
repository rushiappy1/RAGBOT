import pickle
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
from query_rewriter import rewrite_query
from reranker import CrossEncoderReranker

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
reranker = CrossEncoderReranker()

conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, database=DB_NAME,
    user=DB_USER, password=DB_PASSWORD
)
cursor = conn.cursor()

def hybrid_search_v2(query, top_k=10, bm25_weight=0.5, vector_weight=0.5, use_reranking=True):
    """
    Enhanced hybrid retrieval with query rewriting and reranking
    """
    # Step 1: Query rewriting
    rewritten_query = rewrite_query(query)
    print(f"Original: {query}")
    print(f"Rewritten: {rewritten_query}\n")
    
    # Step 2: BM25 search
    tokenized_query = rewritten_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    
    # Step 3: Vector search
    q_emb = model.encode([rewritten_query], normalize_embeddings=True)[0].tolist()
    cursor.execute("""
        SELECT c.chunk_id, 1 - (ce.embedding <=> %s::vector) as similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
    """, (q_emb,))
    vector_results = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Step 4: Hybrid fusion
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
    
    # Sort by hybrid score
    combined_scores.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = combined_scores[:20]  # Get top-20 for reranking
    
    # Step 5: Reranking (optional)
    if use_reranking:
        final_results = reranker.rerank(query, top_candidates, top_k=top_k)
    else:
        final_results = top_candidates[:top_k]
    
    return final_results

# Test on worst 3 queries
test_queries = [
    "Are add-on covers available with third party only policy?",
    "Can a vehicle registered as a private car but used for occasional paid carpooling be insured under a standard private car policy?",
    "If the insured dies after filing a claim but before settlement, can the legal heir claim the salvage value separately from the total loss payout?"
]

print("="*80)
print("HYBRID RETRIEVAL V2 - WITH REWRITING + RERANKING")
print("="*80 + "\n")

for query in test_queries:
    print("="*80)
    print(f"Query: {query}")
    print("="*80)
    
    results = hybrid_search_v2(query, top_k=5, bm25_weight=0.5, vector_weight=0.5)
    
    for rank, result in enumerate(results, 1):
        print(f"[{rank}] {result['doc_id']} | score={result.get('rerank_score', result['score']):.3f}")
        print(f"    {result['text'][:150]}...\n")
    print()

cursor.close()
conn.close()
