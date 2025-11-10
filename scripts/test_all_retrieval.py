import psycopg2
from sentence_transformers import SentenceTransformer
import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, database=DB_NAME,
    user=DB_USER, password=DB_PASSWORD
)
cursor = conn.cursor()

model = SentenceTransformer(MODEL_NAME)

# Test query
query = "What is the No Claim Bonus for 2 consecutive claim-free years?"
q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

print("="*80)
print("CROSS-INSURER RETRIEVAL TEST")
print("="*80)
print(f"\nQuery: {query}\n")
print("-" * 80)

cursor.execute("""
    SELECT 
        c.doc_id,
        c.chunk_id, 
        LEFT(c.text, 200) as preview,
        c.approx_tokens,
        1 - (ce.embedding <=> %s::vector) as similarity
    FROM chunk_embeddings ce
    JOIN chunks c ON c.chunk_id = ce.chunk_id
    ORDER BY similarity DESC
    LIMIT 10
""", (q_emb,))

results = cursor.fetchall()

for rank, (doc_id, chunk_id, preview, tokens, sim) in enumerate(results, 1):
    print(f"[{rank}] {doc_id} | sim={sim:.3f} | tokens={tokens}")
    print(f"    {preview}...\n")

cursor.close()
conn.close()
