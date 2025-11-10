import psycopg2
import numpy as np
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

# Test queries
queries = [
    "What is the minimum age requirement for insuring a private car?",
    "How do I transfer my No Claim Bonus from another insurer?",
    "What is the own damage deductible amount?",
    "When is a vehicle declared as total loss?",
    "Are add-on covers available with third party only policy?"
]

print("="*80)
print("ICICI VECTOR SEARCH VALIDATION")
print("="*80 + "\n")

for qidx, query in enumerate(queries, 1):
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()
    
    cursor.execute("""
        SELECT 
            c.chunk_id, 
            c.text,
            c.approx_tokens,
            1 - (ce.embedding <=> %s::vector) as similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
        WHERE c.doc_id = 'ICICI_Lombard_Motor'
        ORDER BY similarity DESC
        LIMIT 5
    """, (q_emb,))
    
    results = cursor.fetchall()
    
    print(f"Query {qidx}: {query}")
    print("-" * 80)
    for rank, (chunk_id, text, tokens, sim) in enumerate(results, 1):
        print(f"  [{rank}] {chunk_id} | sim={sim:.3f} | tokens={tokens}")
        print(f"      {text[:150]}...\n")
    print()

cursor.close()
conn.close()
