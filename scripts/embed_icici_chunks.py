import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from psycopg2.extras import execute_values

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

# Get all ICICI chunks without embeddings
cursor.execute("""
    SELECT c.id, c.chunk_id, c.text 
    FROM chunks c
    LEFT JOIN chunk_embeddings ce ON c.chunk_id = ce.chunk_id
    WHERE c.doc_id = 'ICICI_Lombard_Motor' AND ce.id IS NULL
    ORDER BY c.id
""")
rows = cursor.fetchall()
print(f"Found {len(rows)} chunks to embed")

if len(rows) > 0:
    # Extract texts
    texts = [row[2] for row in rows]
    
    # Embed in batch
    print("Encoding embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # Prepare data for insert
    embed_data = [
        (row[1], embeddings[i].tolist())
        for i, row in enumerate(rows)
    ]
    
    # Insert embeddings
    print("Inserting embeddings into DB...")
    execute_values(
        cursor,
        "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES %s ON CONFLICT DO NOTHING",
        embed_data,
        page_size=100
    )
    conn.commit()
    print(f"✅ Inserted {len(embed_data)} embeddings")

# Verify
cursor.execute("SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id LIKE 'ICICI_Lombard_Motor%';")
count = cursor.fetchone()[0]
print(f"✅ Verified: {count} embeddings in DB")

cursor.close()
conn.close()
