import psycopg2
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values
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

# Get all chunks without embeddings
cursor.execute("""
    SELECT c.id, c.chunk_id, c.text, c.doc_id
    FROM chunks c
    LEFT JOIN chunk_embeddings ce ON c.chunk_id = ce.chunk_id
    WHERE ce.id IS NULL
    ORDER BY c.id
""")
rows = cursor.fetchall()
print(f"Found {len(rows)} chunks to embed\n")

if len(rows) == 0:
    print("✅ All chunks already embedded")
    cursor.close()
    conn.close()
    exit(0)

# Load model
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Embed in batches
texts = [row[2] for row in rows]
print("Encoding embeddings...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=32)

# Prepare insert data
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

# Verify by doc_id
cursor.execute("""
    SELECT c.doc_id, COUNT(ce.id)
    FROM chunks c
    JOIN chunk_embeddings ce ON c.chunk_id = ce.chunk_id
    GROUP BY c.doc_id
    ORDER BY c.doc_id
""")
results = cursor.fetchall()
print("\n📊 Embeddings per insurer:")
for doc_id, count in results:
    print(f"   {doc_id}: {count}")

cursor.execute("SELECT COUNT(*) FROM chunk_embeddings;")
total = cursor.fetchone()[0]
print(f"\n✅ Total embeddings in DB: {total}")

cursor.close()
conn.close()
