import json
import glob
import psycopg2
from psycopg2.extras import execute_values
import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")

try:
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME, 
        user=DB_USER, password=DB_PASSWORD
    )
    print(f"Connected to {DB_NAME}@{DB_HOST}")
except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)

cursor = conn.cursor()

# Load all chunk files
chunk_files = sorted(glob.glob("data/chunks/*.jsonl"))
print(f"Found {len(chunk_files)} chunk files\n")

total_loaded = 0

for chunk_file in chunk_files:
    doc_name = chunk_file.split('/')[-1].replace('.jsonl', '')
    chunks_data = []
    
    for line in open(chunk_file, encoding="utf-8"):
        chunk = json.loads(line)
        chunks_data.append((
            chunk["doc_id"],
            chunk["chunk_id"],
            chunk["text"],
            chunk["approx_tokens"]
        ))
    
    print(f"{doc_name}: {len(chunks_data)} chunks")
    
    try:
        execute_values(
            cursor,
            "INSERT INTO chunks (doc_id, chunk_id, text, approx_tokens) VALUES %s ON CONFLICT (chunk_id) DO NOTHING",
            chunks_data,
            page_size=100
        )
        conn.commit()
        total_loaded += len(chunks_data)
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()

print(f"\n Total loaded: {total_loaded} chunks")

# Verify
cursor.execute("SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id ORDER BY doc_id;")
results = cursor.fetchall()
print("\n Chunks per insurer:")
for doc_id, count in results:
    print(f"   {doc_id}: {count}")

cursor.execute("SELECT COUNT(*) FROM chunks;")
total = cursor.fetchone()[0]
print(f"\n Total chunks in DB: {total}")

cursor.close()
conn.close()
