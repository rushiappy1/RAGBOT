import json
import psycopg2
from psycopg2.extras import execute_values
import os
from pathlib import Path

# DB config
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
    print(f"✅ Connected to {DB_NAME}@{DB_HOST}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

cursor = conn.cursor()

# Load ICICI chunks from JSONL
icici_path = Path("data/chunks/ICICI_Lombard_Motor.jsonl")
if not icici_path.exists():
    print(f"❌ File not found: {icici_path}")
    exit(1)

chunks_data = []
for line in open(icici_path, encoding="utf-8"):
    chunk = json.loads(line)
    chunks_data.append((
        chunk["doc_id"],
        chunk["chunk_id"],
        chunk["text"],
        chunk["approx_tokens"]
    ))

print(f"Loading {len(chunks_data)} ICICI chunks...")

try:
    execute_values(
        cursor,
        "INSERT INTO chunks (doc_id, chunk_id, text, approx_tokens) VALUES %s ON CONFLICT (chunk_id) DO NOTHING",
        chunks_data,
        page_size=100
    )
    conn.commit()
    print(f"✅ Inserted {len(chunks_data)} chunks into DB")
except Exception as e:
    print(f"❌ Insert failed: {e}")
    conn.rollback()
    exit(1)

# Verify
cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = 'ICICI_Lombard_Motor';")
count = cursor.fetchone()[0]
print(f"✅ Verified: {count} chunks in DB")

cursor.close()
conn.close()
