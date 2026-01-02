import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import os
import torch

# --- CONFIGURATION ---
# 1. Update to the "Finance Grade" Model
MODEL_NAME = "BAAI/bge-m3" 
BATCH_SIZE = 64  # Process 64 chunks at a time (Safe for 16GB VRAM)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )

def main():
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Running on: {device.upper()}")


    print(f" Loading Model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
    
    if device == "cuda":
        model.half()
        print("Model loaded in FP16 mode")

    conn = get_db_connection()
    # Use a named cursor for server-side streaming (Zero RAM bloat)
    cursor = conn.cursor(name="chunk_stream_cursor") 

    try:
      
        print(" Fetching chunks needing embeddings...")
        cursor.execute("""
            SELECT c.id, c.chunk_id, c.text 
            FROM chunks c
            LEFT JOIN chunk_embeddings ce ON c.chunk_id = ce.chunk_id
            WHERE ce.id IS NULL
        """)

        total_processed = 0
        
        while True:
           
            rows = cursor.fetchmany(BATCH_SIZE)
            if not rows:
                break

            # Unpack batch
            # chunk_uuids -> For inserting
            # texts -> For embedding
            chunk_uuids = [row[1] for row in rows]
            texts = [row[2] for row in rows]

            #  Generate Embeddings
            # normalize_embeddings=True is CRITICAL for Cosine Similarity
            embeddings = model.encode(
                texts, 
                batch_size=BATCH_SIZE, 
                normalize_embeddings=True, 
                show_progress_bar=False,
                convert_to_numpy=True
            )

            #  Prepare Insert Data
            insert_data = [
                (uuid, emb.tolist()) 
                for uuid, emb in zip(chunk_uuids, embeddings)
            ]

            #  Insert & Commit immediately (Save progress!)
            # We use a separate cursor for writes because the read cursor is active
            with conn.cursor() as write_cursor:
                execute_values(
                    write_cursor,
                    "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES %s ON CONFLICT (chunk_id) DO NOTHING",
                    insert_data
                )
                conn.commit() # <--- Saves data after every batch

            total_processed += len(rows)
            print(f" Processed {total_processed} chunks...", end="\r")

    except Exception as e:
        print(f"\n CRITICAL ERROR: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
        print(f"\n  Job Complete. Total Embedded: {total_processed}")

if __name__ == "__main__":
    main()
