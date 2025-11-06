-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_id VARCHAR(100) NOT NULL,
    chunk_id VARCHAR(200) UNIQUE NOT NULL,
    text TEXT NOT NULL,
    approx_tokens INT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id);

-- Embeddings table
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id BIGSERIAL PRIMARY KEY,
    chunk_id VARCHAR(200) UNIQUE NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding vector(384),
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunk_embedding_cosine ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops);
