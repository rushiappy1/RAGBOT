CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_docs (
    id SERIAL PRIMARY KEY,
    insurer TEXT NOT NULL,
    product TEXT NOT NULL,
    policy_name TEXT NOT NULL,
    doc_md TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS policy_chunks (
    id SERIAL PRIMARY KEY,
    policy_id INT REFERENCES policy_docs(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1536), -- adjust later based on model used
    section TEXT,
    clause_ref TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON policy_chunks USING ivfflat (embedding vector_cosine_ops);

