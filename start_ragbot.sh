#!/bin/bash

# Start llama-server (with GPU)
cd scripts/llama.cpp
./build/bin/llama-server -m /media/rishikesh/Rishi/RAGBOT/rag-banking/models/Qwen2.5-7B-Instruct-Q6_K.gguf -ngl 10 --port 8080 &
LLAMA_PID=$!

cd ../../

# Activate python env & run FastAPI backend
source .RAGBOT/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Run embedding watcher in background
python scripts/embedding_watcher.py &
EMBED_PID=$!

# Run auto-ingest pipeline in background
python scripts/auto_ingest.py &
INGEST_PID=$!

echo "RAGBot components started with PIDs:"
echo "llama-server: $LLAMA_PID"
echo "FastAPI: $API_PID"
echo "Embedding watcher: $EMBED_PID"
echo "Auto-ingest: $INGEST_PID"
