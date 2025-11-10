from sentence_transformers import CrossEncoder
import numpy as np

class ChunkReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Load cross-encoder for reranking"""
        print(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, chunks, top_k=5):
        """Rerank chunks using cross-encoder scores"""
        # Prepare pairs
        pairs = [(query, chunk["text"][:512]) for chunk in chunks]  # Truncate to 512 chars
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by scores
        ranked_indices = np.argsort(scores)[::-1]
        
        # Return top_k
        reranked = []
        for idx in ranked_indices[:top_k]:
            chunk_copy = chunks[idx].copy()
            chunk_copy["rerank_score"] = float(scores[idx])
            reranked.append(chunk_copy)
        
        return reranked
