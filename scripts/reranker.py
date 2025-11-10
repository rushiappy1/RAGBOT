from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Load cross-encoder model for reranking"""
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, chunks, top_k=5):
        """
        Rerank chunks based on query-chunk relevance
        
        Args:
            query: User query string
            chunks: List of dicts with 'chunk_id', 'text', 'score' keys
            top_k: Number of top results to return
        
        Returns:
            Reranked list of chunks
        """
        if not chunks:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add rerank scores to chunks
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]
