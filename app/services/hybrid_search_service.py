import re
from app.services.qdrant_service import search_documents
from app.services.redis_service import get_bm25_index

def tokenize(text: str):
    return re.findall(r'\w+', text.lower())

def reciprocal_rank_fusion(dense_results: list, bm25_results: list, k: int = 60, top_k: int = 20):
    """Combine Dense and BM25 candidate lists using RRF."""
    rrf_scores = {}
    chunk_map = {}

    # Dense Stream
    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk"]
        chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # BM25 Stream
    for rank, item in enumerate(bm25_results, start=1):
        cid = item["chunk"]
        if cid not in chunk_map:
            chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # Sort candidates by fused score
    sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    
    results = []
    for cid in sorted_cids:
        item = chunk_map[cid].copy()
        item["score"] = rrf_scores[cid]
        results.append(item)
        
    return results

def search_hybrid(document_id: str, query: str, query_embedding: list[float], top_k: int = 20):
    # Stream A: Qdrant Dense Retrieval (Top 20)
    dense_candidates = search_documents(query_embedding=query_embedding, top_k=top_k, document_id=document_id)

    # Stream B: Redis-Cached BM25 Retrieval (Top 20)
    cached_data = get_bm25_index(document_id)
    bm25_candidates = []
    
    if cached_data:
        bm25_index = cached_data["bm25"]
        chunks = cached_data["chunks"]
        
        tokenized_query = tokenize(query)
        scores = bm25_index.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        bm25_candidates = [chunks[i] for i in top_indices]

    # RRF Fusion
    fused_candidates = reciprocal_rank_fusion(dense_candidates, bm25_candidates, k=60, top_k=top_k)
    return fused_candidates
