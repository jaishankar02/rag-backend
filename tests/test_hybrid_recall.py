import json
import os
import re
from rank_bm25 import BM25Okapi

# Paths
BASE_DIR = "/home/jaishankar/Documents/resume_projects/DocRag"
RESULTS_FILE = os.path.join(BASE_DIR, "backend/tests/evaluation_reranking_results.json")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.txt")

def simple_tokenize(text):
    return re.findall(r'\w+', text.lower())

# 1. Load ground truth questions & existing dense retrieval results
with open(RESULTS_FILE, "r") as f:
    eval_data = json.load(f)

# 2. Load all chunks for BM25 indexing
all_chunks = []
if os.path.exists(CHUNKS_FILE):
    with open(CHUNKS_FILE, "r") as f:
        chunks_raw = f.read().split("--------------------------------------------------------------------------------")
        for idx, c in enumerate(chunks_raw):
            c_str = c.strip()
            if c_str:
                all_chunks.append({"chunk": idx + 1, "text": c_str})

# Extract corpus mapping from results if chunks.txt not sufficient
chunk_text_map = {}
for q in eval_data["results"]:
    for source in q.get("retrieved_results", []):
        chunk_text_map[source["chunk"]] = source["text"]
    for source in q.get("ground_truth_chunks", []):
        chunk_text_map[source["chunk"]] = source["text"]

corpus_chunks = []
if all_chunks:
    corpus_chunks = all_chunks
else:
    corpus_chunks = [{"chunk": cid, "text": txt} for cid, txt in chunk_text_map.items()]

tokenized_corpus = [simple_tokenize(doc["text"]) for doc in corpus_chunks]
bm25 = BM25Okapi(tokenized_corpus)

dense_hits_20 = 0
bm25_hits_20 = 0
hybrid_hits_20 = 0
total_questions = len(eval_data["results"])

dense_chunk_recalls = []
bm25_chunk_recalls = []
hybrid_chunk_recalls = []

print(f"=== EVALUATING HYBRID RECALL ON {total_questions} QUESTIONS ===")

for q in eval_data["results"]:
    q_id = q["question_id"]
    query = q["question"]
    gt = set(q["ground_truth_chunk_ids"])
    
    # Dense vector top 20
    dense_top20 = [r["chunk"] for r in q["retrieved_results"][:20]]
    
    # BM25 top 20
    tokenized_query = simple_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    bm25_top20 = [corpus_chunks[i]["chunk"] for i in top_bm25_indices]
    
    # Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    k = 60
    for rank, cid in enumerate(dense_top20, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
    for rank, cid in enumerate(bm25_top20, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
        
    hybrid_top20 = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:20]
    
    # Evaluate Hit@20
    d_hit = 1 if gt.intersection(set(dense_top20)) else 0
    b_hit = 1 if gt.intersection(set(bm25_top20)) else 0
    h_hit = 1 if gt.intersection(set(hybrid_top20)) else 0
    
    dense_hits_20 += d_hit
    bm25_hits_20 += b_hit
    hybrid_hits_20 += h_hit
    
    # Chunk recall
    d_rec = len(gt.intersection(set(dense_top20))) / len(gt)
    b_rec = len(gt.intersection(set(bm25_top20))) / len(gt)
    h_rec = len(gt.intersection(set(hybrid_top20))) / len(gt)
    
    dense_chunk_recalls.append(d_rec)
    bm25_chunk_recalls.append(b_rec)
    hybrid_chunk_recalls.append(h_rec)
    
    print(f"Q{q_id:02d} GT:{list(gt)} | DenseHit:{d_hit} BM25Hit:{b_hit} HybridHit:{h_hit} | HybridRecall:{h_rec:.2f}")

print("\n" + "="*50)
print(f"BASELINE DENSE HIT@20  : {dense_hits_20/total_questions:.4f}")
print(f"BM25 ONLY HIT@20       : {bm25_hits_20/total_questions:.4f}")
print(f"HYBRID (RRF) HIT@20    : {hybrid_hits_20/total_questions:.4f}")
print(f"HYBRID CHUNK RECALL@20 : {sum(hybrid_chunk_recalls)/total_questions:.4f}")
print("="*50)
