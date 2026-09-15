import os
import pickle
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "fly-daisy-charming-80096.db.redis.io")
REDIS_PORT = int(os.getenv("REDIS_PORT", 15567))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "PJDatqDFbQfNE9x10nDtsO7NdMK3UesY")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    username=REDIS_USERNAME,
    password=REDIS_PASSWORD,
    decode_responses=False  # Required for binary pickle objects
)

def save_bm25_index(document_id: str, bm25_index, chunk_mapping: list):
    """Serialize and save BM25 index and chunk mapping to Redis Cloud."""
    data = {
        "bm25": bm25_index,
        "chunks": chunk_mapping
    }
    payload = pickle.dumps(data)
    r.set(f"bm25:{document_id}", payload)

def get_bm25_index(document_id: str):
    """Retrieve and deserialize BM25 index from Redis Cloud."""
    payload = r.get(f"bm25:{document_id}")
    if not payload:
        return None
    return pickle.loads(payload)
