import os
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except Exception:  # pragma: no cover - dependency may be missing in system python
    QdrantClient = None
    Distance = FieldCondition = Filter = MatchValue = PointStruct = VectorParams = None


# QDRANT_URL = os.getenv("QDRANT_URL", "https://7ea0bfa2-12dc-4a10-860d-e2d297315201.sa-east-1-0.aws.cloud.qdrant.io")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "document_chunks")
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODQ5NzZmYTgtMzA4My00MThhLTg3ZGItNTc3YjEyZDFlYTYzIn0.fQv4EhQ4xm9MBrqNmVctuBFKrH3GSemgBWhIUMOe79M


client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
) if QdrantClient is not None else None


def _require_client():
    if client is None:
        raise RuntimeError(
            "qdrant-client is required. Install it in the active environment before storing or searching chunks."
        )


def create_collection_if_not_exists(vector_size: int):
    _require_client()
    collections = client.get_collections().collections

    if any(collection.name == COLLECTION_NAME for collection in collections):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def store_document(document: dict):
    _require_client()
    chunks = document["chunks"]

    if not chunks:
        raise ValueError("No chunks found in document.")

    vector_size = len(chunks[0]["embedding"])

    for chunk in chunks:
        if len(chunk["embedding"]) != vector_size:
            raise ValueError(
                f"Embedding dimension mismatch for chunk {chunk['chunk']}"
            )

    create_collection_if_not_exists(vector_size)

    points = []

    for chunk in chunks:
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{document['document_id']}:{chunk['chunk']}",
            )
        )

        points.append(
            PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={
                    "document_id": document["document_id"],
                    "filename": document.get("filename"),
                    "page": chunk["page"],
                    "chunk": chunk["chunk"],
                    "text": chunk["text"],
                    "start_sentence": chunk.get("start_sentence"),
                    "end_sentence": chunk.get("end_sentence"),
                    "bbox": chunk.get("bbox"),
                    "source_blocks": chunk.get("source_blocks"),
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return {
        "document_id": document["document_id"],
        "chunks_stored": len(points),
    }


def search_documents(
    query_embedding: list[float],
    top_k: int = 5,
    document_id: str | None = None,
):
    """Search Qdrant for the most similar document chunks."""

    _require_client()

    query_filter = None

    if document_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score": result.score,
            "document_id": result.payload.get("document_id"),
            "filename": result.payload.get("filename"),
            "page": result.payload.get("page"),
            "chunk": result.payload.get("chunk"),
            "text": result.payload.get("text"),
            "start_sentence": result.payload.get("start_sentence"),
            "end_sentence": result.payload.get("end_sentence"),
            "bbox": result.payload.get("bbox"),
            "source_blocks": result.payload.get("source_blocks"),
        }
        for result in results.points
    ]


def get_chunks_by_ids(
    document_id: str,
    chunk_ids: list[int],
):
    """Retrieve exact chunks from Qdrant using document_id and chunk IDs."""

    _require_client()

    if not chunk_ids:
        return []

    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
        limit=1000,
        with_payload=True,
    )

    chunks = []

    for result in results:
        chunk_id = result.payload.get("chunk")

        if chunk_id in chunk_ids:
            chunks.append(
                {
                    "document_id": result.payload.get("document_id"),
                    "filename": result.payload.get("filename"),
                    "page": result.payload.get("page"),
                    "chunk": chunk_id,
                    "text": result.payload.get("text"),
                }
            )

    chunks.sort(key=lambda x: chunk_ids.index(x["chunk"]))

    return chunks
