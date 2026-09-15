from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional
import re

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """Load and cache the embedding model once per process."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "sentence-transformers is required for chunk embeddings. Install it first."
        ) from exc

    return SentenceTransformer(MODEL_NAME)


def embed_query(query: str) -> List[float]:
    """Embed a query string using the loaded sentence transformer model."""
    model = get_embedding_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while keeping output compact and readable."""
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def merge_bboxes(bboxes: List[List[float]]) -> Optional[List[float]]:
    """Create a single bounding box covering all supplied boxes."""
    if not bboxes:
        return None

    return [
        float(min(bbox[0] for bbox in bboxes)),
        float(min(bbox[1] for bbox in bboxes)),
        float(max(bbox[2] for bbox in bboxes)),
        float(max(bbox[3] for bbox in bboxes)),
    ]


def _page_number(page_data: Dict[str, Any]) -> int:
    return int(page_data.get("page_number", page_data.get("page", 0)))


def _block_id(page_num: int, block_index: int, block: Dict[str, Any]) -> str:
    existing = block.get("block_id")
    if existing:
        return str(existing)
    return f"p{page_num}_b{block_index}"


def create_chunks(
    pages_data: List[Dict[str, Any]], 
    document_id: str, 
    chunk_size: int = 1024
) -> List[Dict[str, Any]]:
    """Create fixed-size chunks while preserving page, spatial, and page-local sentence metadata."""
    if not pages_data:
        return []

    chunks: List[Dict[str, Any]] = []
    chunk_number = 0

    for page_data in pages_data:
        page_num = _page_number(page_data)
        blocks = page_data.get("blocks", []) or []

        sentence_items: List[Dict[str, Any]] = []
        sentence_index = 1  # Resets to 1 on every page

        for block_index, block in enumerate(blocks, start=1):
            block_text = block.get("text", "")
            if not block_text:
                continue

            # Skip non-text/image blocks if the extractor passed them through.
            if block.get("kind") not in (None, "text"):
                continue

            block_sentences = split_into_sentences(block_text)
            if not block_sentences:
                block_sentences = [block_text.strip()]

            block_bbox = block.get("bbox")
            block_ref = _block_id(page_num, block_index, block)

            for sentence in block_sentences:
                sentence_items.append(
                    {
                        "text": sentence,
                        "bbox": block_bbox,
                        "block_id": block_ref,
                        "sentence_index": sentence_index,
                    }
                )
                sentence_index += 1

        if not sentence_items:
            continue

        page_chunks: List[Dict[str, Any]] = []
        current_sentences: List[str] = []
        current_bboxes: List[List[float]] = []
        current_blocks: List[str] = []
        current_length = 0
        start_sentence = sentence_items[0]["sentence_index"]

        for item in sentence_items:
            sentence = item["text"]
            would_exceed = bool(current_sentences) and (current_length + len(sentence) + 1 > chunk_size)

            if would_exceed:
                chunk_text = " ".join(current_sentences).strip()
                page_chunks.append(
                    {
                        "document_id": document_id,
                        "page": page_num,
                        "chunk": chunk_number,
                        "text": chunk_text,
                        "start_sentence": start_sentence,
                        "end_sentence": item["sentence_index"] - 1,
                        "bbox": merge_bboxes(current_bboxes),
                        "source_blocks": sorted(set(current_blocks)),
                    }
                )
                chunk_number += 1

                current_sentences = [sentence]
                current_bboxes = [item["bbox"]] if item["bbox"] else []
                current_blocks = [item["block_id"]]
                current_length = len(sentence)
                start_sentence = item["sentence_index"]
                continue

            if not current_sentences:
                start_sentence = item["sentence_index"]

            current_sentences.append(sentence)
            current_length += len(sentence) + (1 if len(current_sentences) > 1 else 0)
            if item["bbox"]:
                current_bboxes.append(item["bbox"])
            current_blocks.append(item["block_id"])

        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            page_chunks.append(
                {
                    "document_id": document_id,
                    "page": page_num,
                    "chunk": chunk_number,
                    "text": chunk_text,
                    "start_sentence": start_sentence,
                    "end_sentence": sentence_items[-1]["sentence_index"],
                    "bbox": merge_bboxes(current_bboxes),
                    "source_blocks": sorted(set(current_blocks)),
                }
            )
            chunk_number += 1

        # Embed chunks per page
        embeddings = get_embedding_model().encode(
            [chunk["text"] for chunk in page_chunks],
            normalize_embeddings=True,
        )

        for chunk, embedding in zip(page_chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
            chunks.append(chunk)

    return chunks