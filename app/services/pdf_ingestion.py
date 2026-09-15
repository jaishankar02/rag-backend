from typing import Dict, Any, List
import logging
from pathlib import Path

from .chunking_service import create_chunks
from .pdf_extractor import extract_text_from_pdf
from .qdrant_service import store_document
from .text_cleaner import clean_extracted_pages
import re
from rank_bm25 import BM25Okapi
from app.services.redis_service import save_bm25_index

logger = logging.getLogger(__name__)


def ingest_pdf(pdf_path: str, document_id: str) -> Dict[str, Any]:
    """Ingest a PDF given its path and a document identifier.

    Pipeline:
    - extract page-by-page blocks from the PDF using `extract_text_from_pdf`
    - clean extracted page blocks
    - create fixed-size chunks with embeddings
    - return a chunk-first payload
    """
    logger.info("Starting ingestion for document_id=%s, path=%s", document_id, pdf_path)

    pages: List[Dict[str, Any]] = extract_text_from_pdf(pdf_path)
    pages = clean_extracted_pages(pages)

    chunks = create_chunks(pages, document_id)

    extracted_data = {
        "filename": Path(pdf_path).name,
    }

    # --------------------------------------------------
    # 3. Prepare document
    # --------------------------------------------------

    document = {
        "document_id": document_id,
        "filename": extracted_data["filename"],
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
    def tokenize(text: str):
        return re.findall(r'\w+', text.lower())

     # Inside store_document(document: dict):
    # 1. Upsert points into Qdrant (existing logic)...
    # 2. Build BM25 Index during Upload
    chunks = document["chunks"]
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(corpus)
    # 3. Persist to Redis Cloud
    save_bm25_index(document["document_id"], bm25_index, chunks) 


    # --------------------------------------------------
    # 4. Store embeddings in Qdrant
    # --------------------------------------------------

    qdrant_result = store_document(document)

    payload: Dict[str, Any] = {
        "document_id": document_id,
        "filename": extracted_data["filename"],
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chunks": chunks,
        **qdrant_result,
    }

    logger.info(
        "Finished ingestion for document_id=%s (pages=%d, chunks=%d)",
        document_id,
        len(pages),
        len(chunks),
    )

    return payload
