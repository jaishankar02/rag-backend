from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import RAGService
from app.services.qdrant_service import search_documents


router = APIRouter()


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the document",
    )

    document_id: str = Field(
        ...,
        min_length=1,
        description="ID of the document to search",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


# --------------------------------------------------
# RAG service
# --------------------------------------------------

# IMPORTANT:
# This is created ONCE when this module/application
# process is initialized.

rag_service = RAGService()
# qdrant_service = QdrantService()


# --------------------------------------------------
# Query endpoint
# --------------------------------------------------

@router.post("/query")
async def query_document(request: QueryRequest) -> dict:

    try:

        result = rag_service.answer(
            query=request.query,
            document_id=request.document_id,
            top_k=request.top_k,
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# --------------------------------------------------
# Temporary debugging endpoint
# --------------------------------------------------

@router.get("/debug/document/{document_id}/chunks")
async def get_document_chunks(document_id: str) -> dict:
    """
    Retrieve all chunks for a document from Qdrant.
    
    TEMPORARY DEBUG ENDPOINT - for inspection only.
    """
    try:
        # 'results' is a list returned by Qdrant
        results = search_documents(
            query_embedding=[0.0] * 384,
            top_k=10000,
            document_id=document_id,
        )

        if not results:
            return {
                "document_id": document_id,
                "total_chunks": 0,
                "chunks": [],
            }

        return {
            "document_id": document_id,
            "total_chunks": len(results), # Use len() directly on the list
            "chunks": results,            # Return the list directly
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
