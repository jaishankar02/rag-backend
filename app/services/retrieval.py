from app.services.chunking_service import embed_query
from app.services.qdrant_service import search_documents


def retrieve_documents(
    query: str,
    document_id: str,
    top_k: int = 5,
):
    """
    Retrieve the most relevant chunks for a user query.

    Args:
        query: User's question.
        document_id: ID of the document to search within.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        List of retrieved document chunks with metadata.
    """

    # Convert the user's query into an embedding
    query_embedding = embed_query(query)

    # Search Qdrant using the query embedding
    results = search_documents(
        query_embedding=query_embedding,
        top_k=top_k,
        document_id=document_id,
    )

    # Debug information
    print("\nRAW RETRIEVAL RESULTS:")

    for i, result in enumerate(results, start=1):
        print(
            f"Rank {i} | "
            f"Score: {result['score']:.4f} | "
            f"Page: {result['page']} | "
            f"Chunk: {result['chunk']}"
        )

    return results
