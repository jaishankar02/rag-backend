def build_context(results: list[dict]) -> str:
    """
    Convert retrieved document chunks into a formatted
    context string for the generation model.

    Args:
        results: Retrieved chunks from the vector store.

    Returns:
        Formatted context string.
    """

    context_parts = []

    for result in results:

        context_parts.append(
            f"""[Page {result['page']}, Chunk {result['chunk']}]
{result['text']}"""
        )

    return "\n\n".join(context_parts)
