from functools import lru_cache
import spaces
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """
    Load and cache the Cross-Encoder model.

    The model is loaded only once per process.
    """
    print(f"Loading reranker model: {MODEL_NAME}")

    model = CrossEncoder(MODEL_NAME)

    print("Reranker model loaded successfully.")

    return model


# --------------------------------------------------
# Reranking service
# --------------------------------------------------

class RerankingService:

    def __init__(self):
        """
        Initialize the reranking service.

        The actual model is cached by get_reranker_model().
        """
        self.model = get_reranker_model()
    @spaces.GPU(duration=30)
    def _predic(self,pairs:list[tuple[str,str]],batch_size:int=8):
        return self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )

    def rerank(
        self,
        query: str,
        results: list[dict],
    ) -> list[dict]:
        """
        Rerank retrieved document chunks using a Cross-Encoder.

        Args:
            query:
                User's original question.

            results:
                Candidate chunks retrieved from Qdrant.

        Returns:
            Same result dictionaries with an additional
            'rerank_score' field, sorted by rerank score
            in descending order.
        """

        if not results:
            return []

        # Create query-document pairs.
        pairs = [
            (query, result["text"])
            for result in results
        ]

        # Run all candidates through the model in one batch.
        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        # Attach reranking score to each result.
        reranked_results = []

        for result, score in zip(results, scores):

            reranked_result = result.copy()

            reranked_result["rerank_score"] = float(score)

            reranked_results.append(reranked_result)

        # Sort from most relevant to least relevant.
        reranked_results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked_results
