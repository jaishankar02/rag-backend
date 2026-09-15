import json
import os

from app.services.chunking_service import embed_query
from app.services.hybrid_search_service import search_hybrid
from app.services.reranking import RerankingService


# ============================================================
# Configuration
# ============================================================

BASE_DIR = "/home/jaishankar/Documents/resume_projects/DocRag/backend"

QUESTIONS_FILE = os.path.join(
    BASE_DIR,
    "tests",
    "question.json",
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "tests",
    "evaluation_reranking_results.json",
)

DOCUMENT_ID = "eee0fa53e010454d9d1d80ce149b8da2"

# Number of candidates retrieved from Qdrant
RETRIEVAL_TOP_K = 20

# Number of chunks kept after reranking
RERANK_TOP_K = 5


# ============================================================
# Helper Functions
# ============================================================

def get_chunk_id(result):
    """
    Return the chunk ID from a retrieval result.

    Your Qdrant result uses the field:
        result["chunk"]
    """

    return result.get("chunk")


def find_chunk_ranks(results, ground_truth_chunk_ids):
    """
    Find the ranks of all ground-truth chunks in a result list.

    Returns:
        {
            "found_chunks": [...],
            "missing_chunks": [...],
            "ranks": {...}
        }
    """

    ground_truth = set(ground_truth_chunk_ids)

    ranks = {}
    found_chunks = []
    missing_chunks = []

    for rank, result in enumerate(results, start=1):

        chunk_id = get_chunk_id(result)

        if chunk_id in ground_truth:
            ranks[chunk_id] = rank
            found_chunks.append(chunk_id)

    for chunk_id in ground_truth_chunk_ids:

        if chunk_id not in ranks:
            missing_chunks.append(chunk_id)

    return {
        "found_chunks": found_chunks,
        "missing_chunks": missing_chunks,
        "ranks": ranks,
    }


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Validate input file
    # --------------------------------------------------------

    if not os.path.exists(QUESTIONS_FILE):

        raise FileNotFoundError(
            f"Question file not found:\n{QUESTIONS_FILE}"
        )

    print("=" * 80)
    print("RAG RETRIEVAL + RERANKING EVALUATION")
    print("=" * 80)

    print(f"\nQuestion file : {QUESTIONS_FILE}")
    print(f"Output file   : {OUTPUT_FILE}")

    # --------------------------------------------------------
    # 2. Load questions
    # --------------------------------------------------------

    print("\nLoading evaluation questions...")

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        evaluation_data = json.load(f)

    questions = evaluation_data["questions"]

    print(f"Loaded {len(questions)} questions.")

    # --------------------------------------------------------
    # 3. Initialize reranker
    # --------------------------------------------------------

    print("\nInitializing Cross-Encoder reranker...")

    reranker = RerankingService()

    print("Reranker initialized.")

    # --------------------------------------------------------
    # 4. Process questions
    # --------------------------------------------------------

    all_results = []

    for question_index, question_data in enumerate(
        questions,
        start=1,
    ):

        question_id = question_data["question_id"]
        query = question_data["question"]

        ground_truth_chunk_ids = question_data.get(
            "ground_truth_chunk_ids",
            [],
        )

        print("\n")
        print("=" * 80)
        print(
            f"QUESTION {question_index}/{len(questions)}"
        )
        print("=" * 80)

        print(f"Question ID : {question_id}")
        print(f"Query       : {query}")

        print(
            f"Ground truth chunks: "
            f"{ground_truth_chunk_ids}"
        )

        # ====================================================
        # STEP 1: QDRANT RETRIEVAL
        # ====================================================

        print("\nRetrieving top 20 chunks using Hybrid Search (Dense + Redis BM25 RRF)...")

        query_embedding = embed_query(query)
        retrieved_results = search_hybrid(
            document_id=DOCUMENT_ID,
            query=query,
            query_embedding=query_embedding,
            top_k=RETRIEVAL_TOP_K,
        )

        # ----------------------------------------------------
        # Make sure retrieval actually returned something
        # ----------------------------------------------------

        if not retrieved_results:

            print("WARNING: Qdrant returned no results.")

            reranked_results = []

        else:

            # =================================================
            # STEP 2: CROSS-ENCODER RERANKING
            # =================================================

            print(
                f"\nReranking {len(retrieved_results)} "
                "retrieved chunks..."
            )

            reranked_results = reranker.rerank(
                query=query,
                results=retrieved_results,
            )

        # ====================================================
        # STEP 3: FINAL TOP 5
        # ====================================================

        final_top_5 = reranked_results[
            :RERANK_TOP_K
        ]

        # ====================================================
        # STEP 4: FIND GROUND-TRUTH RANKS
        # ====================================================

        baseline_analysis = find_chunk_ranks(
            retrieved_results,
            ground_truth_chunk_ids,
        )

        reranked_analysis = find_chunk_ranks(
            reranked_results,
            ground_truth_chunk_ids,
        )

        # ====================================================
        # STEP 5: PRINT RETRIEVAL RESULTS
        # ====================================================

        print("\n")
        print("-" * 80)
        print("QDRANT TOP 20")
        print("-" * 80)

        for rank, result in enumerate(
            retrieved_results,
            start=1,
        ):

            chunk_id = get_chunk_id(result)

            marker = ""

            if chunk_id in ground_truth_chunk_ids:
                marker = "  <-- GROUND TRUTH"

            print(
                f"Rank {rank:2d} | "
                f"Score: {result['score']:.4f} | "
                f"Page: {result.get('page', 0)} | "
                f"Chunk: {chunk_id}"
                f"{marker}"
            )

        # ====================================================
        # STEP 6: PRINT RERANKING RESULTS
        # ====================================================

        print("\n")
        print("-" * 80)
        print("CROSS-ENCODER RERANKED TOP 20")
        print("-" * 80)

        for rank, result in enumerate(
            reranked_results,
            start=1,
        ):

            chunk_id = get_chunk_id(result)

            marker = ""

            if chunk_id in ground_truth_chunk_ids:
                marker = "  <-- GROUND TRUTH"

            print(
                f"Rank {rank:2d} | "
                f"Qdrant: {result['score']:.4f} | "
                f"Rerank: "
                f"{result.get('rerank_score', 0):.4f} | "
                f"Page: {result.get('page', 0)} | "
                f"Chunk: {chunk_id}"
                f"{marker}"
            )

        # ====================================================
        # STEP 7: PRINT FINAL TOP 5
        # ====================================================

        print("\n")
        print("-" * 80)
        print("FINAL TOP 5")
        print("-" * 80)

        for rank, result in enumerate(
            final_top_5,
            start=1,
        ):

            chunk_id = get_chunk_id(result)

            marker = ""

            if chunk_id in ground_truth_chunk_ids:
                marker = "  <-- GROUND TRUTH"

            print(
                f"Rank {rank} | "
                f"Page: {result.get('page', 0)} | "
                f"Chunk: {chunk_id} | "
                f"Rerank Score: "
                f"{result.get('rerank_score', 0):.4f}"
                f"{marker}"
            )

            print(f"Text: {result['text']}")
            print()

        # ====================================================
        # STEP 8: PRINT GROUND-TRUTH ANALYSIS
        # ====================================================

        print("-" * 80)
        print("GROUND-TRUTH ANALYSIS")
        print("-" * 80)

        print(
            "Ground truth chunks: "
            f"{ground_truth_chunk_ids}"
        )

        print(
            "Baseline ranks: "
            f"{baseline_analysis['ranks']}"
        )

        print(
            "Reranked ranks: "
            f"{reranked_analysis['ranks']}"
        )

        print(
            "Baseline missing: "
            f"{baseline_analysis['missing_chunks']}"
        )

        print(
            "Reranked missing: "
            f"{reranked_analysis['missing_chunks']}"
        )

        # ====================================================
        # STEP 9: STORE RESULTS
        # ====================================================

        all_results.append(
            {
                "question_id": question_id,

                "question": query,

                "ground_truth_chunk_ids":
                    ground_truth_chunk_ids,

                "retrieval_top_k":
                    RETRIEVAL_TOP_K,

                "rerank_top_k":
                    RERANK_TOP_K,

                # --------------------------------------------
                # Original Qdrant results
                # --------------------------------------------

                "retrieved_results":
                    retrieved_results,

                # --------------------------------------------
                # Reranked results
                # --------------------------------------------

                "reranked_results":
                    reranked_results,

                # --------------------------------------------
                # Final context candidates
                # --------------------------------------------

                "final_top_5":
                    final_top_5,

                # --------------------------------------------
                # Evaluation information
                # --------------------------------------------

                "baseline_analysis":
                    baseline_analysis,

                "reranked_analysis":
                    reranked_analysis,
            }
        )

    # ========================================================
    # 10. Save complete evaluation results
    # ========================================================

    output = {
        "document_id": DOCUMENT_ID,

        "filename": evaluation_data.get(
            "filename",
            "2502.02501v1.pdf",
        ),

        "retrieval_top_k":
            RETRIEVAL_TOP_K,

        "rerank_top_k":
            RERANK_TOP_K,

        "total_questions":
            len(questions),

        "results":
            all_results,
    }

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # 11. Final summary
    # ========================================================

    print("\n")
    print("=" * 80)
    print("EVALUATION DATA GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"Questions processed : {len(questions)}"
    )

    print(
        f"Qdrant candidates   : {RETRIEVAL_TOP_K}"
    )

    print(
        f"Reranking candidates: {RETRIEVAL_TOP_K}"
    )

    print(
        f"Final context       : {RERANK_TOP_K}"
    )

    print(
        f"Output file         : {OUTPUT_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()