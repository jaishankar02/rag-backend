import json
import os


# ============================================================
# Configuration
# ============================================================

BASE_DIR = (
    "/home/jaishankar/Documents/resume_projects/"
    "DocRag/backend"
)
# /home/jaishankar/Documents/resume_projects/DocRag/backend
RESULTS_FILE = os.path.join(
    BASE_DIR,
    "tests",
    "evaluation_reranking_results.json",
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "tests",
    "evaluation_metrics.json",
)


# ============================================================
# Helper Functions
# ============================================================

def get_chunk_id(result):
    """
    Get the chunk ID from a retrieval result.
    """

    return result.get("chunk")


def get_ranked_chunk_ids(results):
    """
    Convert retrieval/reranking results into an ordered
    list of chunk IDs.
    """

    return [
        get_chunk_id(result)
        for result in results
        if get_chunk_id(result) is not None
    ]


def hit_at_k(ranked_chunks, ground_truth_chunks, k):
    """
    Hit-based Recall@K.

    Returns:
        1 if at least one ground-truth chunk appears
        in the top K results.

        0 otherwise.
    """

    top_k = set(ranked_chunks[:k])
    ground_truth = set(ground_truth_chunks)

    return 1 if top_k.intersection(ground_truth) else 0


def chunk_recall_at_k(ranked_chunks, ground_truth_chunks, k):
    """
    Chunk-level Recall@K.

    Measures what fraction of all ground-truth chunks
    were retrieved in the top K results.

    Example:

        Ground truth = [10, 20, 30]

        Top 5 = [10, 20, 5, 6, 7]

        Recall@5 = 2 / 3 = 0.667
    """

    if not ground_truth_chunks:
        return 0.0

    top_k = set(ranked_chunks[:k])
    ground_truth = set(ground_truth_chunks)

    found = top_k.intersection(ground_truth)

    return len(found) / len(ground_truth)


def reciprocal_rank(ranked_chunks, ground_truth_chunks):
    """
    Calculate Reciprocal Rank.

    RR = 1 / rank of the first relevant chunk.

    If no relevant chunk is found:
        RR = 0
    """

    ground_truth = set(ground_truth_chunks)

    for rank, chunk_id in enumerate(
        ranked_chunks,
        start=1,
    ):
        if chunk_id in ground_truth:
            return 1.0 / rank

    return 0.0


def first_relevant_rank(
    ranked_chunks,
    ground_truth_chunks,
):
    """
    Return the rank of the first relevant chunk.

    Returns None if no relevant chunk is found.
    """

    ground_truth = set(ground_truth_chunks)

    for rank, chunk_id in enumerate(
        ranked_chunks,
        start=1,
    ):
        if chunk_id in ground_truth:
            return rank

    return None


def calculate_average(values):
    """
    Calculate the average of a list of values.
    """

    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# Main Evaluation
# ============================================================

def main():

    print("=" * 80)
    print("RAG RETRIEVAL + RERANKING METRICS")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Check input file
    # --------------------------------------------------------

    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"Results file not found:\n{RESULTS_FILE}"
        )

    # --------------------------------------------------------
    # 2. Load evaluation results
    # --------------------------------------------------------

    print(
        f"\nLoading results from:\n{RESULTS_FILE}"
    )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        evaluation_data = json.load(f)

    results = evaluation_data["results"]

    total_questions = len(results)

    print(
        f"Questions found: {total_questions}"
    )

    # --------------------------------------------------------
    # 3. Metric containers
    # --------------------------------------------------------

    # Baseline hit-based Recall@K
    baseline_hit_at_5 = []
    baseline_hit_at_10 = []
    baseline_hit_at_20 = []

    # Reranked hit-based Recall@K
    reranked_hit_at_5 = []
    reranked_hit_at_10 = []
    reranked_hit_at_20 = []

    # Baseline chunk-level Recall@K
    baseline_recall_at_5 = []
    baseline_recall_at_10 = []
    baseline_recall_at_20 = []

    # Reranked chunk-level Recall@K
    reranked_recall_at_5 = []
    reranked_recall_at_10 = []
    reranked_recall_at_20 = []

    # Reciprocal Rank
    baseline_rr = []
    reranked_rr = []

    # Per-question results
    per_question_results = []

    # --------------------------------------------------------
    # 4. Evaluate every question
    # --------------------------------------------------------

    for result in results:

        question_id = result["question_id"]
        question = result["question"]

        ground_truth_chunks = result.get(
            "ground_truth_chunk_ids",
            [],
        )

        retrieved_results = result.get(
            "retrieved_results",
            [],
        )

        reranked_results = result.get(
            "reranked_results",
            [],
        )

        # ----------------------------------------------------
        # Extract chunk IDs
        # ----------------------------------------------------

        baseline_chunks = get_ranked_chunk_ids(
            retrieved_results
        )

        reranked_chunks = get_ranked_chunk_ids(
            reranked_results
        )

        # ====================================================
        # BASELINE — QDRANT
        # ====================================================

        b_hit_5 = hit_at_k(
            baseline_chunks,
            ground_truth_chunks,
            5,
        )

        b_hit_10 = hit_at_k(
            baseline_chunks,
            ground_truth_chunks,
            10,
        )

        b_hit_20 = hit_at_k(
            baseline_chunks,
            ground_truth_chunks,
            20,
        )

        b_recall_5 = chunk_recall_at_k(
            baseline_chunks,
            ground_truth_chunks,
            5,
        )

        b_recall_10 = chunk_recall_at_k(
            baseline_chunks,
            ground_truth_chunks,
            10,
        )

        b_recall_20 = chunk_recall_at_k(
            baseline_chunks,
            ground_truth_chunks,
            20,
        )

        b_rr = reciprocal_rank(
            baseline_chunks,
            ground_truth_chunks,
        )

        b_first_rank = first_relevant_rank(
            baseline_chunks,
            ground_truth_chunks,
        )

        # ====================================================
        # RERANKED — CROSS ENCODER
        # ====================================================

        r_hit_5 = hit_at_k(
            reranked_chunks,
            ground_truth_chunks,
            5,
        )

        r_hit_10 = hit_at_k(
            reranked_chunks,
            ground_truth_chunks,
            10,
        )

        r_hit_20 = hit_at_k(
            reranked_chunks,
            ground_truth_chunks,
            20,
        )

        r_recall_5 = chunk_recall_at_k(
            reranked_chunks,
            ground_truth_chunks,
            5,
        )

        r_recall_10 = chunk_recall_at_k(
            reranked_chunks,
            ground_truth_chunks,
            10,
        )

        r_recall_20 = chunk_recall_at_k(
            reranked_chunks,
            ground_truth_chunks,
            20,
        )

        r_rr = reciprocal_rank(
            reranked_chunks,
            ground_truth_chunks,
        )

        r_first_rank = first_relevant_rank(
            reranked_chunks,
            ground_truth_chunks,
        )

        # ====================================================
        # Store metric values
        # ====================================================

        baseline_hit_at_5.append(b_hit_5)
        baseline_hit_at_10.append(b_hit_10)
        baseline_hit_at_20.append(b_hit_20)

        reranked_hit_at_5.append(r_hit_5)
        reranked_hit_at_10.append(r_hit_10)
        reranked_hit_at_20.append(r_hit_20)

        baseline_recall_at_5.append(b_recall_5)
        baseline_recall_at_10.append(b_recall_10)
        baseline_recall_at_20.append(b_recall_20)

        reranked_recall_at_5.append(r_recall_5)
        reranked_recall_at_10.append(r_recall_10)
        reranked_recall_at_20.append(r_recall_20)

        baseline_rr.append(b_rr)
        reranked_rr.append(r_rr)

        # ====================================================
        # Print question evaluation
        # ====================================================

        print("\n")
        print("-" * 80)
        print(f"QUESTION {question_id}")
        print("-" * 80)

        print(f"Question: {question}")

        print(
            f"Ground truth chunks: "
            f"{ground_truth_chunks}"
        )

        print(
            f"Baseline first relevant rank: "
            f"{b_first_rank}"
        )

        print(
            f"Reranked first relevant rank: "
            f"{r_first_rank}"
        )

        print(
            f"Baseline  Hit@5: "
            f"{b_hit_5}"
        )

        print(
            f"Reranked  Hit@5: "
            f"{r_hit_5}"
        )

        print(
            f"Baseline  Recall@5: "
            f"{b_recall_5:.4f}"
        )

        print(
            f"Reranked  Recall@5: "
            f"{r_recall_5:.4f}"
        )

        print(
            f"Baseline  RR: "
            f"{b_rr:.4f}"
        )

        print(
            f"Reranked  RR: "
            f"{r_rr:.4f}"
        )

        # ====================================================
        # Store per-question results
        # ====================================================

        per_question_results.append(
            {
                "question_id": question_id,

                "question": question,

                "ground_truth_chunk_ids":
                    ground_truth_chunks,

                "baseline": {
                    "first_relevant_rank":
                        b_first_rank,

                    "hit_at_5":
                        b_hit_5,

                    "hit_at_10":
                        b_hit_10,

                    "hit_at_20":
                        b_hit_20,

                    "recall_at_5":
                        b_recall_5,

                    "recall_at_10":
                        b_recall_10,

                    "recall_at_20":
                        b_recall_20,

                    "reciprocal_rank":
                        b_rr,
                },

                "reranked": {
                    "first_relevant_rank":
                        r_first_rank,

                    "hit_at_5":
                        r_hit_5,

                    "hit_at_10":
                        r_hit_10,

                    "hit_at_20":
                        r_hit_20,

                    "recall_at_5":
                        r_recall_5,

                    "recall_at_10":
                        r_recall_10,

                    "recall_at_20":
                        r_recall_20,

                    "reciprocal_rank":
                        r_rr,
                },
            }
        )

    # ========================================================
    # 5. Calculate overall baseline metrics
    # ========================================================

    baseline_metrics = {

        "hit_recall_at_5":
            calculate_average(
                baseline_hit_at_5
            ),

        "hit_recall_at_10":
            calculate_average(
                baseline_hit_at_10
            ),

        "hit_recall_at_20":
            calculate_average(
                baseline_hit_at_20
            ),

        "chunk_recall_at_5":
            calculate_average(
                baseline_recall_at_5
            ),

        "chunk_recall_at_10":
            calculate_average(
                baseline_recall_at_10
            ),

        "chunk_recall_at_20":
            calculate_average(
                baseline_recall_at_20
            ),

        "mrr":
            calculate_average(
                baseline_rr
            ),
    }

    # ========================================================
    # 6. Calculate overall reranked metrics
    # ========================================================

    reranked_metrics = {

        "hit_recall_at_5":
            calculate_average(
                reranked_hit_at_5
            ),

        "hit_recall_at_10":
            calculate_average(
                reranked_hit_at_10
            ),

        "hit_recall_at_20":
            calculate_average(
                reranked_hit_at_20
            ),

        "chunk_recall_at_5":
            calculate_average(
                reranked_recall_at_5
            ),

        "chunk_recall_at_10":
            calculate_average(
                reranked_recall_at_10
            ),

        "chunk_recall_at_20":
            calculate_average(
                reranked_recall_at_20
            ),

        "mrr":
            calculate_average(
                reranked_rr
            ),
    }

    # ========================================================
    # 7. Calculate improvement
    # ========================================================

    improvement = {}

    for metric in baseline_metrics:

        baseline_value = baseline_metrics[metric]

        reranked_value = reranked_metrics[metric]

        absolute_improvement = (
            reranked_value
            - baseline_value
        )

        if baseline_value != 0:

            percentage_improvement = (
                absolute_improvement
                / baseline_value
            ) * 100

        else:

            percentage_improvement = None

        improvement[metric] = {

            "baseline":
                baseline_value,

            "reranked":
                reranked_value,

            "absolute_improvement":
                absolute_improvement,

            "percentage_improvement":
                percentage_improvement,
        }

    # ========================================================
    # 8. Create final evaluation report
    # ========================================================

    evaluation_report = {

        "document_id":
            evaluation_data.get(
                "document_id"
            ),

        "total_questions":
            total_questions,

        "retrieval_top_k":
            evaluation_data.get(
                "retrieval_top_k",
                20,
            ),

        "rerank_top_k":
            evaluation_data.get(
                "rerank_top_k",
                5,
            ),

        "baseline_metrics":
            baseline_metrics,

        "reranked_metrics":
            reranked_metrics,

        "improvement":
            improvement,

        "per_question":
            per_question_results,
    }

    # ========================================================
    # 9. Save evaluation report
    # ========================================================

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
            evaluation_report,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # 10. Print final summary
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)

    print("\nBASELINE — QDRANT")

    print(
        f"Recall@5  : "
        f"{baseline_metrics['hit_recall_at_5']:.4f}"
    )

    print(
        f"Recall@10 : "
        f"{baseline_metrics['hit_recall_at_10']:.4f}"
    )

    print(
        f"Recall@20 : "
        f"{baseline_metrics['hit_recall_at_20']:.4f}"
    )

    print(
        f"MRR       : "
        f"{baseline_metrics['mrr']:.4f}"
    )

    print("\nRERANKED — CROSS-ENCODER")

    print(
        f"Recall@5  : "
        f"{reranked_metrics['hit_recall_at_5']:.4f}"
    )

    print(
        f"Recall@10 : "
        f"{reranked_metrics['hit_recall_at_10']:.4f}"
    )

    print(
        f"Recall@20 : "
        f"{reranked_metrics['hit_recall_at_20']:.4f}"
    )

    print(
        f"MRR       : "
        f"{reranked_metrics['mrr']:.4f}"
    )

    print("\nIMPROVEMENT")

    for metric, values in improvement.items():

        percentage = values[
            "percentage_improvement"
        ]

        if percentage is None:

            percentage_text = "N/A"

        else:

            percentage_text = (
                f"{percentage:+.2f}%"
            )

        print(
            f"{metric:25s}: "
            f"{values['absolute_improvement']:+.4f} "
            f"({percentage_text})"
        )

    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Metrics saved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()