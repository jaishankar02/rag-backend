"""
RAG Service: Orchestrates retrieval, reranking, and generation.
"""
import time
from app.services.retrieval import retrieve_documents
from app.services.context_builder import build_context
from app.services.generation_service import GenerationService
from app.services.reranking import RerankingService
from app.services.chunking_service import embed_query
from app.services.hybrid_search_service import search_hybrid

class RAGService:

    def __init__(self):

        print("Initializing RAG Service...")

        self.generator = GenerationService()

        self.reranker = RerankingService()

        print("RAG Service initialized.")

    def answer(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
        max_new_tokens: int = 300,
        use_reranker: bool = True,
    ) -> dict:
        """
        Generate an answer using either:

        1. Baseline RAG:
           Qdrant retrieval -> Generation

        2. Reranked RAG:
           Qdrant retrieval -> Cross-Encoder reranking
           -> Generation
        """
        t_start=time.time()
        # ==========================================================
        # STEP 1: RETRIEVAL
        # ==========================================================

        if use_reranker:

            # Retrieve more candidates before reranking.
            retrieval_top_k = top_k * 4

        else:

            # Baseline uses exactly top_k retrieved chunks.
            retrieval_top_k = top_k
        t0=time.time()
        query_embedding = embed_query(query)
        t1=time.time()
        print(f"[TIMING]embed_query:{t1-t0:.2f}s")
        candidates = search_hybrid(
            document_id=document_id,
            query=query,
            query_embedding=query_embedding,
            top_k=retrieval_top_k,
        )
        t2=time.time()
        print(f"[TIMING] search_hybrid:{t2-t1:.2f}s")
        if not candidates:

            return {
                "answer": (
                    "No relevant information was found "
                    "in the provided document."
                ),
                "sources": [],
                "retrieval_type": (
                    "reranked"
                    if use_reranker
                    else "baseline"
                ),
            }

        # ==========================================================
        # STEP 2: OPTIONAL RERANKING
        # ==========================================================

        if use_reranker:

            reranked_results = self.reranker.rerank(
                query=query,
                results=candidates,
            )

            results = reranked_results[:top_k]

            print("\nRERANKED RESULTS:")

            for i, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"Rank {i} | "
                    f"Qdrant Score: "
                    f"{result['score']:.4f} | "
                    f"Rerank Score: "
                    f"{result['rerank_score']:.4f} | "
                    f"Page: {result.get('page', 0)} | "
                    f"Chunk: {result.get('chunk')}"
                )

        else:

            results = candidates[:top_k]

            print("\nBASELINE RESULTS:")

            for i, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"Rank {i} | "
                    f"Qdrant Score: "
                    f"{result['score']:.4f} | "
                    f"Page: {result.get('page', 0)} | "
                    f"Chunk: {result.get('chunk')}"
                )
        t3=time.time()
        print(f"[TIMING] rerank/select:{t3-t2:.2f}s")
        # ==========================================================
        # STEP 3: BUILD CONTEXT
        # ==========================================================

        context = build_context(results)
        t4=time.time()
        print(f"[TIMING]build_context:{t4-t3:.2f}s")
        # ==========================================================
        # STEP 4: GENERATION
        # ==========================================================

        answer = self.generator.generate(
            query=query,
            context=context,
            max_new_tokens=max_new_tokens,
        )
        t5=time.time()
        print(f"[TIMING] generation:{t5-t4:.2f}s")
        # ==========================================================
        # STEP 5: SOURCE INFORMATION
        # ==========================================================

        sources = []

        for result in results:

            source = {
                "document_id": result.get("document_id", document_id),
                "filename": result.get("filename", ""),
                "page": result.get("page", 0),
                "chunk": result.get("chunk"),
                "score": result.get("score", 0.0),
                "text": result.get("text", ""),
            }

            # Add reranking score only when available.
            if "rerank_score" in result:

                source["rerank_score"] = (
                    result["rerank_score"]
                )

            sources.append(source)
        print(f"[TIMING] TOTAL: {time.time() - t_start:.2f}s\n")
        # ==========================================================
        # STEP 6: RETURN
        # ==========================================================

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_type": (
                "reranked"
                if use_reranker
                else "baseline"
            ),
        }
