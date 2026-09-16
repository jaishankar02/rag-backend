import multiprocessing
import os
from huggingface_hub import hf_hub_download

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    from ctransformers import AutoModelForCausalLM
    HAS_LLAMA_CPP = False


class GenerationService:

    def __init__(
        self,
        model_path: str = None,
    ):
        # Auto-download from Hugging Face Hub if local path is not present
        if model_path is None or not os.path.exists(model_path):
            print("Downloading GGUF model from Hugging Face Hub...")
            model_path = hf_hub_download(
                repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            )

        print(f"Loading GGUF model from: {model_path}")

        if HAS_LLAMA_CPP:
            threads = max(1, multiprocessing.cpu_count() // 2)
            self.llm_type = "llama_cpp"
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=threads,
                verbose=False,
            )
        else:
            self.llm_type = "ctransformers"
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="qwen2",
                context_length=2048,
            )

        print(f"Generation model ({self.llm_type}) loaded successfully.")

    def build_prompt(self, query: str, context: str) -> str:
        return f"""You are an AI assistant for a document knowledge base.

Answer the user's question using ONLY the information
provided in the context.

Rules:
- Do not use outside knowledge.
- Do not invent or assume information.
- If the answer cannot be found in the context,
  say that the information is not available in
  the provided document.
- Give a clear and concise answer.
- Preserve important technical details, equations,
  definitions, and terminology.

Context:
--------------------
{context}
--------------------

Question:
{query}

Answer:"""

    def generate(self, query: str, context: str, max_new_tokens: int = 200) -> str:
        prompt = self.build_prompt(query=query, context=context)

        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.1,  # Equivalent to do_sample=False
            top_p=0.9,
            repeat_penalty=1.15,
            stop=["\nQuestion:", "Context:"],
        )

        return output["choices"][0]["text"].strip()
