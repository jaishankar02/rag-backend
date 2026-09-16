import gradio as gr
from app.main import app as fastapi_app

# Create a lightweight Gradio interface showing server status
demo = gr.Interface(
    fn=lambda: "DocRag Enterprise RAG Backend API is Online & Running 🚀",
    inputs=[],
    outputs="text",
    title="DocRag RAG Backend Engine",
    description="Hybrid Search RAG (Qdrant + Redis BM25 RRF + Cross-Encoder + Qwen GGUF)",
)

# Mount FastAPI app onto Gradio on port 7860
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
