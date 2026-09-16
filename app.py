import gradio as gr
import spaces
from app.main import app as fastapi_app

@spaces.GPU(duration=60)
def status_check():
    return "DocRag Enterprise RAG Backend API is Online & Running 🚀"

# Create a lightweight Gradio interface showing server status
demo = gr.Interface(
    fn=status_check,
    inputs=[],
    outputs="text",
    title="DocRag RAG Backend Engine",
    description="Hybrid Search RAG (Qdrant + Redis BM25 RRF + Cross-Encoder + Qwen GGUF)",
)

# Mount Gradio demo into FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
