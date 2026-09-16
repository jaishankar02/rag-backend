import gradio as gr
import spaces

# Import FastAPI routers and services (triggers model loading)
from app.api.routes.upload import upload_router
from app.api.routes.query import router as query_router
from dotenv import load_dotenv

load_dotenv()

@spaces.GPU(duration=60)
def status_check():
    return "DocRag Enterprise RAG Backend API is Online & Running 🚀"

# Create Gradio Blocks app
with gr.Blocks(title="DocRag RAG Backend Engine") as demo:
    gr.Markdown("# 🚀 DocRag RAG Backend Engine")
    gr.Markdown("Hybrid Search RAG: Qdrant Dense Vector + Redis BM25 RRF + Cross-Encoder + Qwen GGUF")
    gr.Markdown("### API Endpoints available at:")
    gr.Markdown("- `GET /health` — Health check")
    gr.Markdown("- `POST /upload/file` — Upload & index a PDF")
    gr.Markdown("- `POST /query` — Ask a question about your document")
    btn = gr.Button("Check Server Status")
    output = gr.Textbox(label="Status")
    btn.click(fn=status_check, inputs=[], outputs=output)

# Mount FastAPI routes onto Gradio's internal FastAPI app
demo.app.include_router(upload_router)
demo.app.include_router(query_router)

@demo.app.get("/health")
async def health_check():
    return {"status": "ok"}

# Launch using Gradio (required for ZeroGPU)
demo.launch(server_name="0.0.0.0", server_port=7860, ssr=False)
