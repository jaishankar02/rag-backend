import gradio as gr
import spaces
from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.routes.upload import upload_router
from app.api.routes.query import router as query_router

load_dotenv()

# Initialize FastAPI directly as the primary app
fastapi_app = FastAPI(title="DocRag API Engine")

# Mount API routes early
fastapi_app.include_router(upload_router)
fastapi_app.include_router(query_router)

@fastapi_app.get("/health")
async def health_check():
    return {"status": "ok"}

@spaces.GPU(duration=60)
def status_check():
    return "DocRag Enterprise RAG Backend API is Online & Running 🚀"

# Create Gradio Blocks app
with gr.Blocks(title="DocRag RAG Backend Engine") as demo:
    gr.Markdown("# 🚀 DocRag RAG Backend Engine")
    gr.Markdown("Hybrid Search RAG: Qdrant Dense Vector + Redis BM25 RRF + Cross-Encoder + Qwen GGUF")
    btn = gr.Button("Check Server Status")
    output = gr.Textbox(label="Status")
    btn.click(fn=status_check, inputs=[], outputs=output)

# BEST PRACTICE FOR HUGGING FACE SPACES:
# Mount Gradio ON TOP of the standard FastAPI app globally.
# Hugging Face Spaces will automatically detect the variable named `app` and run it via Uvicorn.
# It MUST be named 'app'. Do not call demo.launch()!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
