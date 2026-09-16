import gradio as gr
import spaces
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

from app.api.routes.upload import upload_router
from app.api.routes.query import router as query_router

load_dotenv()

# Initialize FastAPI directly as the primary app
fastapi_app = FastAPI(title="DocRag API Engine")

# Mount API routes
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

# Mount Gradio onto a sub-path so it DOES NOT intercept FastAPI's POST routes
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    # 1. Launch Gradio server purely to satisfy ZeroGPU's startup `@spaces.GPU` detection!
    demo.launch(server_name="0.0.0.0", server_port=7861, prevent_thread_lock=True)
    
    # 2. Run the actual unified FastAPI app on the main exposed port (7860)
    uvicorn.run(app, host="0.0.0.0", port=7860)
