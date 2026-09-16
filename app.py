import socket

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

# Mount Gradio onto a sub-path so it doesn't intercept FastAPI's POST routes
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")


def find_free_port() -> int:
    """Ask the OS for an unused port instead of hardcoding one that might
    already be held by a stale process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


if __name__ == "__main__":
    # 1. Launch Gradio's own throwaway server purely so ZeroGPU's patched
    #    Blocks.launch() fires and registers the @spaces.GPU function at
    #    startup. This is never meant to serve real traffic, so:
    #      - bind it to 127.0.0.1 only (no need to expose it externally)
    #      - disable SSR mode, which needs a Node subprocess to talk back
    #        to this server and is what's actually failing above
    dummy_port = find_free_port()
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=dummy_port,
            prevent_thread_lock=True,
            show_error=True,
            ssr_mode=False,
        )
    except OSError:
        dummy_port = find_free_port()
        demo.launch(
            server_name="127.0.0.1",
            server_port=dummy_port,
            prevent_thread_lock=True,
            show_error=True,
            ssr_mode=False,
        )

    # 2. Run the actual unified FastAPI app on the main exposed port (7860)
    uvicorn.run(app, host="0.0.0.0", port=7860)