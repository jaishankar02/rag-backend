import gradio as gr
import spaces

# Import FastAPI routes
from app.main import app as fastapi_app

@spaces.GPU(duration=60)
def status_check():
    return "DocRag Enterprise RAG Backend API is Online & Running 🚀"

# Create Gradio Blocks app with GPU-decorated function
with gr.Blocks(title="DocRag RAG Backend Engine") as demo:
    gr.Markdown("# DocRag RAG Backend Engine")
    gr.Markdown("Hybrid Search RAG (Qdrant + Redis BM25 RRF + Cross-Encoder + Qwen GGUF)")
    btn = gr.Button("Check Status")
    output = gr.Textbox(label="Status")
    btn.click(fn=status_check, inputs=[], outputs=output)

# Mount FastAPI endpoints onto Gradio's internal app
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# Use Gradio's launch to start server (required for ZeroGPU detection)
demo.launch(server_name="0.0.0.0", server_port=7860)
