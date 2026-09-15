from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import uuid

from app.services.pdf_ingestion import ingest_pdf

upload_router = APIRouter(prefix="/upload", tags=["upload"])


@upload_router.post("/file")
async def upload_pdf(file: UploadFile = File(...)):
    """Save uploaded PDF to backend/uploads, generate document_id, call ingestion.

    Minimal implementation as requested — does not perform validation beyond
    saving and calling `ingest_pdf`.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # compute backend root (three levels up from this file: backend/)
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    uploads_dir = os.path.join(backend_root, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    ext = os.path.splitext(file.filename)[1] or ".pdf"
    document_id = uuid.uuid4().hex
    dest_path = os.path.join(uploads_dir, f"{document_id}{ext}")

    # write uploaded file to disk
    with open(dest_path, "wb") as out_f:
        content = await file.read()
        out_f.write(content)

    # call ingestion (synchronous)
    result = ingest_pdf(dest_path, document_id)

    return result
