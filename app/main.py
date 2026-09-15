from fastapi import FastAPI

from app.api.routes.upload import  upload_router
from app.api.routes.query import router as query_router
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="AI Knowledge Base")

app.include_router(upload_router)
app.include_router(query_router)

@app.get("/")
async def root() -> dict[str, str]:
	return {"message": "FastAPI server is running"}


@app.get("/health")
async def health_check() -> dict[str, str]:
	return {"status": "ok"}
