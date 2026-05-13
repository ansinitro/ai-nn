"""
FastAPI server — serves React frontend + RAG API.
Run: source rag_env/bin/activate && python server.py
Tunnel: cloudflared tunnel --url http://localhost:8000
"""
import sys, time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

# ─── Global pipeline ─────────────────────────────────────────
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading RAG pipeline...")
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    print("RAG pipeline ready!")
    yield
    print("Shutting down...")


app = FastAPI(title="RAG QA API", lifespan=lifespan)


# ─── API Routes ───────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceItem(BaseModel):
    doc_name: str
    score: float
    text_preview: str
    full_text: str
    used_in_context: bool


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    retrieval_time_ms: float
    generation_time_s: float
    num_chunks: int


@app.get("/api/status")
def status():
    if pipeline is None:
        return {"status": "loading"}
    return {
        "status": "ready",
        "chunks": pipeline.retriever.index.ntotal,
        "llm": pipeline.llm.backend_name,
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = pipeline.ask(req.question, top_k=req.top_k)
        return AskResponse(
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
            retrieval_time_ms=result["retrieval_time"] * 1000,
            generation_time_s=result["generation_time"],
            num_chunks=result["num_chunks_used"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Serve React SPA ──────────────────────────────────────────
DIST = Path(__file__).parent / "frontend" / "dist"
PAPERS = Path(__file__).parent / "data" / "papers"


@app.get("/papers/{filename}")
def serve_paper(filename: str):
    """Serve a PDF paper file directly in the browser."""
    paper_path = PAPERS / filename
    if not paper_path.exists() or not paper_path.suffix.lower() == ".pdf":
        raise HTTPException(status_code=404, detail="Paper not found")
    return FileResponse(
        path=paper_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        return FileResponse(DIST / "index.html")
else:
    @app.get("/")
    def root():
        return {"message": "Frontend not built yet. Run: cd frontend && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
