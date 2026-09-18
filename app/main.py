"""FastAPI entry point serving the Jinja frontend and Corrective RAG graph."""

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

api = FastAPI(
    title="Corrective RAG",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
api.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
templates = Jinja2Templates(directory=FRONTEND_DIR)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    answer: str


@lru_cache(maxsize=1)
def get_rag_graph():
    """Build the notebook graph and local index on the first question."""
    from backend.app.graph import app as rag_graph

    return rag_graph


@api.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@api.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    try:
        rag_graph = get_rag_graph()
        result = rag_graph.invoke({"question": payload.question.strip()})
        answer = result["answer"]
        return QueryResponse(answer=getattr(answer, "content", str(answer)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
