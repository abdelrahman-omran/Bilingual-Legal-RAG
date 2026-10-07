from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from legal_rag.core.llm_backends import build_backend_from_env
from legal_rag.core.rag import LegalRAGPipeline


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Legal query in Arabic or English")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


rag_pipeline: LegalRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    rag_pipeline = LegalRAGPipeline(llm_backend=build_backend_from_env())
    yield


app = FastAPI(
    title="Bilingual Legal RAG API",
    description="API for Egyptian Civil Code RAG System",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns API health and the total number of indexed documents."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )
    try:
        collection_info = rag_pipeline.qdrant_client.get_collection(
            rag_pipeline.collection_name
        )
        doc_count = collection_info.points_count
        return HealthResponse(status="healthy", documents_indexed=doc_count)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unavailable: {e!s}",
        )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Answers legal queries using the Egyptian Civil Code corpus."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )
    try:
        result = rag_pipeline.answer(query=request.question, top_k=3)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )