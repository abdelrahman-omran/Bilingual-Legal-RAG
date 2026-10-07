from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, field_validator

from legal_rag.core.rag import LegalRAGPipeline

app = FastAPI(title="Bilingual Legal RAG API", version="0.1.0")

# Initialize RAG Pipeline globally
rag_pipeline = LegalRAGPipeline()


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty.")
        return v


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns API health and the total number of indexed documents."""
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
    try:
        # Pydantic automatically handles the 422 Unprocessable Entity for empty strings
        result = rag_pipeline.answer(query=request.question, top_k=3)

        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
