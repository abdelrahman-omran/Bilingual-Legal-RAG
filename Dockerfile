FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Needed by Docker Compose healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --format requirements-txt --no-dev -o requirements.txt && \
    uv pip install --system -r requirements.txt

# Download BGE-M3 into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY src/ ./src/
COPY data/qdrant_db/ ./data/qdrant_db/

EXPOSE 8000

CMD ["uvicorn", "legal_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]