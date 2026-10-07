FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Copy fast uv binary directly from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./

# Install CPU dependencies directly using uv sync
# Passing PyTorch CPU index directly avoids hash mismatches from exported requirement files
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --system \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match

COPY src/ ./src/

# Pre-download BGE-M3 directly into image layers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy local qdrant database
COPY data/qdrant_db/ ./data/qdrant_db/

EXPOSE 8000

CMD ["uvicorn", "legal_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]