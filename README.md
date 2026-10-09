# Bilingual Legal RAG — Egyptian Civil Code
# NOTE: the system is STILL under testing, some bugs might appear, I don't prefer you to test before I remove this note, contact me if you have any questions
A retrieval-augmented generation system that answers Arabic and English questions
about the **Egyptian Civil Code (Law no. 131 of 1948)** with **article-level
citations**. Final project for the MLOps Practitioner course (LLM/RAG track).

Built for CPU-only environments: the generative model is served by
**llama.cpp** (instructor-approved alternative to vLLM) running a quantized
Qwen2.5-1.5B-Instruct GGUF, and all embeddings run on CPU with bge-m3.

## Why citations matter here

In a legal setting an unverifiable answer is worse than no answer. Every response
returns `sources` as article citations (e.g. `Egyptian Civil Code, Article 147`)
— never chunk IDs or page numbers — so a lawyer can check every claim against
the code itself. Repealed articles (54–80) are flagged in the corpus, not
silently deleted: *"that article no longer exists"* is a valid, verifiable answer.

## Quickstart (3 commands)

Run these three steps from a terminal. Replace `OWNER/REPOSITORY` with the actual
GitHub owner and repository name.

**1. Clone the repository**

```bash
git clone https://github.com/abdelrahman-omran/Bilingual-Legal-RAG.git && cd Bilingual-Legal-RAG
```

**2. Start the services with Docker Compose**

```bash
docker compose up --build -d
```

**3. test connection health**

```bash
curl localhost:8000/health
```


## First-time setup for local development

If you want to run the pipeline outside Docker Compose, install the development
dependencies and fetch the model/data first:

```bash
uv sync
python scripts/download_model.py       # fetch Qwen2.5-1.5B-Instruct GGUF (~1 GB)
dvc pull                               # fetch tracked data (PDF + corpus.json)
```

## Usage
Run these commands at first
```bash
LLM_BACKEND=llamacpp \
LLM_BASE_URL=http://localhost:8080/v1 \
QDRANT_URL=http://localhost:6333 \
LANGFUSE_HOST=http://localhost:3001 \
bentoml serve legal_rag.serving.bentoml_service:LegalRAGService
```

For the BentoML service started directly on the host, the non-streaming endpoint
is `/ask` on port `3000`:

```bash
curl -X POST http://localhost:3000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does Article 147 say about contracts?"}'
```

Streaming (NDJSON: first line = sources, then one JSON object per token):

```bash
curl -N -X POST http://localhost:3000/ask_stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "ما هي حالات بطلان التصرفات القانونية؟"}'
```

Health endpoint for the standalone BentoML service:

```bash
curl http://localhost:3000/health
# Example: {"status": "healthy", "documents_indexed": 1149}
```

Response shape:

```json
{
  "answer": "Under Article 147, the contract makes the law of the parties...",
  "sources": ["Egyptian Civil Code, Article 147"]
}
```

An empty question is rejected with `422` (input validation, covered by tests).

## How it works

1. **Corpus extraction** — `scripts/extract_corpus.py` (DVC stage) converts the
   bilingual two-column PDF into structured JSON: **one record per article** with
   `article_number`, hierarchy metadata (book/chapter/section/topic), Arabic and
   English text, `is_repealed` flag, and a canonical `citation`. Extraction
   handles the column interleaving, Arabic glyph normalization, and Arabic-Indic
   numeral conversion, and is validated by `tests/test_extraction.py` and
   `tests/test_validation.py` (contiguous article numbers, non-empty Arabic text,
   sane record lengths, repealed flags present).
2. **Indexing** — `scripts/build_index.py` embeds each article (Arabic + English
   in one text field) with `BAAI/bge-m3` and upserts it into Qdrant with the
   **article number as the point ID**. One article = one retrievable, citable unit.
3. **Querying** — the question is embedded, top-3 articles are retrieved from
   Qdrant, and a strict prompt (cite articles or admit the answer is not found)
   is sent to the LLM. Retrieval, prompt formatting, and generation are separated
   behind a backend interface (`src/legal_rag/core/llm_backends.py`).

## LLM serving: llama.cpp instead of vLLM

The generative model is **Qwen/Qwen2.5-1.5B-Instruct** served by a
**llama.cpp server** (`llm` container) running the `q4_k_m` GGUF — an
instructor-approved choice for this CPU-only project. vLLM is CUDA-first and
impractical on this hardware; llama.cpp provides an OpenAI-compatible API with
native token streaming at interactive speed on CPU. The backend is swappable via
`LLM_BACKEND=transformers|llamacpp`, and a local Hugging Face `transformers`
fallback is included for offline development and CI.

Model name: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (`qwen2.5-1.5b-instruct-q4_k_m.gguf`,
downloaded by `scripts/download_model.py`).

## Repository layout

```text
├── src/legal_rag/
│   ├── extraction/      # PDF → structured JSON (bilingual, per-article)
│   ├── ingestion/       # embedding + Qdrant indexing (article = one point)
│   ├── core/            # RAG pipeline + pluggable LLM backends
│   ├── api/             # FastAPI app (/ask, /health) — dev & CI
│   └── serving/         # BentoML production service
├── scripts/             # extraction, indexing, experiments, evals, reindex
├── tests/               # pytest suite + eval sets + locustfile
├── deploy/              # nginx canary config
├── data/                # raw PDF, corpus.json, qdrant storage (DVC-tracked inputs)
├── models/              # GGUF model weights (not committed)
├── reports/             # Locust load-test report
└── dvc.yaml              # reproducible corpus pipeline
```

## Data versioning & reproducibility (DVC)

The source PDF and the derived `corpus.json` are DVC-tracked. The full pipeline
reproduces with:

```bash
dvc pull && dvc repro   # PDF → corpus.json → validated, citable articles
```

Batch updates to a live index go through `scripts/reindex.py`, which upserts new
or updated articles **without rebuilding the collection** (tested with a new
document; `documents_indexed` increments accordingly).

## Experiment tracking (MLflow)

`scripts/run_experiments.py` logs chunking/indexing configurations (embedding
model, chunk strategy, overlap) together with RAGAS faithfulness; runs are
compared in the MLflow UI and the best configuration is registered. View with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

RAGAS evaluation results (`scripts/evaluate_ragas.py`) are logged to MLflow as
runs, so metric trends are visible across sessions.

## Evaluation & quality gates

| Check | Where | Threshold |
|---|---|---|
| Retrieval Hit@3 (20 gold bilingual questions) | `scripts/eval_ci.py` (runs in CI) | ≥ 0.75 — CI fails below |
| RAGAS faithfulness | `scripts/evaluate_ragas.py` → MLflow | faithfulness alert < 0.80 (`scripts/check_faithfulness_alret.py`) |

The bilingual eval set doubles as an Arabic-embedding ablation: the same
question is asked in both languages and retrieval must return the same article.

## Production serving (BentoML) & load testing

The BentoML service wraps the pipeline with async `/ask` and streaming
`/ask_stream` endpoints:

```bash
uv build                                  # build the project wheel
UV_INDEX_STRATEGY=unsafe-best-match \
bentoml build                              # build the bento
bentoml containerize legal_rag_service:<tag> --image-tag legal-rag-bento:latest
```

Load test (50 concurrent users, Locust):

```bash
locust -f tests/locustfile.py --host http://localhost:3000 \
  --users 50 --spawn-rate 5 --run-time 3m --headless \
  --html reports/locust_report.html
```

**p95 latency for `/ask` at 50 concurrent users: see `reports/locust_report.html`.**
(The llama.cpp server runs with `--parallel` slots so concurrent requests are
queued server-side; on CPU, latency is dominated by token generation.)

## Canary rollout

`docker-compose.canary.yaml` + `deploy/nginx.canary.conf` split traffic ~90/10
between `legal-rag-bento:stable` and `legal-rag-bento:canary`. The
`X-Served-By` response header reveals which backend served a request, so the
split is verifiable with a simple curl loop. Rollback is immediate (stopping the
canary container takes it out of rotation); promotion is a re-tag of the image.

## Observability

- **Langfuse** tracing is integrated in the pipeline (every request can produce
  a trace with retrieval and generation spans; see `test_langfuse.py`), with
  RAGAS faithfulness scores attachable to traces during evaluation runs.
  Requires a Langfuse instance and `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` /
  `LANGFUSE_SECRET_KEY` environment variables.
- **Alerting**: `scripts/check_faithfulness_alret.py` reads the latest RAGAS
  run from MLflow and exits non-zero (and posts to `ALERT_WEBHOOK_URL` if set)
  when mean faithfulness drops below 0.80.

## CI/CD

GitHub Actions runs lint → test → `dvc repro` (rebuild index) → Docker image
build & push on merge. The pipeline fails if the retrieval quality gate
(Hit@3 ≥ 0.75 on the 20-question set) does not pass.

## Development

```bash
uv sync --extra dev
pytest            # unit + integration tests (API, extraction, validation, indexing)
ruff check .
```

Run tests **without** `QDRANT_URL` exported — tests use an isolated local Qdrant
storage per test session.

## Design notes & known limitations

- **Judge-model leniency**: RAGAS metrics use the same 1.5B model family as the
  generator (both served via llama.cpp), so faithfulness scores may run high;
  they are treated as a *relative trend* across sessions rather than absolute truth.
- **CPU ceiling**: throughput is bounded by llama.cpp token generation on CPU;
  the architecture (LLM as a sidecar behind an OpenAI-compatible API) is exactly
  what would let you swap in vLLM/GPU serving later without touching the RAG code.
- Embedding drift detection (cosine/MMD on query distributions) is a natural
  extension given the bilingual corpus — noted for future work.
