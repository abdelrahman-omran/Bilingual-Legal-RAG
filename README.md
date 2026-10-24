# Bilingual Legal RAG — Egyptian Civil Code
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

# Quickstart

## Clone the repository

This is the initial step for a fresh machine.

```bash
git clone https://github.com/abdelrahman-omran/Bilingual-Legal-RAG.git
cd Bilingual-Legal-RAG
```

## Set up the system  "3 commands to run on any machine" 

Choose **one** of the following options to prepare the data and start the system.

### Option 1: Build the corpus and index from scratch (`dvc repro`)

Use this option when you want to reproduce the data-processing and indexing pipeline yourself.

First, install the project dependencies and activate the virtual environment:

```bash
uv sync
source .venv/bin/activate
```

Run the DVC pipeline:

```bash
dvc repro
```

**Note:** On the first run, `dvc repro` downloads the `BAAI/bge-m3` embedding model (approximately **2.27 GB**) to generate embeddings and build the Qdrant index. The model is cached locally for subsequent runs.

Start the application and supporting services:

```bash
docker compose up --build -d
```

The first Docker startup may download the embedding model again because the API uses its own persistent Docker cache volume. Subsequent restarts and image rebuilds normally reuse this volume.

### Option 2: Download the prebuilt data using DVC (`dvc pull`) "Needs Access Token"

Use this option when you want to avoid rebuilding the corpus and index. The prebuilt artifacts are stored on DagsHub, so you need a DagsHub access token with permission to download them.

First, install the project dependencies and activate the virtual environment:

```bash
uv sync
source .venv/bin/activate
```

Configure your DagsHub credentials locally. Run the following commands and paste the access token in the terminal:

```bash
uv run dvc remote modify --local dagshub-public auth basic
uv run dvc remote modify --local dagshub-public user 11abdelrahmanhamdy11

read -rsp "DagsHub token: " DAGSHUB_TOKEN
echo
uv run dvc remote modify --local dagshub-public password "$DAGSHUB_TOKEN"
uv run dvc remote modify --local dagshub-public ask_password false
unset DAGSHUB_TOKEN

uv run dvc pull
```

This restores the versioned artifacts without running the extraction and indexing pipeline, so you can skip the initial embedding-generation step.

**Important:** The API still needs the `BAAI/bge-m3` embedding model at runtime to turn user questions into vectors for retrieval. 

Start the application and supporting services:

```bash
docker compose up --build -d
```

## Check the application health

Once the services have started, check the API:

```bash
curl http://localhost:8000/health
```

The health endpoint relies on the indexed data being available. If you choose Option 1, complete `dvc repro` first. If you choose Option 2, make sure `dvc pull` succeeds before starting the application.

You can also test the `/ask` endpoint by running:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Article 147 say about contracts?"}'
```

The first question may take a few minutes because the models need to be downloaded and saved in a Docker volume. After that, requests should take only seconds. You may not see the model download logs, so just wait a few minutes for the first answer to arrive.

## Which option should I choose?

- **Option 1 — `dvc repro`:** Best for reproducing the complete data-processing and indexing pipeline.
- **Option 2 — `dvc pull`:** Best for quickly setting up the application using the prebuilt artifacts. Requires a DagsHub access token.

Both options use the same Docker Compose deployment after the data is ready.


## Choose an LLM serving method

Choose one of the three backends below. The Hugging Face `transformers` backend
is the default when you do not select `llamacpp` or `vllm`, but the examples set
`LLM_BACKEND` explicitly so the selected backend is clear.

You don't need a fresh setup each time to be able to use each of the 3 models, you can switch between them using the commands below:

### Option 1: Local llama.cpp — Qwen2.5-1.5B-Instruct-GGUF

Download the GGUF model, then start the `llm` service:

```bash
python scripts/download_model.py
docker compose up -d llm
```

Configure BentoML to use the local llama.cpp OpenAI-compatible endpoint:

```bash
export LLM_BACKEND=llamacpp
export LLM_BASE_URL=http://localhost:8080/v1
export QDRANT_URL=http://localhost:6333

# Inspect the model exposed by llama.cpp
curl -s http://localhost:8080/v1/models
```
### Option 2: vLLM on Colab — `qwen2.5-3b-awq`

The `qwen2.5-3b-awq` model was quantized on Colab. Start the vLLM server by
running the notebook at `notebooks/vllm_serving/vLLM_ngrok_HF.ipynb` in Colab.
Copy the current ngrok URL and API key from that setup into the environment
variables below. Qdrant still needs to be running locally.

```bash
export QDRANT_URL=http://localhost:6333
export LLM_BACKEND=vllm
export LLM_BASE_URL=https://YOUR-CURRENT-NGROK-HOST.ngrok-free.app/v1
export LLM_MODEL_NAME=qwen2.5-3b-awq
export LLM_API_KEY=YOUR_VLLM_API_KEY
```

Replace the ngrok URL and API key placeholders with the current values from your
Colab session; the ngrok URL may change between sessions.

### Option 3: Hugging Face Transformers — Qwen2.5-1.5B-Instruct (Backup model)

This runs the model locally through Hugging Face Transformers. The model is
downloaded from Hugging Face the first time it is needed.
It acts as a backup model if one of the two previous models failed to run

```bash
source .venv/bin/activate

export QDRANT_URL=http://localhost:6333
export LLM_BACKEND=transformers
export LLM_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
unset LLM_BASE_URL
unset LLM_API_KEY
```



### Monitoring and observability

Start the Docker Compose services that you want to use. For the full set of
services defined in the Compose configuration, run:

```bash
docker compose up -d
```

### Langfuse

Create or obtain the keys for your Langfuse project, then export the real values
(the examples below are placeholders, not working credentials):

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-YOUR_PUBLIC_KEY
export LANGFUSE_SECRET_KEY=sk-lf-YOUR_SECRET_KEY
export LANGFUSE_HOST=http://localhost:3001
```

Open [http://localhost:3001](http://localhost:3001) to access Langfuse.

### Prometheus, Alertmanager, and Grafana

After the relevant Compose services are running, open:

- Prometheus: [http://localhost:9090](http://localhost:9090)
- Alertmanager: [http://localhost:9093](http://localhost:9093)
- Grafana: [http://localhost:3002](http://localhost:3002)

### Experiments and RAGAS faithfulness

### MLflow

Start the MLflow UI with the local SQLite tracking database:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

### Enable Real-Time RAGAS Evaluation
By default, request-level evaluations are disabled to maintain low latency. You can optionally enable **RAGAS Faithfulness Scoring** to automatically judge response quality and log evaluation metrics to MLflow on every incoming request.

#### When to Enable This:
* **Development & Auditing**: To track answer grounding and hallucination metrics in real time.
* **Experimentation**: To log detailed evaluation runs to your local MLflow tracking database.

> **Note**: Running RAGAS evaluations during active requests introduces additional processing latency and requires an accessible judge LLM endpoint.

#### Configuration Variables
Set the following environment variables before launching the BentoML service:
```bash
docker compose up -d llm
export RAGAS_ENABLED=true
export FAITHFULNESS_JUDGE_BASE_URL=http://localhost:8080/v1
export RAGAS_TIMEOUT_SECONDS=300
export RAGAS_MAX_CONTEXT_CHARS=2400
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export RAGAS_MLFLOW_EXPERIMENT=ragas_evaluation
```


- `FAITHFULNESS_JUDGE_BASE_URL` points to the local llama.cpp endpoint in this example. All three LLM options can retrieve and generate answers, but currently, due to versioning issues with other models, only local llama.cpp can be used as an LLM judge for RAGAS. To use RAGAS, you need to set up llama.cpp as described in **Option 1**. You can still use the other models to generate answers by setting them up after setting up llama.cpp for ragas.

## Usage
start the BentoML serve
```bash
bentoml serve legal_rag.serving.bentoml_service:LegalRAGService --host 0.0.0.0 --port 3000
```

For the BentoML service started directly on the host, the non-streaming endpoint
is `/ask` on port `3000`:

```bash
curl -X POST http://localhost:3000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does Article 147 say about contracts?"}'
```

Response shape:

```json
{
  "answer": "Under Article 147, the contract makes the law of the parties...",
  "sources": ["Egyptian Civil Code, Article 147"]
}
```

Streaming:

```bash
curl -N -X POST http://localhost:3000/ask_stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does Article 147 say about contracts?"}'
```

Health endpoint for the standalone BentoML service:

```bash
curl -X POST http://localhost:3000/health
# Example: {"status": "healthy", "documents_indexed": 1149}
```




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
## System Architecture

![system architecture](assets/system_architecture.png)

**Notes:** FastAPI and BentoML are separate serving interfaces that route requests through the RAG pipeline. `QDRANT_URL` must target the same Qdrant instance for indexing and runtime retrieval. The embedding model may download on first use if it is not cached. The LLM backend is selected through `LLM_BACKEND`; the code defaults to Transformers when that variable is unset.

For the editable diagram source, see [`SYSTEM_ARCHITECTURE.mmd`](SYSTEM_ARCHITECTURE.mmd).
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
| RAGAS faithfulness | `scripts/evaluate_ragas.py` → MLflow | faithfulness alert for CI < 0.80 (`scripts/check_faithfulness_alret.py`) |


## load testing

Load test (50 concurrent users, Locust):

```bash
locust -f tests/locustfile.py --host http://localhost:3000 \
  --users 50 --spawn-rate 5 --run-time 3m --headless \
  --html reports/locust_report.html
```

**reports saved in `reports/locust_report.html`.**
(The llama.cpp server runs with `--parallel` slots so concurrent requests are
queued server-side; on CPU, latency is dominated by token generation.)

## Canary rollout

`docker-compose.canary.yaml` + `deploy/nginx.canary.conf` split traffic ~95/5
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
