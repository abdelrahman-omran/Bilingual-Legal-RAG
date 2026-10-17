# Changes Report

## Serving

| Requested | Implemented |
|---|---|
| Serve the model with local vLLM | **1. llama.cpp (local):** Serves the model `Qwen/Qwen2.5-1.5B-Instruct-GGUF` model making local CPU serving practical without requiring a GPU-focused vLLM setup.
||**2. vLLM (remote GPU):** Serves the quantized `qwen2.5-3b-awq` model (I quantized it on colab) to serve it from Colab and exposes it through ngrok.
||**3. Hugging Face Transformers (backup):** Runs `Qwen/Qwen2.5-1.5B-Instruct` locally if the other serving options are unavailable or fail. |

**Benefit:** This gives users serving options that fit the capabilities they have: llama.cpp for local CPU use or vLLM for a remote GPU. Using llama.cpp instead of local vLLM also reduces Dockerization complexity and resource demands. Hugging Face Transformers is available as a fallback, so the system is not tied to a single serving method.

## Smaller Changes to Reduce CPU Load

| Requested | Implemented | Reason / Effect |
|---|---|---|
| Locust load test with 50 concurrent users | 10 concurrent users for 1 minute | Preserves the load-testing purpose while reducing pressure on the CPU. |
| RAGAS evaluation with 50 questions | 20 questions | Preserves the faithfulness-evaluation purpose while reducing runtime and CPU load. |

These changes keep the same overall testing and evaluation goals, but use smaller workloads to avoid placing excessive pressure on the available CPU.
