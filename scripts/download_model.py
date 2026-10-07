from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=MODEL_DIR,
    )

    print(f"Model available at: {Path(path).resolve()}")