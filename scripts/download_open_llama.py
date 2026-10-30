"""Download the open Llama stand-ins used by the e2e campaign."""
import os

from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HOME", "/scratch/ll5914/.huggingface")
token_path = os.path.join(os.environ["HF_HOME"], "token")
if os.path.exists(token_path):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", open(token_path).read().strip())
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

for repo in ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "openlm-research/open_llama_7b"):
    print("downloading", repo, flush=True)
    path = snapshot_download(repo)
    print("ready", repo, path, flush=True)
