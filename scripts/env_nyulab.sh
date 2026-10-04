# Source before any run on NYULAB (plain multi-GPU box, no slurm).
# Driver 570 = CUDA 12.8, so the venv holds the vLLM 0.27.1 +cu129 wheel
# (minor-version compatible), not the default CUDA 13 build in requirements/gpu.txt.
export DSD_VENV=/mnt/hdd/external/dsd/venv
export PATH="$DSD_VENV/bin:$PATH"          # flashinfer JIT shells out to bare `ninja`
export DSD_PY="$DSD_VENV/bin/python"
export HF_HOME=/mnt/hdd/hf_cache           # shared cache: Llama-3.1-8B/3.2-1B-Instruct, EAGLE3 heads
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
export TOKENIZERS_PARALLELISM=false
