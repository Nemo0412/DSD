"""Minimal vLLM smoke test on this GPU. Confirms model load + generation."""
import os, time


def main():
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model="Qwen/Qwen3-0.6B", gpu_memory_utilization=0.35,
              max_model_len=1024, enforce_eager=True, dtype="bfloat16")
    t1 = time.time()
    out = llm.generate(["The capital of France is"],
                       SamplingParams(temperature=0.0, max_tokens=16))
    t2 = time.time()
    print(f"[smoke] load={t1-t0:.1f}s gen={t2-t1:.2f}s")
    print("[smoke] text:", repr(out[0].outputs[0].text))
    print("[smoke] OK")


if __name__ == "__main__":
    main()
