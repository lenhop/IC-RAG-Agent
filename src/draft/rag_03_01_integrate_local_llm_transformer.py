"""
Integrate local LLM via Hugging Face Transformers (Qwen3, etc.).

Two implementation methods:
  - Method "direct": model.generate() flow (tokenize -> generate -> decode)
  - Method "pipeline": HuggingFace pipeline, can wrap for LangChain RAG-Agent

Note: Transformers loads raw weights (PyTorch, SafeTensors) or HuggingFace Hub models.
      GGUF format is NOT supported natively; use llama-cpp-python or convert GGUF first.
"""

import time
import torch


def _patch_torch_autocast_for_older_pytorch() -> None:
    """Patch torch.is_autocast_enabled for PyTorch < 2.3 (transformers 4.57+ compatibility)."""
    if getattr(torch.is_autocast_enabled, "_qwen_patched", False):
        return
    try:
        torch.is_autocast_enabled("cpu")
    except TypeError:
        original = torch.is_autocast_enabled

        def _patched(device_type=None):
            return original()

        _patched._qwen_patched = True
        torch.is_autocast_enabled = _patched


_patch_torch_autocast_for_older_pytorch()

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -----------------------------------------------------------------------------
# Default config
# -----------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "/Users/hzz/KMS/IC-RAG-Agent/models/Qwen3-1.7B"
DEFAULT_DEVICE_MAP = "auto"  # "cpu" for CPU-only; "auto" for GPU if available
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_PROMPT = "Give me a short introduction to large language model."


def main(
    method: str = "direct",
    model_path: str | None = None,
    device_map: str | None = None,
    max_new_tokens: int | None = None,
    prompt: str | None = None,
) -> None:
    """
    Run local LLM inference.

    Args:
        method: "direct" | "pipeline" | "both"
        model_path: Path to model weights
        device_map: "cpu" or "auto"
        max_new_tokens: Max tokens to generate
        prompt: User prompt text
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    device_map = device_map or DEFAULT_DEVICE_MAP
    max_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS
    prompt = prompt or DEFAULT_PROMPT

    # Shared setup: load tokenizer and model
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    t1 = time.time()
    print(f"[1] Tokenizer loading: {t1 - t0:.2f}s")

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", **load_kwargs)
    t2 = time.time()
    print(f"[2] Model loading: {t2 - t1:.2f}s")

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    t3 = time.time()
    print(f"[3] Chat template: {t3 - t2:.2f}s")

    def _run_direct(t_start: float) -> str:
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        t4 = time.time()
        print(f"  [direct] Tokenization: {t4 - t_start:.2f}s")

        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        t5 = time.time()
        print(f"  [direct] Generation: {t5 - t4:.2f}s")

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        t6 = time.time()
        print(f"  [direct] Decode: {t6 - t5:.2f}s")
        return output

    def _run_pipeline(t_start: float) -> str:
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
        )
        t4 = time.time()
        print(f"  [pipeline] Pipeline build: {t4 - t_start:.2f}s")

        result = llm_pipeline(text)
        t5 = time.time()
        print(f"  [pipeline] Generation: {t5 - t4:.2f}s")

        output = result[0]["generated_text"]

        try:
            from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

            lc_llm = HuggingFacePipeline(pipeline=llm_pipeline)
        except ImportError:
            try:
                from langchain.llms import HuggingFacePipeline

                lc_llm = HuggingFacePipeline(pipeline=llm_pipeline)
            except ImportError:
                lc_llm = None
        t6 = time.time()
        print(f"  [pipeline] LangChain wrap: {t6 - t5:.2f}s")
        return output

    total_start = t0
    if method in ("direct", "both"):
        print("\n--- Method: direct ---")
        out_direct = _run_direct(t3)
        print("output (direct):", out_direct)

    if method in ("pipeline", "both"):
        print("\n--- Method: pipeline ---")
        out_pipeline = _run_pipeline(t3)
        print("output (pipeline):", out_pipeline)

    t_end = time.time()
    print(f"\nTotal: {t_end - total_start:.2f}s")


if __name__ == "__main__":
    main(method="direct")
