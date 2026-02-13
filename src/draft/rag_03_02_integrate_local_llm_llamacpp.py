"""
Integrate local LLM Qwen3-1.7B-GGUF via llama.cpp.

This module supports two usage modes:
  - Method "direct": llama-cpp-python (loads GGUF in-process)
  - Method "server": HTTP client to llama-server (OpenAI-compatible API)

-------------------------------------------------------------------------------
SETUP STEPS (prerequisites before running this script)
-------------------------------------------------------------------------------

Step 1: Build llama.cpp (CMake, NOT make)
-----------------------------------------
The legacy Makefile is deprecated. Use CMake:

  cd IC-RAG-Agent/libs/llama.cpp
  cmake -B build
  cmake --build build --config Release

On macOS with Metal acceleration:
  cmake -B build -DGGML_METAL=ON
  cmake --build build --config Release -j

Step 2: Locate your Qwen3-1.7B-GGUF model
-----------------------------------------
Place your downloaded GGUF file in one of these locations (or set via env):
  - IC-RAG-Agent/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf
  - ~/.cache/llama.cpp/
  - Any path you prefer

Example filenames: Qwen3-1.7B-Q8_0.gguf

Step 3: Run the model
----------------------
Option A - CLI (quick test):
  ./libs/llama.cpp/build/bin/llama-cli -m ./models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf -p "Give me a short introduction to large language model." -n 256

Option B - HTTP server (for RAG / OpenAI-compatible API):
  ./libs/llama.cpp/build/bin/llama-server -m ./models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf --host 127.0.0.1 --port 8080

  - Web UI: http://127.0.0.1:8080
  - API base: http://127.0.0.1:8080/v1

Option C - Download from Hugging Face (if not downloaded):
  ./libs/llama.cpp/build/bin/llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q8_0

Step 4: Python dependencies
---------------------------
For method "direct": pip install llama-cpp-python
For method "server": pip install openai  (OpenAI-compatible client)

Note: On some Macs (x86 + Metal), llama-cpp-python may fail with "Failed to create
llama_context" due to Metal shader compilation. Use method "server" instead: start
llama-server (Option B above), then run this script with --method server.

-------------------------------------------------------------------------------
"""

import os
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# Default config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-1.7B-GGUF"
DEFAULT_GGUF_FILE = "Qwen3-1.7B-Instruct-Q4_K_M.gguf"
DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MAX_TOKENS = 256
DEFAULT_PROMPT = "Give me a short introduction to large language model."


def _resolve_gguf_path() -> Path | None:
    """
    Resolve the GGUF model path from env or default locations.
    Returns Path to .gguf file if found, else None.
    """
    env_path = os.environ.get("QWEN3_GGUF_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    # Check default project models dir
    model_dir = Path(os.environ.get("QWEN3_GGUF_DIR", str(DEFAULT_MODEL_PATH)))
    if model_dir.is_dir():
        for f in model_dir.glob("*.gguf"):
            return f
        candidate = model_dir / DEFAULT_GGUF_FILE
        if candidate.exists():
            return candidate
    return None


def run_direct(model_path: str | Path | None = None, prompt: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """
    Run inference via llama-cpp-python (loads GGUF in-process).

    Requires: pip install llama-cpp-python

    Args:
        model_path: Path to GGUF file. If None, uses _resolve_gguf_path().
        prompt: User prompt text.
        max_tokens: Max tokens to generate.

    Returns:
        Generated text.
    """
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python not installed. Run: pip install llama-cpp-python"
        ) from e

    path = model_path or _resolve_gguf_path()
    if not path:
        raise FileNotFoundError(
            "GGUF model not found. Set QWEN3_GGUF_PATH or place model in "
            f"{DEFAULT_MODEL_PATH}/"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    prompt = prompt or DEFAULT_PROMPT
    t0 = time.time()
    # n_ctx: 512 to minimize RAM; raise if you have more memory.
    # n_gpu_layers: 0=CPU (try -1 for Metal if GPU fails).
    # use_mlock=False: avoid locking RAM (helps on memory-constrained systems).
    llm = Llama(
        model_path=str(path),
        n_ctx=512,
        n_gpu_layers=0,
        n_batch=256,
        use_mlock=False,
        verbose=False,
    )
    t1 = time.time()
    print(f"[1] Model load: {t1 - t0:.2f}s")

    out = llm(prompt, max_tokens=max_tokens, temperature=0.3)
    t2 = time.time()
    print(f"[2] Generation: {t2 - t1:.2f}s")

    text = out["choices"][0]["text"].strip()
    print(f"Total: {t2 - t0:.2f}s")
    return text


def run_server(
    base_url: str = DEFAULT_SERVER_URL,
    prompt: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Run inference via llama-server HTTP API (OpenAI-compatible).

    Requires: llama-server running, and pip install openai

    Args:
        base_url: Base URL of llama-server (e.g. http://127.0.0.1:8080/v1).
        prompt: User prompt text.
        max_tokens: Max tokens to generate.

    Returns:
        Generated text.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai client not installed. Run: pip install openai") from e

    prompt = prompt or DEFAULT_PROMPT
    client = OpenAI(base_url=base_url, api_key="lm-studio")  # api_key can be dummy for local

    t0 = time.time()
    resp = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    t1 = time.time()
    print(f"[1] HTTP request: {t1 - t0:.2f}s")

    text = resp.choices[0].message.content or ""
    print(f"Total: {t1 - t0:.2f}s")
    return text


def main(
    method: str = "direct",
    model_path: str | Path | None = None,
    server_url: str | None = None,
    prompt: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> None:
    """
    Run local LLM inference via llama.cpp.

    Args:
        method: "direct" (llama-cpp-python) or "server" (HTTP to llama-server).
        model_path: Path to GGUF file (for direct mode).
        server_url: llama-server base URL (for server mode).
        prompt: User prompt text.
        max_tokens: Max tokens to generate.
    """
    prompt = prompt or DEFAULT_PROMPT

    if method == "direct":
        print("\n--- Method: direct (llama-cpp-python) ---")
        out = run_direct(model_path=model_path, prompt=prompt, max_tokens=max_tokens)
    elif method == "server":
        print("\n--- Method: server (HTTP to llama-server) ---")
        out = run_server(
            base_url=server_url or DEFAULT_SERVER_URL,
            prompt=prompt,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError("method must be 'direct' or 'server'")

    print("output:", out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Qwen3-1.7B-GGUF via llama.cpp")
    parser.add_argument(
        "--method",
        choices=["direct", "server"],
        default="direct",
        help="direct=llama-cpp-python, server=HTTP to llama-server",
    )
    parser.add_argument("--model-path", type=str, help="Path to GGUF file (direct mode)")
    parser.add_argument(
        "--server-url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help="llama-server base URL (server mode)",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)

    args = parser.parse_args()
    main(
        method=args.method,
        model_path=args.model_path,
        server_url=args.server_url,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
