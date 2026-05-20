#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


class LocalLlamaRuntime:
    def __init__(self) -> None:
        self._llm = None
        self._model_name = os.getenv("PLUME_LOCAL_LLM_MODEL_NAME")

    @property
    def model_name(self) -> str:
        if self._model_name:
            return self._model_name
        gguf_path = os.getenv(
            "PLUME_LOCAL_LLM_GGUF_PATH",
            "/workspace/llm_runtime/models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf",
        )
        return Path(gguf_path).name

    def ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama

        gguf_path = Path(os.getenv(
            "PLUME_LOCAL_LLM_GGUF_PATH",
            "/workspace/llm_runtime/models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf",
        )).expanduser()
        n_ctx = int(os.getenv("PLUME_LOCAL_LLM_N_CTX", "4096"))
        n_gpu_layers = int(os.getenv("PLUME_LOCAL_LLM_N_GPU_LAYERS", "-1"))
        n_threads = int(os.getenv("PLUME_LOCAL_LLM_N_THREADS", str(os.cpu_count() or 1)))
        n_batch = int(os.getenv("PLUME_LOCAL_LLM_N_BATCH", "512"))
        chat_format = os.getenv("PLUME_LOCAL_LLM_CHAT_FORMAT", "chatml")
        verbose = os.getenv("PLUME_LOCAL_LLM_VERBOSE", "false").lower() in {"1", "true", "yes", "on"}
        self._llm = Llama(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=n_batch,
            chat_format=chat_format,
            verbose=verbose,
        )

    def chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> str:
        self.ensure_loaded()
        output = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        content = (((output or {}).get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if not isinstance(content, str):
            content = str(content or "")
        return content.strip()


def main() -> int:
    runtime = LocalLlamaRuntime()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        started = time.perf_counter()
        request_id = "unknown"
        try:
            payload = json.loads(line)
            request_id = str(payload.get("request_id") or "unknown")
            messages = payload.get("messages") or []
            max_tokens = int(payload.get("max_tokens") or os.getenv("PLUME_LOCAL_LLM_MAX_TOKENS", "300"))
            temperature = float(payload.get("temperature") or os.getenv("PLUME_LOCAL_LLM_TEMPERATURE", "0.1"))
            top_p = float(payload.get("top_p") or os.getenv("PLUME_LOCAL_LLM_TOP_P", "0.9"))
            content = runtime.chat(messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            elapsed = time.perf_counter() - started
            if not content:
                response: dict[str, Any] = {
                    "request_id": request_id,
                    "ok": False,
                    "error": "local GGUF produced empty output",
                    "model": runtime.model_name,
                    "elapsed_seconds": round(elapsed, 4),
                }
            else:
                response = {
                    "request_id": request_id,
                    "ok": True,
                    "content": content,
                    "model": runtime.model_name,
                    "elapsed_seconds": round(elapsed, 4),
                }
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"[local-llm-worker] request failed: {exc}", file=sys.stderr, flush=True)
            response = {
                "request_id": request_id,
                "ok": False,
                "error": str(exc),
                "model": runtime.model_name,
                "elapsed_seconds": round(elapsed, 4),
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
