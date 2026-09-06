"""
Model downloader for Sheprd.
Provides secure, streaming downloads for curated starter models and Hugging Face GGUF repositories.
"""

import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .security import GGUF_MAGIC, SecurityError


DEFAULT_MODELS_DIR = Path.home() / ".local/share/sheprd/models"

# Curated starter models optimized for quick setup and local performance
STARTER_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen2.5-0.5b": {
        "name": "Qwen 2.5 0.5B Instruct (Q4_K_M)",
        "arch": "qwen2",
        "size_mb": 469,
        "context": 32768,
        "description": "Blazing fast 0.5B model, perfect for general tasks and rapid responses.",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    },
    "smollm2-135m": {
        "name": "SmolLM2 135M Instruct (Q4_K_M)",
        "arch": "llama",
        "size_mb": 85,
        "context": 8192,
        "description": "Ultra-lightweight 135M model for testing and resource-constrained environments.",
        "url": "https://huggingface.co/prithivMLmods/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct.Q4_K_M.gguf",
        "filename": "smollm2-135m-instruct-q4_k_m.gguf",
    },
    "llama-3.2-1b": {
        "name": "Llama 3.2 1B Instruct (Q4_K_M)",
        "arch": "llama",
        "size_mb": 780,
        "context": 131072,
        "description": "State-of-the-art lightweight general model from Meta with 128k context.",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
    },
    "qwen2.5-coder-1.5b": {
        "name": "Qwen 2.5 Coder 1.5B Instruct (Q4_K_M)",
        "arch": "qwen2",
        "size_mb": 1100,
        "context": 32768,
        "description": "Dedicated lightweight coding specialist for refactoring and code review.",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    },
}


def list_starter_models() -> Dict[str, Dict[str, Any]]:
    return STARTER_MODELS


def download_model(
    preset_or_url: str,
    dest_dir: Optional[Path] = None,
    progress_hook: Optional[Callable[[int, int, float], None]] = None,
) -> Path:
    """
    Downloads a GGUF model with progress reporting, verifies the header signature,
    and returns the local file path.
    """
    target_dir = dest_dir or DEFAULT_MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    key = preset_or_url.strip().lower()
    if key in STARTER_MODELS:
        info = STARTER_MODELS[key]
        url = info["url"]
        filename = info["filename"]
    else:
        url = preset_or_url.strip()
        if not (url.startswith("https://") or url.startswith("http://")):
            raise ValueError(f"Invalid URL or unknown starter model key '{preset_or_url}'.")
        # Extract filename from URL
        filename = url.split("?")[0].split("/")[-1]
        if not filename.endswith(".gguf"):
            filename += ".gguf"

    dest_file = target_dir / filename
    if dest_file.exists() and dest_file.stat().st_size > 1024 * 1024:
        # Check header
        with open(dest_file, "rb") as f:
            if f.read(4) == GGUF_MAGIC:
                return dest_file

    temp_file = target_dir / f"{filename}.part"

    headers = {
        "User-Agent": "Sheprd-Downloader/1.0 (Linux; x86_64)",
    }

    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_bytes = int(resp.headers.get("content-length", 0))
            downloaded_bytes = 0

            with open(temp_file, "wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded_bytes += len(chunk)
                    elapsed = max(0.1, time.time() - start_time)
                    speed_mb = (downloaded_bytes / (1024 * 1024)) / elapsed

                    if progress_hook:
                        progress_hook(downloaded_bytes, total_bytes, speed_mb)

        # Verify GGUF signature
        with open(temp_file, "rb") as vf:
            sig = vf.read(4)
            if sig != GGUF_MAGIC:
                temp_file.unlink(missing_ok=True)
                raise SecurityError(
                    f"Downloaded file did not match GGUF magic bytes (got {sig!r}, expected {GGUF_MAGIC!r})."
                )

        # Atomic rename
        temp_file.replace(dest_file)
        dest_file.chmod(0o644)
        return dest_file

    except Exception:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise
