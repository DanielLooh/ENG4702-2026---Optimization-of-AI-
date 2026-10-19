"""Shared helpers for the local Ollama resource experiments."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# This is the fixed, basic prompt specified by the initial paper.  Keeping it
# short and fixing the sampling options makes the three-model comparison cheap
# and repeatable.
DEFAULT_PROMPT = """You are an expert software engineer. Provide a Python solution only.
Solve this: Given an integer array nums and an integer target, return the indices of two numbers such that they add up to target. Assume exactly one solution exists. Use an efficient algorithm."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def call_ollama(model: str, prompt: str, *, keep_alive: str, num_predict: int,
                num_ctx: int, seed: int) -> dict[str, Any]:
    """Make one non-streaming, deterministic local Ollama generation."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=1800) as response:
        return json.loads(response.read().decode("utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def wait(seconds: float) -> None:
    """A named wrapper makes timing intent clear at call sites."""
    time.sleep(seconds)
