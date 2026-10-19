#!/usr/bin/env python3
"""Experiment 1: sample the local Ollama process while one model generates code."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import statistics
import threading
import time
from pathlib import Path

import psutil

from common import DEFAULT_PROMPT, call_ollama, utc_now, write_json


def ollama_processes() -> list[psutil.Process]:
    processes = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            text = " ".join(proc.info.get("cmdline") or [])
            if "ollama" in (proc.info.get("name") or "").lower() or "ollama" in text.lower():
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes


def sample_processes() -> tuple[float, float]:
    """Return aggregate CPU percentage and RSS MiB for all Ollama processes."""
    cpu = rss = 0.0
    for proc in ollama_processes():
        try:
            cpu += proc.cpu_percent(None)
            rss += proc.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return cpu, rss


def sample_apple_gpu() -> tuple[float | None, float | None]:
    """Read Apple GPU device utilisation and unified GPU memory from IOKit."""
    try:
        out = subprocess.check_output(
            ["ioreg", "-l", "-w0", "-r", "-c", "AGXAccelerator"], text=True, stderr=subprocess.DEVNULL
        )
        util = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
        memory = re.search(r'"In use system memory"\s*=\s*(\d+)', out)
        return (float(util.group(1)) if util else None,
                float(memory.group(1)) / (1024 * 1024) if memory else None)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp1"))
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()
    prompt = args.prompt_file.read_text() if args.prompt_file else DEFAULT_PROMPT

    # Prime cpu_percent counters before the measured request.
    sample_processes()
    time.sleep(0.2)
    result: dict = {}
    failure: list[BaseException] = []

    def generate() -> None:
        try:
            result.update(call_ollama(args.model, prompt, keep_alive="0", num_predict=args.num_predict,
                                      num_ctx=args.num_ctx, seed=args.seed))
        except BaseException as exc:  # preserve the API error after monitoring stops
            failure.append(exc)

    worker = threading.Thread(target=generate, daemon=True)
    started = time.perf_counter()
    worker.start()
    samples = []
    while worker.is_alive():
        cpu, rss = sample_processes()
        gpu_util, gpu_mem = sample_apple_gpu()
        samples.append({"elapsed_s": round(time.perf_counter() - started, 3), "cpu_percent": cpu, "rss_mb": rss, "gpu_util_percent": gpu_util, "gpu_memory_mb": gpu_mem})
        time.sleep(args.sample_seconds)
    worker.join()
    elapsed = time.perf_counter() - started
    cpu, rss = sample_processes()
    gpu_util, gpu_mem = sample_apple_gpu()
    samples.append({"elapsed_s": round(elapsed, 3), "cpu_percent": cpu, "rss_mb": rss, "gpu_util_percent": gpu_util, "gpu_memory_mb": gpu_mem})
    if failure:
        raise failure[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.model.replace(":", "_").replace("/", "_")
    csv_path = args.output_dir / f"{stem}_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["elapsed_s", "cpu_percent", "rss_mb", "gpu_util_percent", "gpu_memory_mb"])
        writer.writeheader(); writer.writerows(samples)
    summary = {
        "experiment": "Experiment 1 - Real-Time Resource Utilisation Monitoring",
        "timestamp_utc": utc_now(), "model": args.model, "sample_interval_s": args.sample_seconds,
        "prompt": "fixed basic Two Sum prompt", "options": {"temperature": 0, "seed": args.seed, "num_predict": args.num_predict, "num_ctx": args.num_ctx},
        "avg_cpu_percent": round(statistics.fmean(x["cpu_percent"] for x in samples), 2),
        "peak_cpu_percent": round(max(x["cpu_percent"] for x in samples), 2),
        "avg_memory_mb": round(statistics.fmean(x["rss_mb"] for x in samples), 2),
        "peak_memory_mb": round(max(x["rss_mb"] for x in samples), 2),
        "generation_time_s": round(elapsed, 3),
        "avg_gpu_usage_percent": round(statistics.fmean(x["gpu_util_percent"] for x in samples if x["gpu_util_percent"] is not None), 2),
        "peak_gpu_usage_percent": round(max(x["gpu_util_percent"] for x in samples if x["gpu_util_percent"] is not None), 2),
        "peak_gpu_memory_mb": round(max(x["gpu_memory_mb"] for x in samples if x["gpu_memory_mb"] is not None), 2),
        "gpu_note": "System-level Apple GPU telemetry from IOKit. GPU memory is unified system memory in use by the GPU driver, not per-process VRAM.",
        "ollama_eval_count": result.get("eval_count"), "ollama_eval_duration_s": round(result.get("eval_duration", 0) / 1e9, 3),
        "sample_csv": str(csv_path),
    }
    json_path = args.output_dir / f"{stem}_summary.json"
    write_json(json_path, summary)
    print(json_path)
    print(summary)


if __name__ == "__main__":
    main()
