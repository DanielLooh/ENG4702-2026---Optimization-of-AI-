#!/usr/bin/env python3
"""Experiment 5: measure one loaded model at idle and during one generation."""

from __future__ import annotations

import argparse
import csv
import statistics
import threading
import time
from pathlib import Path

import psutil

from common import DEFAULT_PROMPT, call_ollama, utc_now, write_json
from exp1_resource_monitor import ollama_processes, sample_processes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp5"))
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--idle-seconds", type=float, default=10)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # A one-token warm-up loads the model. It is deliberately excluded from the
    # measured generation and retained for five minutes while idle is sampled.
    call_ollama(args.model, "Reply with OK.", keep_alive="5m", num_predict=1, num_ctx=args.num_ctx, seed=args.seed)
    sample_processes(); time.sleep(0.2)
    idle = []
    idle_end = time.perf_counter() + args.idle_seconds
    while time.perf_counter() < idle_end:
        _, rss = sample_processes(); idle.append(rss); time.sleep(args.sample_seconds)

    result: dict = {}; failure: list[BaseException] = []
    def generate() -> None:
        try:
            result.update(call_ollama(args.model, DEFAULT_PROMPT, keep_alive="5m", num_predict=args.num_predict,
                                      num_ctx=args.num_ctx, seed=args.seed))
        except BaseException as exc: failure.append(exc)
    worker = threading.Thread(target=generate, daemon=True)
    started = time.perf_counter(); worker.start(); active = []
    while worker.is_alive():
        _, rss = sample_processes(); active.append(rss); time.sleep(args.sample_seconds)
    worker.join(); elapsed = time.perf_counter() - started
    _, final_rss = sample_processes(); active.append(final_rss)
    if failure: raise failure[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.model.replace(":", "_").replace("/", "_")
    csv_path = args.output_dir / f"{stem}_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["phase", "rss_mb"]); writer.writeheader()
        writer.writerows({"phase": "idle", "rss_mb": x} for x in idle)
        writer.writerows({"phase": "generation", "rss_mb": x} for x in active)
    idle_mb = statistics.fmean(idle); active_avg = statistics.fmean(active); active_peak = max(active)
    summary = {
        "experiment": "Experiment 5 - Model Size vs. Memory Footprint", "timestamp_utc": utc_now(), "model": args.model,
        "sample_interval_s": args.sample_seconds, "idle_sample_seconds": args.idle_seconds,
        "options": {"temperature": 0, "seed": args.seed, "num_predict": args.num_predict, "num_ctx": args.num_ctx},
        "idle_memory_mb": round(idle_mb, 2), "avg_memory_during_generation_mb": round(active_avg, 2),
        "peak_memory_during_generation_mb": round(active_peak, 2), "memory_increase_mb": round(active_peak - idle_mb, 2),
        "generation_time_s": round(elapsed, 3), "ollama_eval_count": result.get("eval_count"),
        "measurement_note": "RSS is the sum of the Ollama server and its loaded runner process. On Apple Silicon it is unified memory, not separate VRAM.",
        "sample_csv": str(csv_path),
    }
    json_path = args.output_dir / f"{stem}_summary.json"; write_json(json_path, summary)
    print(json_path); print(summary)


if __name__ == "__main__":
    main()
