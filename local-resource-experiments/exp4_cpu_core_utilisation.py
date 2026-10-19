#!/usr/bin/env python3
"""Experiment 4: measure CPU-core use while executing a generated Python program."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import time
from pathlib import Path

import psutil

from common import utc_now, write_json


def descendants(root: psutil.Process) -> list[psutil.Process]:
    try:
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def cpu_seconds(root: psutil.Process) -> float:
    """CPU seconds used by the runner and any still-live child processes."""
    total = 0.0
    for child in descendants(root):
        try:
            times = child.cpu_times()
            total += times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Use -- before the program command.")
    parser.add_argument("--label", required=True, help="model and task label, e.g. qwen_two_sum")
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp4"))
    parser.add_argument("--sample-seconds", type=float, default=0.2)
    parser.add_argument("--core-threshold-percent", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command: parser.error("supply a command after --")

    proc = subprocess.Popen(command)
    root = psutil.Process(proc.pid)
    samples = []; started = last_wall = time.perf_counter(); last_cpu = cpu_seconds(root)
    while proc.poll() is None:
        time.sleep(args.sample_seconds)
        now = time.perf_counter(); current_cpu = cpu_seconds(root)
        interval = now - last_wall
        process_cpu = max(0.0, (current_cpu - last_cpu) / interval * 100) if interval else 0.0
        # CPU percent is measured in units of one logical core, so 100% is
        # approximately one fully occupied core and 200% approximately two.
        active_cores = math.ceil(process_cpu / 100) if process_cpu >= args.core_threshold_percent else 0
        samples.append({"elapsed_s": round(now-started, 3), "process_cpu_percent": round(process_cpu, 2),
                        "active_cores": active_cores})
        last_wall, last_cpu = now, current_cpu
    elapsed = time.perf_counter() - started
    if proc.returncode != 0: raise SystemExit(f"program exited with {proc.returncode}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.label}_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["elapsed_s", "process_cpu_percent", "active_cores"])
        writer.writeheader(); writer.writerows(samples)
    avg_cores = statistics.fmean(x["active_cores"] for x in samples) if samples else 0
    summary = {"experiment": "Experiment 4 - CPU Core Utilisation Analysis", "timestamp_utc": utc_now(),
               "label": args.label, "command": command, "available_cpu_cores": psutil.cpu_count(logical=True),
               "avg_cores_utilised": round(avg_cores, 2), "peak_cores_utilised": max((x["active_cores"] for x in samples), default=0),
               "avg_cpu_usage_percent": round(statistics.fmean(x["process_cpu_percent"] for x in samples), 2) if samples else 0,
               "peak_cpu_usage_percent": round(max((x["process_cpu_percent"] for x in samples), default=0), 2),
               "multi_core_utilisation": avg_cores >= 2, "execution_time_s": round(elapsed, 3),
               "core_threshold_percent": args.core_threshold_percent, "sample_csv": str(csv_path),
               "interpretation_note": "A core is counted when the runner's CPU use reaches the stated threshold. CPU use is calculated from process CPU seconds, so 100% is approximately one occupied logical core. Compare only trials run on an otherwise idle machine."}
    json_path = args.output_dir / f"{args.label}_summary.json"; write_json(json_path, summary)
    print(json_path); print(summary)


if __name__ == "__main__":
    main()
