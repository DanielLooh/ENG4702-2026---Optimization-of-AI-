"""Measure Qwen2.5-Coder 7B inference and code execution for algorithmic tasks."""

import json
import math
import platform
import random
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import codecarbon
from codecarbon import OfflineEmissionsTracker

from tasks_core import (
    PROBLEMS,
    prompt_basic,
    prompt_efficiency,
    prompt_two_step_generate,
    prompt_two_step_optimize,
    extract_code,
    run_unit_tests,
)


MODEL = "qwen2.5-coder:7b"
MODEL_DIGEST = "dae161e27b0e"

TASKS = ["two_sum", "max_subarray"]
STRATEGIES = ["basic", "efficiency", "two_step"]
N_RUNS = 5

TEMPERATURE = 0.0
MAX_TOKENS = 4096
KEEP_ALIVE = -1

# Reproducible interleaving of the six task/strategy conditions.
SCHEDULE_SEED = 4702

# Execution benchmark settings.
# The runner calibrates the repeat count per task so each implementation
# executes for approximately TARGET_EXEC_SECONDS.
TARGET_EXEC_SECONDS = 10.0
CALIBRATION_REPEATS = 5
MIN_REPEATS = 20
MAX_REPEATS = 5000

EXECUTION_INPUT_SIZE = {
    "two_sum": 100_000,
    "max_subarray": 1_000_000,
}

# Machine-level idle baselines are recorded for transparency.
IDLE_BASELINE_SECONDS = 10.0
SETTLE_SECONDS = 8.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_JSON = str(RESULTS_DIR / "stage2_qwen_algorithmic_full.json")


# Human reference implementations.
def human_two_sum(nums, target):
    hashmap = {}
    for i in range(len(nums)):
        complement = target - nums[i]
        if complement in hashmap:
            return [i, hashmap[complement]]
        hashmap[nums[i]] = i
    return []


def human_max_subarray(nums):
    cur_max, max_till_now = 0, float("-inf")
    for c in nums:
        cur_max = max(c, cur_max + c)
        max_till_now = max(max_till_now, cur_max)
    return max_till_now


HUMAN_FUNCTIONS = {
    "two_sum": human_two_sum,
    "max_subarray": human_max_subarray,
}


# GENERAL HELPERS
def shell_output(args):
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        )
        return (p.stdout or p.stderr).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def save_dataset(dataset):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dataset, f, indent=2)


def ollama_request(payload):
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def preload_model():
    payload = {
        "model": MODEL,
        "prompt": "",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
    }
    ollama_request(payload)


def unload_model():
    payload = {
        "model": MODEL,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }
    ollama_request(payload)


def warmup_model():
    """
    One unmeasured generation after preload.
    This avoids treating first-use/model warm-up behaviour as Run 1.
    """
    payload = {
        "model": MODEL,
        "system": "You are an expert software engineer.",
        "prompt": "Write a Python function identity(x) that returns x.",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 128,
        },
    }
    ollama_request(payload)


def make_tracker(project_name):
    # Machine mode is necessary because Ollama is a separate process and
    # Apple GPU energy is device-level.
    return OfflineEmissionsTracker(
        country_iso_code="AUS",
        project_name=project_name,
        measure_power_secs=1,
        tracking_mode="machine",
        save_to_file=False,
        log_level="error",
    )


def safe_num(x):
    return 0.0 if x is None else float(x)


def tracker_energy(tracker):
    e = tracker.final_emissions_data
    total_kwh = safe_num(e.energy_consumed)
    return {
        "energy_kwh": total_kwh,
        "energy_joules": total_kwh * 3_600_000,
        "cpu_energy_joules": safe_num(e.cpu_energy) * 3_600_000,
        "gpu_energy_joules": safe_num(e.gpu_energy) * 3_600_000,
        "ram_energy_joules": safe_num(e.ram_energy) * 3_600_000,
    }


def measure_idle_baseline(label):
    tracker = make_tracker(label)
    tracker.start()
    start = time.perf_counter()
    time.sleep(IDLE_BASELINE_SECONDS)
    elapsed = time.perf_counter() - start
    tracker.stop()

    energy = tracker_energy(tracker)

    return {
        "elapsed_s": elapsed,
        **energy,
        "average_power_watts": (
            energy["energy_joules"] / elapsed if elapsed > 0 else 0
        ),
    }


def baseline_adjust(raw_energy_j, elapsed_s, baseline_power_w):
    """
    Supplemental value only.
    Raw CodeCarbon/PowerMetrics energy remains stored separately.
    """
    return max(0.0, raw_energy_j - baseline_power_w * elapsed_s)


# PROMPT / INFERENCE
def build_single_prompt(task_key, strategy):
    problem = PROBLEMS[task_key]

    if strategy == "basic":
        return prompt_basic(problem)

    if strategy == "efficiency":
        return prompt_efficiency(problem)

    if strategy == "two_step":
        return prompt_two_step_generate(problem)

    raise ValueError(strategy)


def measured_generation(system_prompt, user_prompt, label):
    payload = {
        "model": MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    tracker = make_tracker(label)
    tracker.start()
    start = time.perf_counter()

    try:
        data = ollama_request(payload)
    finally:
        elapsed = time.perf_counter() - start
        tracker.stop()

    input_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)
    energy = tracker_energy(tracker)

    return {
        "text": data.get("response", ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": output_tokens / elapsed if elapsed > 0 else 0,
        "ollama_load_duration_s": (
            data.get("load_duration", 0) / 1_000_000_000
        ),
        **energy,
    }


def run_inference_trial(task_key, strategy, run_num, block_num, idle_power_w):
    problem = PROBLEMS[task_key]

    if strategy in ("basic", "efficiency"):
        sp, up = build_single_prompt(task_key, strategy)
        r = measured_generation(
            sp,
            up,
            f"qwen_{task_key}_{strategy}_run_{run_num}_inference",
        )

        final_text = r["text"]

        inference = {
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "total_tokens": r["total_tokens"],
            "inference_elapsed_s": r["elapsed_s"],
            "tokens_per_sec": r["tokens_per_sec"],
            "ollama_load_duration_s": r["ollama_load_duration_s"],
            "inference_energy_kwh": r["energy_kwh"],
            "inference_energy_joules": r["energy_joules"],
            "inference_cpu_energy_joules": r["cpu_energy_joules"],
            "inference_gpu_energy_joules": r["gpu_energy_joules"],
            "inference_ram_energy_joules": r["ram_energy_joules"],
            "inference_baseline_adjusted_energy_joules": baseline_adjust(
                r["energy_joules"], r["elapsed_s"], idle_power_w
            ),
            "inference_steps": [
                {
                    "step": 1,
                    "system_prompt": sp,
                    "user_prompt": up,
                    "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"],
                    "total_tokens": r["total_tokens"],
                    "elapsed_s": r["elapsed_s"],
                    "energy_joules": r["energy_joules"],
                }
            ],
        }

    elif strategy == "two_step":
        # Step 1 — generate initial solution.
        sp1, up1 = prompt_two_step_generate(problem)
        r1 = measured_generation(
            sp1,
            up1,
            f"qwen_{task_key}_{strategy}_run_{run_num}_step1",
        )

        # Step 2 — optimize the exact first response.
        sp2, up2 = prompt_two_step_optimize(r1["text"])
        r2 = measured_generation(
            sp2,
            up2,
            f"qwen_{task_key}_{strategy}_run_{run_num}_step2",
        )

        final_text = r2["text"]

        total_elapsed = r1["elapsed_s"] + r2["elapsed_s"]
        total_output = r1["output_tokens"] + r2["output_tokens"]
        total_energy_j = r1["energy_joules"] + r2["energy_joules"]

        inference = {
            "input_tokens": r1["input_tokens"] + r2["input_tokens"],
            "output_tokens": total_output,
            "total_tokens": r1["total_tokens"] + r2["total_tokens"],
            "inference_elapsed_s": total_elapsed,
            "tokens_per_sec": (
                total_output / total_elapsed if total_elapsed > 0 else 0
            ),
            "ollama_load_duration_s": (
                r1["ollama_load_duration_s"] + r2["ollama_load_duration_s"]
            ),
            "inference_energy_kwh": (
                r1["energy_kwh"] + r2["energy_kwh"]
            ),
            "inference_energy_joules": total_energy_j,
            "inference_cpu_energy_joules": (
                r1["cpu_energy_joules"] + r2["cpu_energy_joules"]
            ),
            "inference_gpu_energy_joules": (
                r1["gpu_energy_joules"] + r2["gpu_energy_joules"]
            ),
            "inference_ram_energy_joules": (
                r1["ram_energy_joules"] + r2["ram_energy_joules"]
            ),
            "inference_baseline_adjusted_energy_joules": baseline_adjust(
                total_energy_j, total_elapsed, idle_power_w
            ),
            "inference_steps": [
                {
                    "step": 1,
                    "system_prompt": sp1,
                    "user_prompt": up1,
                    "input_tokens": r1["input_tokens"],
                    "output_tokens": r1["output_tokens"],
                    "total_tokens": r1["total_tokens"],
                    "elapsed_s": r1["elapsed_s"],
                    "energy_joules": r1["energy_joules"],
                },
                {
                    "step": 2,
                    "system_prompt": sp2,
                    "user_prompt": up2,
                    "input_tokens": r2["input_tokens"],
                    "output_tokens": r2["output_tokens"],
                    "total_tokens": r2["total_tokens"],
                    "elapsed_s": r2["elapsed_s"],
                    "energy_joules": r2["energy_joules"],
                },
            ],
        }

    else:
        raise ValueError(strategy)

    generated_code = extract_code(final_text)
    passed, total = run_unit_tests(generated_code, task_key)

    return {
        "run": run_num,
        "block": block_num,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "task": task_key,
        "strategy": strategy,
        "generated_response": final_text,
        "generated_code": generated_code,
        "tests_passed": passed,
        "tests_total": total,
        **inference,
    }


# EXECUTION BENCHMARKS
def get_generated_callable(code, task_key):
    ns = {}
    exec(
        compile("from typing import *\n" + code, "<generated>", "exec"),
        ns,
    )

    if task_key == "two_sum":
        names = ["two_sum", "twoSum"]
    elif task_key == "max_subarray":
        names = ["max_subarray", "maxSubArray", "maxSubarray", "maxSub"]
    else:
        raise ValueError(task_key)

    for name in names:
        fn = ns.get(name)
        if callable(fn):
            return fn

    solution_cls = ns.get("Solution")
    if solution_cls is not None:
        obj = solution_cls()
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn

    raise RuntimeError(f"Could not find callable for {task_key}")


def make_execution_case(task_key):
    n = EXECUTION_INPUT_SIZE[task_key]

    if task_key == "two_sum":
        nums = list(range(n))
        target = nums[-2] + nums[-1]
        expected = [n - 2, n - 1]
        return {
            "nums": nums,
            "target": target,
            "expected": expected,
            "input_size": n,
        }

    if task_key == "max_subarray":
        # Repeated mixed-sign pattern; expected value is computed once
        # with the exact shared human baseline.
        pattern = [1, -2, 3, 4, -1, 2, -5, 4]
        nums = (pattern * (n // len(pattern) + 1))[:n]
        expected = human_max_subarray(nums)
        return {
            "nums": nums,
            "expected": expected,
            "input_size": n,
        }

    raise ValueError(task_key)


def call_task(fn, task_key, case):
    if task_key == "two_sum":
        return fn(case["nums"], case["target"])
    return fn(case["nums"])


def result_is_correct(task_key, result, expected):
    if task_key == "two_sum":
        try:
            return sorted(result) == sorted(expected)
        except Exception:
            return False
    return result == expected


def function_mutates_input(fn, task_key, case):
    probe_case = dict(case)
    probe_case["nums"] = list(case["nums"])
    before = list(probe_case["nums"])
    call_task(fn, task_key, probe_case)
    return probe_case["nums"] != before


def calibrate_repeats(task_key, case):
    fn = HUMAN_FUNCTIONS[task_key]

    start = time.perf_counter()
    for _ in range(CALIBRATION_REPEATS):
        call_task(fn, task_key, case)
    elapsed = time.perf_counter() - start

    per_call = elapsed / CALIBRATION_REPEATS if elapsed > 0 else 0.001
    repeats = math.ceil(TARGET_EXEC_SECONDS / per_call)

    return max(MIN_REPEATS, min(MAX_REPEATS, repeats))


def measure_function_execution(
    fn,
    task_key,
    case,
    repeats,
    label,
    idle_power_w,
):
    # Unmeasured warm-up call.
    call_task(fn, task_key, case)

    tracker = make_tracker(label)
    tracker.start()
    start = time.perf_counter()

    try:
        for _ in range(repeats):
            call_task(fn, task_key, case)
    finally:
        elapsed = time.perf_counter() - start
        tracker.stop()

    energy = tracker_energy(tracker)

    return {
        "execution_elapsed_s": elapsed,
        "execution_repeats": repeats,
        "execution_time_per_call_s": elapsed / repeats,
        "execution_energy_kwh": energy["energy_kwh"],
        "execution_energy_joules": energy["energy_joules"],
        "execution_cpu_energy_joules": energy["cpu_energy_joules"],
        "execution_gpu_energy_joules": energy["gpu_energy_joules"],
        "execution_ram_energy_joules": energy["ram_energy_joules"],
        "execution_energy_per_call_joules": (
            energy["energy_joules"] / repeats
        ),
        "execution_baseline_adjusted_energy_joules": baseline_adjust(
            energy["energy_joules"], elapsed, idle_power_w
        ),
        "execution_baseline_adjusted_energy_per_call_joules": (
            baseline_adjust(
                energy["energy_joules"], elapsed, idle_power_w
            ) / repeats
        ),
    }


# SCHEDULE + SUMMARY
def build_schedule():
    """
    Five blocks.
    Every block contains all 6 task/strategy conditions exactly once.
    Order is shuffled deterministically to reduce time/thermal/order bias.
    """
    schedule = []

    base_conditions = [
        (task, strategy)
        for task in TASKS
        for strategy in STRATEGIES
    ]

    for block in range(1, N_RUNS + 1):
        conditions = list(base_conditions)
        rng = random.Random(SCHEDULE_SEED + block)
        rng.shuffle(conditions)

        for task_key, strategy in conditions:
            schedule.append({
                "block": block,
                "task": task_key,
                "strategy": strategy,
            })

    return schedule


def condition_results(results, task_key, strategy):
    return [
        r for r in results
        if r["task"] == task_key and r["strategy"] == strategy
    ]


def mean_field(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def median_field(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def print_summary(results):
    print("\n\n================ FINAL CATEGORY 1 SUMMARY ================")

    for task_key in TASKS:
        print(f"\n--- {task_key} ---")

        for strategy in STRATEGIES:
            rows = condition_results(results, task_key, strategy)
            if not rows:
                continue

            passed = sum(r["tests_passed"] for r in rows)
            total = sum(r["tests_total"] for r in rows)

            avg_inf_j = mean_field(rows, "inference_energy_joules")
            med_inf_j = median_field(rows, "inference_energy_joules")
            avg_inf_s = mean_field(rows, "inference_elapsed_s")
            avg_tokens = mean_field(rows, "total_tokens")

            exec_rows = [
                r for r in rows
                if "execution_error" not in r
                and r.get("execution_energy_ratio_llm_vs_human") is not None
            ]

            if exec_rows:
                avg_ratio = mean_field(
                    exec_rows, "execution_energy_ratio_llm_vs_human"
                )
                med_ratio = median_field(
                    exec_rows, "execution_energy_ratio_llm_vs_human"
                )
            else:
                avg_ratio = None
                med_ratio = None

            print(
                f"{strategy:<11} | "
                f"tests {passed}/{total} | "
                f"avg tokens {avg_tokens:.1f} | "
                f"avg inf {avg_inf_s:.3f}s | "
                f"inf energy mean/median {avg_inf_j:.2f}/{med_inf_j:.2f} J",
                end="",
            )

            if avg_ratio is not None:
                print(
                    f" | exec LLM/Human energy mean/median "
                    f"{avg_ratio:.3f}/{med_ratio:.3f}x"
                )
            else:
                print(" | exec comparison unavailable")


# MAIN
def main():
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "tasks": TASKS,
        "strategies": STRATEGIES,
        "runs_per_condition": N_RUNS,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "schedule_seed": SCHEDULE_SEED,
        "machine": {
            "chip": "Apple M4",
            "ram_gb": 16,
            "python_version": platform.python_version(),
            "macos_version_direct": shell_output(
                ["sw_vers", "-productVersion"]
            ),
            "ollama_version": shell_output(["ollama", "--version"]),
            "codecarbon_version": codecarbon.__version__,
            "energy_backend": "Apple PowerMetrics via CodeCarbon",
        },
        "execution_protocol": {
            "target_seconds_per_implementation": TARGET_EXEC_SECONDS,
            "input_sizes": EXECUTION_INPUT_SIZE,
            "shared_human_baseline": True,
            "model_unloaded_during_execution_phase": True,
        },
    }

    dataset = {
        "metadata": metadata,
        "idle_baselines": {},
        "execution_calibration": {},
        "results": [],
    }

    print("\n============================================================")
    print("STAGE 2 — QWEN ALGORITHMIC CATEGORY")
    print("Two Sum + Maximum Subarray")
    print("Basic + Efficiency + Two-Step")
    print("5 runs per condition = 30 logical trials")
    print("============================================================")
    print("\nIMPORTANT FOR ENERGY VALIDITY:")
    print("- Leave the machine alone while measured phases run.")
    print("\nThe script saves after every trial, so partial progress is retained.")

    input("\nPress Enter when the machine is ready and you will leave it idle... ")

    print(f"\nSettling for {SETTLE_SECONDS:.0f} seconds...")
    time.sleep(SETTLE_SECONDS)

    # --------------------------------------------------------
    # PHASE A — INFERENCE
    # --------------------------------------------------------
    print("\n================ PHASE A: INFERENCE ================")
    print("Preloading model...")
    preload_model()

    print("Performing one UNMEASURED warm-up generation...")
    warmup_model()

    print(f"Settling for {SETTLE_SECONDS:.0f} seconds...")
    time.sleep(SETTLE_SECONDS)

    print(
        f"Measuring {IDLE_BASELINE_SECONDS:.0f}s inference-phase idle baseline "
        "(model loaded, no generation)..."
    )
    inf_idle = measure_idle_baseline("qwen_inference_idle_baseline")
    dataset["idle_baselines"]["inference_model_loaded"] = inf_idle
    save_dataset(dataset)

    print(
        f"Inference idle baseline: "
        f"{inf_idle['average_power_watts']:.3f} W"
    )

    schedule = build_schedule()
    condition_counts = {
        (task, strategy): 0
        for task in TASKS
        for strategy in STRATEGIES
    }

    for i, item in enumerate(schedule, 1):
        task_key = item["task"]
        strategy = item["strategy"]
        block = item["block"]

        condition_counts[(task_key, strategy)] += 1
        run_num = condition_counts[(task_key, strategy)]

        print(
            f"\n[Inference {i}/30] "
            f"block={block} task={task_key} strategy={strategy} "
            f"run={run_num}/5"
        )

        result = run_inference_trial(
            task_key,
            strategy,
            run_num,
            block,
            inf_idle["average_power_watts"],
        )

        dataset["results"].append(result)
        save_dataset(dataset)

        print(
            f"  tests {result['tests_passed']}/{result['tests_total']} | "
            f"{result['total_tokens']} tokens | "
            f"{result['inference_elapsed_s']:.3f}s | "
            f"{result['inference_energy_joules']:.2f} J raw | "
            f"{result['inference_baseline_adjusted_energy_joules']:.2f} J adjusted"
        )

        time.sleep(2)

    # --------------------------------------------------------
    # PHASE B — EXECUTION
    # --------------------------------------------------------
    print("\n================ PHASE B: EXECUTION ================")
    print("Unloading Qwen so model residency does NOT contaminate code execution energy...")
    unload_model()

    print(f"Settling for {SETTLE_SECONDS:.0f} seconds...")
    time.sleep(SETTLE_SECONDS)

    print(
        f"Measuring {IDLE_BASELINE_SECONDS:.0f}s execution-phase idle baseline "
        "(model unloaded)..."
    )
    exec_idle = measure_idle_baseline("qwen_execution_idle_baseline")
    dataset["idle_baselines"]["execution_model_unloaded"] = exec_idle

    print(
        f"Execution idle baseline: "
        f"{exec_idle['average_power_watts']:.3f} W"
    )

    cases = {
        task_key: make_execution_case(task_key)
        for task_key in TASKS
    }

    repeats = {
        task_key: calibrate_repeats(task_key, cases[task_key])
        for task_key in TASKS
    }

    dataset["execution_calibration"] = {
        task_key: {
            "input_size": cases[task_key]["input_size"],
            "repeats": repeats[task_key],
            "target_seconds": TARGET_EXEC_SECONDS,
        }
        for task_key in TASKS
    }
    save_dataset(dataset)

    for task_key in TASKS:
        print(
            f"{task_key}: input={cases[task_key]['input_size']:,}, "
            f"repeats={repeats[task_key]}"
        )

    for i, result in enumerate(dataset["results"], 1):
        task_key = result["task"]
        strategy = result["strategy"]
        run_num = result["run"]

        print(
            f"\n[Execution {i}/30] "
            f"task={task_key} strategy={strategy} run={run_num}/5"
        )

        code = result.get("generated_code")

        if not code or result["tests_passed"] != result["tests_total"]:
            result["execution_error"] = (
                "Skipped: generated code did not pass all unit tests."
            )
            save_dataset(dataset)
            print("  skipped — generated code did not pass all unit tests")
            continue

        try:
            llm_fn = get_generated_callable(code, task_key)
            human_fn = HUMAN_FUNCTIONS[task_key]
            case = cases[task_key]
            expected = case["expected"]

            # Validate large benchmark input.
            llm_large_result = call_task(llm_fn, task_key, case)
            human_large_result = call_task(human_fn, task_key, case)

            if not result_is_correct(task_key, llm_large_result, expected):
                raise RuntimeError(
                    f"LLM code failed large-input validation: "
                    f"got {llm_large_result}, expected {expected}"
                )

            if not result_is_correct(task_key, human_large_result, expected):
                raise RuntimeError(
                    "Shared human baseline failed large-input validation."
                )

            # Avoid silently benchmarking a mutating implementation on reused data.
            llm_mutates = function_mutates_input(llm_fn, task_key, case)
            result["generated_code_mutates_input"] = llm_mutates

            if llm_mutates:
                raise RuntimeError(
                    "Generated code mutates the benchmark input. "
                    "Execution-energy benchmark skipped to avoid mixing "
                    "algorithm energy with repeated input reconstruction."
                )

            n_repeats = repeats[task_key]

            # Alternate pair order to reduce order / thermal bias.
            if i % 2 == 1:
                llm_exec = measure_function_execution(
                    llm_fn,
                    task_key,
                    case,
                    n_repeats,
                    f"{task_key}_{strategy}_run_{run_num}_llm_exec",
                    exec_idle["average_power_watts"],
                )
                human_exec = measure_function_execution(
                    human_fn,
                    task_key,
                    case,
                    n_repeats,
                    f"{task_key}_{strategy}_run_{run_num}_human_exec",
                    exec_idle["average_power_watts"],
                )
            else:
                human_exec = measure_function_execution(
                    human_fn,
                    task_key,
                    case,
                    n_repeats,
                    f"{task_key}_{strategy}_run_{run_num}_human_exec",
                    exec_idle["average_power_watts"],
                )
                llm_exec = measure_function_execution(
                    llm_fn,
                    task_key,
                    case,
                    n_repeats,
                    f"{task_key}_{strategy}_run_{run_num}_llm_exec",
                    exec_idle["average_power_watts"],
                )

            for key, value in llm_exec.items():
                result[f"llm_{key}"] = value

            for key, value in human_exec.items():
                result[f"human_{key}"] = value

            raw_human_j = human_exec["execution_energy_joules"]
            raw_llm_j = llm_exec["execution_energy_joules"]

            adj_human_j = human_exec[
                "execution_baseline_adjusted_energy_joules"
            ]
            adj_llm_j = llm_exec[
                "execution_baseline_adjusted_energy_joules"
            ]

            result["execution_energy_ratio_llm_vs_human"] = (
                raw_llm_j / raw_human_j if raw_human_j > 0 else None
            )
            result[
                "execution_baseline_adjusted_energy_ratio_llm_vs_human"
            ] = (
                adj_llm_j / adj_human_j if adj_human_j > 0 else None
            )

            result["execution_time_ratio_llm_vs_human"] = (
                llm_exec["execution_elapsed_s"]
                / human_exec["execution_elapsed_s"]
                if human_exec["execution_elapsed_s"] > 0
                else None
            )

            print(
                f"  LLM:   {llm_exec['execution_elapsed_s']:.3f}s | "
                f"{raw_llm_j:.2f} J raw | "
                f"{adj_llm_j:.2f} J adjusted"
            )
            print(
                f"  Human: {human_exec['execution_elapsed_s']:.3f}s | "
                f"{raw_human_j:.2f} J raw | "
                f"{adj_human_j:.2f} J adjusted"
            )
            print(
                f"  raw energy ratio LLM/Human: "
                f"{result['execution_energy_ratio_llm_vs_human']:.3f}x"
            )

        except Exception as exc:
            result["execution_error"] = str(exc)
            print("  execution benchmark error:", exc)

        save_dataset(dataset)
        time.sleep(2)

    print_summary(dataset["results"])
    save_dataset(dataset)

    print(f"\nSaved complete raw dataset to: {OUTPUT_JSON}")
    print("\nYou can now use this file as the Qwen Category 1 dataset.")


if __name__ == "__main__":
    main()
