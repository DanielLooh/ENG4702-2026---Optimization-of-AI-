"""Measure DeepSeek-Coder-V2-Lite inference and code execution for weather API tasks."""

import ast
import copy
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import codecarbon
from codecarbon import OfflineEmissionsTracker

import tasks_api as cat3

from tasks_api import (
    PROBLEMS,
    prompt_basic,
    prompt_efficiency,
    prompt_two_step_generate,
    prompt_two_step_optimize,
    extract_code,
    run_unit_tests,
    _CITY_DB,
    _build_mock_requests,
    _exec_with_mock,
    _set_requests,
)


MODEL = "deepseek-coder-v2:lite"
MODEL_DIGEST = "63fb193b3a9b"

TASKS = ["weather_fetch", "weather_compare"]
STRATEGIES = ["basic", "efficiency", "two_step"]
N_RUNS = 5

TEMPERATURE = 0.0
MAX_TOKENS = 4096
KEEP_ALIVE = -1
SCHEDULE_SEED = 4702

TARGET_EXEC_SECONDS = 10.0
CALIBRATION_REPEATS = 3
MIN_REPEATS = 5
MAX_REPEATS = 50_000

# Each measured benchmark invocation does enough local mock-API work to
# produce a stable multi-second PowerMetrics window after calibration.
EXECUTION_INPUT_SIZE = {
    "weather_fetch": 500,      # 500 single-city mock fetches per invocation
    "weather_compare": 100,    # one 100-city comparison per invocation
}

IDLE_BASELINE_SECONDS = 10.0
SETTLE_SECONDS = 8.0
TRACKER_COOLDOWN_SECONDS = 3.0
TRACKER_RETRY_SETTLE_SECONDS = 8.0
TRACKER_MAX_ATTEMPTS = 3

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_JSON = str(RESULTS_DIR / "stage2_deepseek_api_full.json")


# Define API solutions without running unrelated top-level examples.
# The AST filter is an evaluation helper, not a security sandbox.

def _safe_exec_with_mock(code: str, mock) -> dict:
    saved = {
        k: sys.modules[k]
        for k in ("requests", "requests.exceptions")
        if k in sys.modules
    }
    sys.modules["requests"] = mock
    sys.modules["requests.exceptions"] = mock.exceptions

    source = "from typing import *\n" + code
    tree = ast.parse(source, filename="<generated>", mode="exec")

    safe_body = []
    for node in tree.body:
        if isinstance(node, (
            ast.Import, ast.ImportFrom, ast.FunctionDef,
            ast.AsyncFunctionDef, ast.ClassDef
        )):
            safe_body.append(node)
            continue

        # Preserve constants/configuration such as API_URL = "...", but
        # suppress assignments that actually call functions at module scope
        # (commonly demo code like result = compare_weather(...)).
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = getattr(node, "value", None)
            has_call = value is not None and any(
                isinstance(child, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom))
                for child in ast.walk(value)
            )
            if not has_call:
                safe_body.append(node)
            continue

        # Keep a module docstring, if present.
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            safe_body.append(node)

        # Deliberately skip top-level demo calls, loops, and __main__ blocks.

    safe_tree = ast.Module(body=safe_body, type_ignores=[])
    ast.fix_missing_locations(safe_tree)

    ns = {"__name__": "__generated__"}
    try:
        exec(compile(safe_tree, "<generated>", "exec"), ns)
    finally:
        for k, v in saved.items():
            sys.modules[k] = v
        for k in ("requests", "requests.exceptions"):
            if k not in saved:
                sys.modules.pop(k, None)

    return ns


# Use the shared weather tests with top-level example execution suppressed.
cat3._exec_with_mock = _safe_exec_with_mock

# Human reference implementations.
# Uses the same deterministic mock service as generated code.
_EXECUTION_MOCK = _build_mock_requests(_CITY_DB)


def human_fetch_weather(city, api_key):
    try:
        response = _EXECUTION_MOCK.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
        }
    except Exception as exc:
        return {"error": str(exc), "city": city}


def human_compare_weather(cities, api_key):
    results = []
    for city in cities:
        weather = human_fetch_weather(city, api_key)
        if "error" not in weather:
            results.append(weather)

    if not results:
        return {"error": "No data available", "results": []}

    hottest = max(results, key=lambda x: x["temperature"])["city"]
    coldest = min(results, key=lambda x: x["temperature"])["city"]
    avg_temp = sum(x["temperature"] for x in results) / len(results)

    return {
        "results": results,
        "hottest": hottest,
        "coldest": coldest,
        "avg_temp": avg_temp,
    }


HUMAN_FUNCTIONS = {
    "weather_fetch": human_fetch_weather,
    "weather_compare": human_compare_weather,
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
    """
    Atomically save strict JSON.

    allow_nan=False prevents an invalid PowerMetrics measurement from being
    silently written as NaN, which is not valid JSON.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_JSON + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(dataset, f, indent=2, allow_nan=False)
    os.replace(tmp_path, OUTPUT_JSON)


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


def require_finite_num(x, field_name):
    """
    Reject missing/NaN/inf energy values instead of silently contaminating
    the experiment dataset.
    """
    if x is None:
        raise RuntimeError(f"CodeCarbon returned None for {field_name}")

    value = float(x)
    if not math.isfinite(value):
        raise RuntimeError(
            f"CodeCarbon returned non-finite {field_name}: {value}"
        )
    return value


def tracker_energy(tracker):
    e = tracker.final_emissions_data

    total_kwh = require_finite_num(e.energy_consumed, "energy_consumed")
    cpu_kwh = require_finite_num(e.cpu_energy, "cpu_energy")
    gpu_kwh = require_finite_num(e.gpu_energy, "gpu_energy")
    ram_kwh = require_finite_num(e.ram_energy, "ram_energy")

    return {
        "energy_kwh": total_kwh,
        "energy_joules": total_kwh * 3_600_000,
        "cpu_energy_joules": cpu_kwh * 3_600_000,
        "gpu_energy_joules": gpu_kwh * 3_600_000,
        "ram_energy_joules": ram_kwh * 3_600_000,
    }


def measure_idle_baseline(label):
    last_error = None

    for attempt in range(1, TRACKER_MAX_ATTEMPTS + 1):
        tracker = make_tracker(f"{label}_attempt_{attempt}")
        tracker.start()
        start = time.perf_counter()

        try:
            time.sleep(IDLE_BASELINE_SECONDS)
        finally:
            elapsed = time.perf_counter() - start
            tracker.stop()

        try:
            energy = tracker_energy(tracker)
        except RuntimeError as exc:
            last_error = exc
            if attempt == TRACKER_MAX_ATTEMPTS:
                break

            print(
                f"  WARNING: invalid idle energy measurement "
                f"(attempt {attempt}/{TRACKER_MAX_ATTEMPTS}): {exc}"
            )
            print(
                f"  Retrying after {TRACKER_RETRY_SETTLE_SECONDS:.0f}s..."
            )
            time.sleep(TRACKER_RETRY_SETTLE_SECONDS)
            continue

        time.sleep(TRACKER_COOLDOWN_SECONDS)

        return {
            "elapsed_s": elapsed,
            **energy,
            "average_power_watts": (
                energy["energy_joules"] / elapsed if elapsed > 0 else 0
            ),
            "measurement_attempts": attempt,
        }

    raise RuntimeError(
        "Could not obtain a valid idle energy measurement after "
        f"{TRACKER_MAX_ATTEMPTS} attempts: {last_error}"
    )


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

    last_error = None

    for attempt in range(1, TRACKER_MAX_ATTEMPTS + 1):
        tracker = make_tracker(f"{label}_attempt_{attempt}")
        tracker.start()
        start = time.perf_counter()

        try:
            data = ollama_request(payload)
        finally:
            elapsed = time.perf_counter() - start
            tracker.stop()

        try:
            energy = tracker_energy(tracker)
        except RuntimeError as exc:
            last_error = exc

            if attempt == TRACKER_MAX_ATTEMPTS:
                break

            print(
                f"  WARNING: invalid inference energy measurement "
                f"(attempt {attempt}/{TRACKER_MAX_ATTEMPTS}): {exc}"
            )
            print(
                f"  Discarding that generation and retrying after "
                f"{TRACKER_RETRY_SETTLE_SECONDS:.0f}s..."
            )
            time.sleep(TRACKER_RETRY_SETTLE_SECONDS)
            continue

        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        # Give the PowerMetrics subprocess a moment to fully tear down before
        # another tracker is started. This is especially important between
        # Two-Step step 1 and step 2.
        time.sleep(TRACKER_COOLDOWN_SECONDS)

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
            "measurement_attempts": attempt,
            **energy,
        }

    raise RuntimeError(
        "Could not obtain a valid inference energy measurement after "
        f"{TRACKER_MAX_ATTEMPTS} attempts: {last_error}"
    )


def run_inference_trial(task_key, strategy, run_num, block_num, idle_power_w):
    problem = PROBLEMS[task_key]

    if strategy in ("basic", "efficiency"):
        sp, up = build_single_prompt(task_key, strategy)
        r = measured_generation(
            sp,
            up,
            f"deepseek_{task_key}_{strategy}_run_{run_num}_inference",
        )

        final_text = r["text"]

        inference = {
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "total_tokens": r["total_tokens"],
            "inference_elapsed_s": r["elapsed_s"],
            "tokens_per_sec": r["tokens_per_sec"],
            "ollama_load_duration_s": r["ollama_load_duration_s"],
            "inference_measurement_attempts": r["measurement_attempts"],
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
                    "measurement_attempts": r["measurement_attempts"],
                }
            ],
        }

    elif strategy == "two_step":
        # Step 1 — generate initial solution.
        sp1, up1 = prompt_two_step_generate(problem)
        r1 = measured_generation(
            sp1,
            up1,
            f"deepseek_{task_key}_{strategy}_run_{run_num}_step1",
        )

        # Step 2 — optimize the exact first response.
        sp2, up2 = prompt_two_step_optimize(r1["text"])
        r2 = measured_generation(
            sp2,
            up2,
            f"deepseek_{task_key}_{strategy}_run_{run_num}_step2",
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
            "inference_measurement_attempts": (
                r1["measurement_attempts"] + r2["measurement_attempts"]
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
                    "measurement_attempts": r1["measurement_attempts"],
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
                    "measurement_attempts": r2["measurement_attempts"],
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
    # Compile generated code with the exact same deterministic mock requests
    # module used by the Stage 1 API unit tests. No network access occurs.
    mock = _build_mock_requests(_CITY_DB)
    ns = _safe_exec_with_mock(code, mock)
    _set_requests(ns, mock)

    if task_key == "weather_fetch":
        names = [
            "fetch_weather", "get_weather", "weather_fetch",
            "current_weather", "get_current_weather",
        ]
    elif task_key == "weather_compare":
        names = [
            "compare_weather", "fetch_multiple", "multi_weather",
            "compare_cities", "weather_comparison",
            "get_weather_comparison", "fetch_cities_weather",
        ]
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
    api_key = "fake_key"
    valid_cities = ["London", "Paris", "Berlin", "Tokyo", "Sydney"]

    if task_key == "weather_fetch":
        n = EXECUTION_INPUT_SIZE[task_key]
        cities = [valid_cities[i % len(valid_cities)] for i in range(n)]
        expected = [human_fetch_weather(city, api_key) for city in cities]
        return {
            "cities": cities,
            "api_key": api_key,
            "expected": expected,
            "input_size": n,
        }

    if task_key == "weather_compare":
        n = EXECUTION_INPUT_SIZE[task_key]
        cities = [valid_cities[i % len(valid_cities)] for i in range(n)]
        expected = human_compare_weather(cities, api_key)
        return {
            "cities": cities,
            "api_key": api_key,
            "expected": expected,
            "input_size": n,
        }

    raise ValueError(task_key)


def call_task(fn, task_key, case):
    if task_key == "weather_fetch":
        return [fn(city, case["api_key"]) for city in case["cities"]]
    if task_key == "weather_compare":
        return fn(case["cities"], case["api_key"])
    raise ValueError(task_key)


def _weather_record_matches(actual, expected):
    if not isinstance(actual, dict) or "error" in actual:
        return False
    try:
        temp = actual.get("temperature", actual.get("temp"))
        return (
            actual.get("city") == expected["city"]
            and abs(float(temp) - float(expected["temperature"])) < 1e-9
            and abs(float(actual.get("feels_like")) - float(expected["feels_like"])) < 1e-9
            and int(actual.get("humidity")) == int(expected["humidity"])
            and actual.get("description") == expected["description"]
            and abs(float(actual.get("wind_speed")) - float(expected["wind_speed"])) < 1e-9
        )
    except (TypeError, ValueError):
        return False


def result_is_correct(task_key, result, expected):
    if task_key == "weather_fetch":
        return (
            isinstance(result, list)
            and len(result) == len(expected)
            and all(_weather_record_matches(a, e) for a, e in zip(result, expected))
        )

    if task_key == "weather_compare":
        if not isinstance(result, dict):
            return False
        hottest = result.get("hottest", result.get("hottest_city"))
        coldest = result.get("coldest", result.get("coldest_city"))
        avg_temp = result.get(
            "avg_temp",
            result.get("average_temp", result.get("average_temperature")),
        )
        results = result.get("results")
        try:
            return (
                isinstance(results, list)
                and len(results) == len(expected["results"])
                and hottest == expected["hottest"]
                and coldest == expected["coldest"]
                and abs(float(avg_temp) - float(expected["avg_temp"])) < 1e-6
            )
        except (TypeError, ValueError):
            return False

    return False


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

    last_error = None

    for attempt in range(1, TRACKER_MAX_ATTEMPTS + 1):
        tracker = make_tracker(f"{label}_attempt_{attempt}")
        tracker.start()
        start = time.perf_counter()

        try:
            for _ in range(repeats):
                call_task(fn, task_key, case)
        finally:
            elapsed = time.perf_counter() - start
            tracker.stop()

        try:
            energy = tracker_energy(tracker)
        except RuntimeError as exc:
            last_error = exc

            if attempt == TRACKER_MAX_ATTEMPTS:
                break

            print(
                f"  WARNING: invalid execution energy measurement "
                f"(attempt {attempt}/{TRACKER_MAX_ATTEMPTS}): {exc}"
            )
            print(
                f"  Retrying after {TRACKER_RETRY_SETTLE_SECONDS:.0f}s..."
            )
            time.sleep(TRACKER_RETRY_SETTLE_SECONDS)
            continue

        adjusted_j = baseline_adjust(
            energy["energy_joules"], elapsed, idle_power_w
        )

        time.sleep(TRACKER_COOLDOWN_SECONDS)

        return {
            "execution_elapsed_s": elapsed,
            "execution_repeats": repeats,
            "execution_time_per_call_s": elapsed / repeats,
            "execution_measurement_attempts": attempt,
            "execution_energy_kwh": energy["energy_kwh"],
            "execution_energy_joules": energy["energy_joules"],
            "execution_cpu_energy_joules": energy["cpu_energy_joules"],
            "execution_gpu_energy_joules": energy["gpu_energy_joules"],
            "execution_ram_energy_joules": energy["ram_energy_joules"],
            "execution_energy_per_call_joules": (
                energy["energy_joules"] / repeats
            ),
            "execution_baseline_adjusted_energy_joules": adjusted_j,
            "execution_baseline_adjusted_energy_per_call_joules": (
                adjusted_j / repeats
            ),
        }

    raise RuntimeError(
        "Could not obtain a valid execution energy measurement after "
        f"{TRACKER_MAX_ATTEMPTS} attempts: {last_error}"
    )


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
    print("\n\n================ FINAL CATEGORY 3 SUMMARY ================")

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
                avg_time_ratio = mean_field(
                    exec_rows, "execution_time_ratio_llm_vs_human"
                )
                med_time_ratio = median_field(
                    exec_rows, "execution_time_ratio_llm_vs_human"
                )
            else:
                avg_ratio = None
                med_ratio = None
                avg_time_ratio = None
                med_time_ratio = None

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
                    f" | exec energy mean/median "
                    f"{avg_ratio:.3f}/{med_ratio:.3f}x"
                    f" | exec time mean/median "
                    f"{avg_time_ratio:.3f}/{med_time_ratio:.3f}x"
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
            "network_access_during_execution": False,
            "api_backend": "deterministic mock OpenWeatherMap responses",
        },
        "energy_measurement_safeguards": {
            "reject_non_finite_values": True,
            "strict_json_allow_nan": False,
            "max_attempts_per_measurement": TRACKER_MAX_ATTEMPTS,
            "cooldown_seconds_between_trackers": TRACKER_COOLDOWN_SECONDS,
            "retry_settle_seconds": TRACKER_RETRY_SETTLE_SECONDS,
        },
    }

    dataset = {
        "metadata": metadata,
        "idle_baselines": {},
        "execution_calibration": {},
        "results": [],
    }

    print("\n============================================================")
    print("STAGE 2 — DEEPSEEK API INTEGRATION CATEGORY")
    print("Weather Fetch + Weather Compare")
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
    inf_idle = measure_idle_baseline("deepseek_inference_idle_baseline")
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
    print("Unloading DeepSeek so model residency does NOT contaminate code execution energy...")
    unload_model()

    print(f"Settling for {SETTLE_SECONDS:.0f} seconds...")
    time.sleep(SETTLE_SECONDS)

    print(
        f"Measuring {IDLE_BASELINE_SECONDS:.0f}s execution-phase idle baseline "
        "(model unloaded)..."
    )
    exec_idle = measure_idle_baseline("deepseek_execution_idle_baseline")
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
    print("\nYou can now use this file as the DeepSeek Category 3 dataset.")


if __name__ == "__main__":
    main()
