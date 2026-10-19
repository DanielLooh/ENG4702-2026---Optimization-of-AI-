"""Repeat invalid Qwen algorithm measurements while preserving valid results."""

import json
import math
import time
from datetime import datetime

import stage2_qwen_algorithmic_full as exp

DATASET_PATH = exp.OUTPUT_JSON

# Give PowerMetrics/CodeCarbon more time to settle between repeated trackers.
TRACKER_RETRY_DELAY_S = 4
MAX_MEASUREMENT_ATTEMPTS = 4
POST_UNLOAD_SETTLE_S = 20


def is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def save(data):
    with open(DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)


def inference_is_bad(row):
    required = [
        "inference_energy_joules",
        "inference_cpu_energy_joules",
        "inference_gpu_energy_joules",
        "inference_ram_energy_joules",
    ]
    return any(not is_finite_number(row.get(k)) for k in required)


def execution_pair_is_bad(row):
    required = [
        "llm_execution_energy_joules",
        "human_execution_energy_joules",
    ]
    return any(not is_finite_number(row.get(k)) for k in required)


def valid_execution_measurement(m):
    required = [
        "execution_energy_joules",
        "execution_cpu_energy_joules",
        "execution_ram_energy_joules",
    ]
    return all(is_finite_number(m.get(k)) for k in required)


def measure_exec_with_retry(fn, task_key, case, repeats, label, idle_power_w):
    last = None

    for attempt in range(1, MAX_MEASUREMENT_ATTEMPTS + 1):
        if attempt > 1:
            print(
                f"      energy sample invalid; waiting "
                f"{TRACKER_RETRY_DELAY_S}s and retrying "
                f"({attempt}/{MAX_MEASUREMENT_ATTEMPTS})..."
            )
            time.sleep(TRACKER_RETRY_DELAY_S)

        m = exp.measure_function_execution(
            fn,
            task_key,
            case,
            repeats,
            f"{label}_attempt_{attempt}",
            idle_power_w,
        )
        last = m

        if valid_execution_measurement(m):
            m["measurement_attempt"] = attempt
            return m

    return last


def copy_repaired_inference(old_row, new_row, repair_idle):
    # Preserve the run identity and the original generated artifact, but replace
    # the entire inference measurement with the successful replacement trial.
    keys = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "inference_elapsed_s",
        "tokens_per_sec",
        "ollama_load_duration_s",
        "inference_energy_kwh",
        "inference_energy_joules",
        "inference_cpu_energy_joules",
        "inference_gpu_energy_joules",
        "inference_ram_energy_joules",
        "inference_baseline_adjusted_energy_joules",
        "inference_steps",
    ]

    for key in keys:
        old_row[key] = new_row[key]

    old_row["inference_measurement_repaired"] = True
    old_row["inference_repair_timestamp"] = datetime.now().isoformat()
    old_row["inference_repair_idle_power_watts"] = repair_idle[
        "average_power_watts"
    ]


def copy_repaired_execution(row, llm_exec, human_exec, repair_idle):
    # Delete stale values from the failed pair before inserting the replacement.
    for key in list(row):
        if key.startswith("llm_execution_") or key.startswith("human_execution_"):
            row.pop(key, None)

    for key, value in llm_exec.items():
        row[f"llm_{key}"] = value

    for key, value in human_exec.items():
        row[f"human_{key}"] = value

    raw_llm_j = llm_exec["execution_energy_joules"]
    raw_human_j = human_exec["execution_energy_joules"]

    adj_llm_j = llm_exec["execution_baseline_adjusted_energy_joules"]
    adj_human_j = human_exec["execution_baseline_adjusted_energy_joules"]

    row["execution_energy_ratio_llm_vs_human"] = (
        raw_llm_j / raw_human_j if raw_human_j > 0 else None
    )
    row["execution_baseline_adjusted_energy_ratio_llm_vs_human"] = (
        adj_llm_j / adj_human_j if adj_human_j > 0 else None
    )
    row["execution_time_ratio_llm_vs_human"] = (
        llm_exec["execution_elapsed_s"] / human_exec["execution_elapsed_s"]
        if human_exec["execution_elapsed_s"] > 0
        else None
    )

    row["execution_measurement_repaired"] = True
    row["execution_repair_timestamp"] = datetime.now().isoformat()
    row["execution_repair_idle_power_watts"] = repair_idle[
        "average_power_watts"
    ]
    row.pop("execution_error", None)


def finite_values(rows, key):
    return [
        r[key]
        for r in rows
        if is_finite_number(r.get(key))
    ]


def safe_mean(values):
    return sum(values) / len(values) if values else None


def safe_median(values):
    values = sorted(values)
    if not values:
        return None
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def print_clean_summary(results):
    print("\n================ REPAIRED CATEGORY 1 SUMMARY ================")

    for task_key in exp.TASKS:
        print(f"\n--- {task_key} ---")

        for strategy in exp.STRATEGIES:
            rows = [
                r for r in results
                if r["task"] == task_key and r["strategy"] == strategy
            ]

            passed = sum(r["tests_passed"] for r in rows)
            total = sum(r["tests_total"] for r in rows)

            inf_j = finite_values(rows, "inference_energy_joules")
            inf_s = finite_values(rows, "inference_elapsed_s")
            toks = finite_values(rows, "total_tokens")
            ratios = finite_values(
                rows, "execution_energy_ratio_llm_vs_human"
            )
            time_ratios = finite_values(
                rows, "execution_time_ratio_llm_vs_human"
            )

            print(
                f"{strategy:<11} | tests {passed}/{total} | "
                f"valid inference {len(inf_j)}/5 | "
                f"valid execution pairs {len(ratios)}/5 | "
                f"avg tokens {safe_mean(toks):.1f} | "
                f"avg inf {safe_mean(inf_s):.3f}s | "
                f"inf energy mean/median "
                f"{safe_mean(inf_j):.2f}/{safe_median(inf_j):.2f} J | "
                f"exec energy ratio mean/median "
                f"{safe_mean(ratios):.3f}/{safe_median(ratios):.3f}x | "
                f"exec time ratio mean "
                f"{safe_mean(time_ratios):.3f}x"
            )


def main():
    with open(DATASET_PATH) as f:
        data = json.load(f)

    results = data["results"]

    bad_inf = [r for r in results if inference_is_bad(r)]
    bad_exec = [r for r in results if execution_pair_is_bad(r)]

    print("============================================================")
    print("QWEN CATEGORY 1 — MEASUREMENT REPAIR")
    print("============================================================")
    print(f"Invalid inference measurements found: {len(bad_inf)}")
    print(f"Invalid execution pairs found:       {len(bad_exec)}")
    print("\nThis script keeps all valid measurements and only repeats")
    print("the measurements that PowerMetrics/CodeCarbon failed to record.")
    print("\nLeave Mac idle.")

    input("\nPress Enter when ready... ")

    # --------------------------------------------------------
    # Repair inference measurements
    # --------------------------------------------------------
    if bad_inf:
        print("\n========== REPAIRING INFERENCE ==========")
        exp.preload_model()
        exp.warmup_model()
        time.sleep(10)

        inf_idle = exp.measure_idle_baseline(
            "qwen_inference_repair_idle_baseline"
        )
        data.setdefault("repair_idle_baselines", {})[
            "inference"
        ] = inf_idle
        save(data)

        print(
            f"Repair inference idle baseline: "
            f"{inf_idle['average_power_watts']:.3f} W"
        )

        for idx, row in enumerate(bad_inf, 1):
            print(
                f"\n[Inference repair {idx}/{len(bad_inf)}] "
                f"{row['task']} / {row['strategy']} / run {row['run']}"
            )

            successful = None

            for attempt in range(1, MAX_MEASUREMENT_ATTEMPTS + 1):
                if attempt > 1:
                    print(
                        f"  invalid energy sample; waiting "
                        f"{TRACKER_RETRY_DELAY_S}s and retrying..."
                    )
                    time.sleep(TRACKER_RETRY_DELAY_S)

                new_row = exp.run_inference_trial(
                    row["task"],
                    row["strategy"],
                    row["run"],
                    row["block"],
                    inf_idle["average_power_watts"],
                )

                if (
                    new_row["generated_code"] != row["generated_code"]
                    or new_row["generated_response"] != row["generated_response"]
                ):
                    raise RuntimeError(
                        "Repair regeneration produced a different response/code. "
                        "Stopped rather than silently pairing a new inference measurement "
                        "with the original artifact."
                    )

                if not inference_is_bad(new_row):
                    successful = new_row
                    print(
                        f"  valid replacement: "
                        f"{new_row['inference_elapsed_s']:.3f}s | "
                        f"{new_row['inference_energy_joules']:.2f} J"
                    )
                    break

            if successful is None:
                print("  FAILED after all attempts; original missing value retained.")
            else:
                copy_repaired_inference(
                    row,
                    successful,
                    inf_idle,
                )
                save(data)

    # --------------------------------------------------------
    # Repair execution pairs
    # --------------------------------------------------------
    # Ensure model is unloaded, then allow PowerMetrics to settle.
    print("\n========== PREPARING EXECUTION REPAIR ==========")
    try:
        exp.unload_model()
    except Exception:
        pass

    print(f"Settling for {POST_UNLOAD_SETTLE_S}s...")
    time.sleep(POST_UNLOAD_SETTLE_S)

    # One short, deliberately discarded tracker cycle helps avoid the
    # early-series PowerMetrics NaNs seen in the original run.
    print("Warming up the energy tracker (discarded measurement)...")
    _ = exp.measure_idle_baseline("qwen_execution_tracker_warmup")
    time.sleep(TRACKER_RETRY_DELAY_S)

    exec_idle = exp.measure_idle_baseline(
        "qwen_execution_repair_idle_baseline"
    )
    data.setdefault("repair_idle_baselines", {})[
        "execution"
    ] = exec_idle
    save(data)

    print(
        f"Repair execution idle baseline: "
        f"{exec_idle['average_power_watts']:.3f} W"
    )

    cases = {
        task_key: exp.make_execution_case(task_key)
        for task_key in exp.TASKS
    }

    # Reuse the original calibrated repeat counts exactly.
    repeats = {
        task_key: data["execution_calibration"][task_key]["repeats"]
        for task_key in exp.TASKS
    }

    # Recompute bad rows after inference repair; execution set is independent.
    bad_exec = [r for r in results if execution_pair_is_bad(r)]

    for idx, row in enumerate(bad_exec, 1):
        task_key = row["task"]
        strategy = row["strategy"]
        run_num = row["run"]

        print(
            f"\n[Execution repair {idx}/{len(bad_exec)}] "
            f"{task_key} / {strategy} / run {run_num}"
        )

        llm_fn = exp.get_generated_callable(
            row["generated_code"],
            task_key,
        )
        human_fn = exp.HUMAN_FUNCTIONS[task_key]
        case = cases[task_key]
        n_repeats = repeats[task_key]

        # Alternate order between repair pairs.
        if idx % 2 == 1:
            llm_exec = measure_exec_with_retry(
                llm_fn,
                task_key,
                case,
                n_repeats,
                f"{task_key}_{strategy}_run_{run_num}_llm_exec_repair",
                exec_idle["average_power_watts"],
            )
            time.sleep(TRACKER_RETRY_DELAY_S)
            human_exec = measure_exec_with_retry(
                human_fn,
                task_key,
                case,
                n_repeats,
                f"{task_key}_{strategy}_run_{run_num}_human_exec_repair",
                exec_idle["average_power_watts"],
            )
        else:
            human_exec = measure_exec_with_retry(
                human_fn,
                task_key,
                case,
                n_repeats,
                f"{task_key}_{strategy}_run_{run_num}_human_exec_repair",
                exec_idle["average_power_watts"],
            )
            time.sleep(TRACKER_RETRY_DELAY_S)
            llm_exec = measure_exec_with_retry(
                llm_fn,
                task_key,
                case,
                n_repeats,
                f"{task_key}_{strategy}_run_{run_num}_llm_exec_repair",
                exec_idle["average_power_watts"],
            )

        if (
            not valid_execution_measurement(llm_exec)
            or not valid_execution_measurement(human_exec)
        ):
            print("  FAILED to obtain a complete valid pair; original values retained.")
            continue

        copy_repaired_execution(
            row,
            llm_exec,
            human_exec,
            exec_idle,
        )
        save(data)

        print(
            f"  LLM   {llm_exec['execution_elapsed_s']:.3f}s | "
            f"{llm_exec['execution_energy_joules']:.2f} J"
        )
        print(
            f"  Human {human_exec['execution_elapsed_s']:.3f}s | "
            f"{human_exec['execution_energy_joules']:.2f} J"
        )
        print(
            f"  ratio "
            f"{row['execution_energy_ratio_llm_vs_human']:.3f}x"
        )

    # Final validation.
    remaining_bad_inf = [
        r for r in results if inference_is_bad(r)
    ]
    remaining_bad_exec = [
        r for r in results if execution_pair_is_bad(r)
    ]

    data["repair_summary"] = {
        "completed_at": datetime.now().isoformat(),
        "remaining_invalid_inference_measurements": len(remaining_bad_inf),
        "remaining_invalid_execution_pairs": len(remaining_bad_exec),
    }
    save(data)

    print_clean_summary(results)

    print("\n============================================================")
    print(f"Remaining invalid inference measurements: {len(remaining_bad_inf)}")
    print(f"Remaining invalid execution pairs:       {len(remaining_bad_exec)}")
    print(f"Saved repaired dataset to: {DATASET_PATH}")
    print("============================================================")


if __name__ == "__main__":
    main()
