"""Retest saved Qwen weather solutions and recover their execution measurements."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import stage2_qwen_api_full as s

INPUT_JSON = s.OUTPUT_JSON
OUTPUT_JSON = str(Path(INPUT_JSON).with_name("stage2_qwen_api_full_corrected.json"))


def save(data):
    s.save_dataset(data)


def main():
    # Keep the input dataset separate from the corrected output.
    s.OUTPUT_JSON = OUTPUT_JSON
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Could not find {INPUT_JSON}")

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # ------------------------------------------------------------
    # 1. Correct the stored unit-test classifications.
    # ------------------------------------------------------------
    print("\n================ CORRECTING UNIT TEST RESULTS ================\n")

    corrected = 0

    for r in data["results"]:
        old_passed = r.get("tests_passed")
        old_total = r.get("tests_total")

        passed, total = s.run_unit_tests(
            r["generated_code"],
            r["task"],
        )

        r["tests_passed"] = passed
        r["tests_total"] = total

        # Remove the old skip caused by the broken mock hierarchy.
        if r.get("execution_error", "").startswith(
            "Skipped: generated code did not pass all unit tests."
        ):
            r.pop("execution_error", None)

        if (old_passed, old_total) != (passed, total):
            corrected += 1

        print(
            f"{r['task']:<16} {r['strategy']:<11} "
            f"run {r['run']} | "
            f"{old_passed}/{old_total} -> {passed}/{total}"
        )

    data.setdefault("metadata", {})
    data["metadata"]["category3_test_harness_correction"] = {
        "applied_at": datetime.now().isoformat(),
        "reason": (
            "Mock HTTPError was corrected to inherit from RequestException, "
            "matching the requests exception hierarchy."
        ),
        "inference_measurements_reused": True,
        "inference_regenerated": False,
        "corrected_trials": corrected,
    }

    save(data)

    print(
        f"\nCorrected {corrected} stored classifications."
        f"\nSaved corrected pre-execution dataset to: {OUTPUT_JSON}"
    )

    if not all(
        r["tests_passed"] == r["tests_total"]
        for r in data["results"]
    ):
        print(
            "\nAt least one solution still fails the corrected tests. "
            "Execution will only run for solutions that pass all tests."
        )

    # ------------------------------------------------------------
    # 2. Execution-only recovery.
    # ------------------------------------------------------------
    print("\n================ EXECUTION-ONLY RECOVERY ================")
    print("No inference will be regenerated.")
    print("The saved Qwen code will be benchmarked against the human baselines.")
    print("\nIMPORTANT: leave the machine alone during measured phases.")

    input("\nPress Enter when the machine is ready and you will leave it idle... ")

    print("\nEnsuring Qwen is unloaded...")
    try:
        s.unload_model()
    except Exception as exc:
        print("  Warning while unloading model:", exc)

    print(f"Settling for {s.SETTLE_SECONDS:.0f} seconds...")
    time.sleep(s.SETTLE_SECONDS)

    print(
        f"Measuring {s.IDLE_BASELINE_SECONDS:.0f}s execution-phase "
        "idle baseline..."
    )
    exec_idle = s.measure_idle_baseline(
        "qwen_api_execution_recovery_idle"
    )

    data.setdefault("idle_baselines", {})
    data["idle_baselines"]["execution_model_unloaded_corrected"] = exec_idle

    print(
        f"Execution idle baseline: "
        f"{exec_idle['average_power_watts']:.3f} W"
    )

    cases = {
        task_key: s.make_execution_case(task_key)
        for task_key in s.TASKS
    }

    repeats = {
        task_key: s.calibrate_repeats(task_key, cases[task_key])
        for task_key in s.TASKS
    }

    data["execution_calibration_corrected"] = {
        task_key: {
            "input_size": cases[task_key]["input_size"],
            "repeats": repeats[task_key],
            "target_seconds": s.TARGET_EXEC_SECONDS,
        }
        for task_key in s.TASKS
    }

    save(data)

    for task_key in s.TASKS:
        print(
            f"{task_key}: "
            f"input={cases[task_key]['input_size']:,}, "
            f"repeats={repeats[task_key]}"
        )

    total_results = len(data["results"])

    for i, result in enumerate(data["results"], 1):
        task_key = result["task"]
        strategy = result["strategy"]
        run_num = result["run"]

        print(
            f"\n[Execution {i}/{total_results}] "
            f"task={task_key} strategy={strategy} run={run_num}/5"
        )

        code = result.get("generated_code")

        if not code or result["tests_passed"] != result["tests_total"]:
            result["execution_error"] = (
                "Skipped: generated code did not pass corrected unit tests."
            )
            save(data)
            print("  skipped — did not pass corrected unit tests")
            continue

        # Clear any stale error before retrying this execution benchmark.
        result.pop("execution_error", None)

        try:
            llm_fn = s.get_generated_callable(code, task_key)
            human_fn = s.HUMAN_FUNCTIONS[task_key]
            case = cases[task_key]
            expected = case["expected"]

            # Unmeasured large-workload validation before energy measurement.
            llm_large_result = s.call_task(llm_fn, task_key, case)
            human_large_result = s.call_task(human_fn, task_key, case)

            if not s.result_is_correct(
                task_key, llm_large_result, expected
            ):
                raise RuntimeError(
                    "LLM code failed the large mock-API workload validation."
                )

            if not s.result_is_correct(
                task_key, human_large_result, expected
            ):
                raise RuntimeError(
                    "Human baseline failed the large mock-API workload validation."
                )

            n_repeats = repeats[task_key]

            # Alternate order to reduce order/thermal bias.
            if i % 2 == 1:
                llm_exec = s.measure_function_execution(
                    llm_fn,
                    task_key,
                    case,
                    n_repeats,
                    f"{task_key}_{strategy}_run_{run_num}_llm_exec_recovery",
                    exec_idle["average_power_watts"],
                )
                human_exec = s.measure_function_execution(
                    human_fn,
                    task_key,
                    case,
                    n_repeats,
                    f"{task_key}_{strategy}_run_{run_num}_human_exec_recovery",
                    exec_idle["average_power_watts"],
                )
            else:
                human_exec = s.measure_function_execution(
                    human_fn,
                    task_key,
                    case,
                    n_repeats,
                    f"{task_key}_{strategy}_run_{run_num}_human_exec_recovery",
                    exec_idle["average_power_watts"],
                )
                llm_exec = s.measure_function_execution(
                    llm_fn,
                    task_key,
                    case,
                    n_repeats,
                    f"{task_key}_{strategy}_run_{run_num}_llm_exec_recovery",
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
                raw_llm_j / raw_human_j
                if raw_human_j > 0 else None
            )

            result[
                "execution_baseline_adjusted_energy_ratio_llm_vs_human"
            ] = (
                adj_llm_j / adj_human_j
                if adj_human_j > 0 else None
            )

            result["execution_time_ratio_llm_vs_human"] = (
                llm_exec["execution_elapsed_s"]
                / human_exec["execution_elapsed_s"]
                if human_exec["execution_elapsed_s"] > 0
                else None
            )

            print(
                f"  LLM:   "
                f"{llm_exec['execution_elapsed_s']:.3f}s | "
                f"{raw_llm_j:.2f} J raw"
            )
            print(
                f"  Human: "
                f"{human_exec['execution_elapsed_s']:.3f}s | "
                f"{raw_human_j:.2f} J raw"
            )
            print(
                "  raw energy ratio LLM/Human: "
                f"{result['execution_energy_ratio_llm_vs_human']:.3f}x"
            )
            print(
                "  time ratio LLM/Human: "
                f"{result['execution_time_ratio_llm_vs_human']:.3f}x"
            )

        except Exception as exc:
            result["execution_error"] = str(exc)
            print("  execution benchmark error:", exc)

        save(data)
        time.sleep(2)

    print("\n================ CORRECTED FINAL SUMMARY ================")
    s.print_summary(data["results"])
    save(data)

    print(
        f"\nSaved corrected Category 3 dataset to: {OUTPUT_JSON}"
    )
    print("Original JSON was left untouched.")


if __name__ == "__main__":
    main()
