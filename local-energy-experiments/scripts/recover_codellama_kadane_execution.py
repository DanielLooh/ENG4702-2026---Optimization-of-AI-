"""Recover CodeLlama Maximum Subarray execution after function-name correction."""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import stage2_codellama_algorithmic_full as s


PATH = Path(s.OUTPUT_JSON)


def save(data):
    PATH.write_text(
        json.dumps(data, indent=2, allow_nan=False)
    )


def main():
    data = json.loads(PATH.read_text())

    targets = [
        (i, r)
        for i, r in enumerate(data["results"], 1)
        if r["task"] == "max_subarray"
        and r["strategy"] == "efficiency"
    ]

    if len(targets) != 5:
        raise SystemExit(
            f"Expected 5 max_subarray/efficiency results, found {len(targets)}"
        )

    # Re-evaluate the saved code with the corrected function-name lookup.
    for _, r in targets:
        passed, total = s.run_unit_tests(r["generated_code"], r["task"])
        if (passed, total) != (3, 3):
            raise SystemExit(f"Run {r['run']} does not pass the corrected tests.")
        r["tests_passed"], r["tests_total"] = passed, total

    print("Found 5 corrected max_subarray / efficiency runs.")
    print("No inference will be performed.")
    print("Only execution benchmarking will be recovered.")

    # Match original execution protocol
    case = s.make_execution_case("max_subarray")
    expected = case["expected"]

    original_repeats = data["execution_calibration"]["max_subarray"]["repeats"]

    print(f"Input size: {case['input_size']:,}")
    print(f"Reusing original repeats: {original_repeats}")

    if original_repeats != 237:
        raise SystemExit(
            f"Expected original max_subarray repeats=237, got {original_repeats}"
        )

    # Model must be unloaded during execution measurements
    print("\nUnloading CodeLlama...")
    s.unload_model()

    print(f"Settling for {s.SETTLE_SECONDS:.0f}s...")
    time.sleep(s.SETTLE_SECONDS)

    print(
        f"Measuring fresh {s.IDLE_BASELINE_SECONDS:.0f}s "
        "execution-phase idle baseline..."
    )

    recovery_idle = s.measure_idle_baseline(
        "codellama_kadane_recovery_execution_idle"
    )

    print(
        f"Recovery execution idle baseline: "
        f"{recovery_idle['average_power_watts']:.3f} W"
    )

    # Preserve original baseline and store recovery baseline separately.
    data["idle_baselines"][
        "execution_model_unloaded_kadane_recovery"
    ] = recovery_idle

    data.setdefault("corrections", []).append({
        "timestamp": datetime.now().isoformat(),
        "reason": (
            "Maximum Subarray Efficiency originally scored 0/3 because "
            "the generated function name 'kadane' was not recognized. "
            "After adding function-name compatibility, all five saved "
            "solutions scored 3/3. Inference was not rerun. Only the "
            "previously skipped execution measurements were recovered."
        ),
        "affected_condition": "max_subarray / efficiency",
        "inference_rerun": False,
        "execution_recovery": True,
        "execution_repeats": original_repeats,
        "recovery_idle_power_watts": recovery_idle["average_power_watts"],
    })

    save(data)

    # Recover ONLY the five previously skipped executions
    for global_index, result in targets:
        run_num = result["run"]

        print(
            f"\n[Recovery run {run_num}/5] "
            f"original execution index={global_index}/30"
        )

        try:
            llm_fn = s.get_generated_callable(
                result["generated_code"],
                "max_subarray",
            )

            human_fn = s.HUMAN_FUNCTIONS["max_subarray"]

            # Full-size validation using same benchmark input.
            llm_large_result = s.call_task_with_timeout(
                llm_fn,
                "max_subarray",
                case,
            )

            human_large_result = s.call_task(
                human_fn,
                "max_subarray",
                case,
            )

            if not s.result_is_correct(
                "max_subarray",
                llm_large_result,
                expected,
            ):
                raise RuntimeError(
                    f"LLM failed full-size validation: "
                    f"got {llm_large_result}, expected {expected}"
                )

            if not s.result_is_correct(
                "max_subarray",
                human_large_result,
                expected,
            ):
                raise RuntimeError(
                    "Human baseline failed full-size validation."
                )

            mutates = s.function_mutates_input(
                llm_fn,
                "max_subarray",
                case,
            )

            result["generated_code_mutates_input"] = mutates

            if mutates:
                raise RuntimeError(
                    "Generated code mutates benchmark input."
                )

            # Preserve the original alternating LLM/Human ordering.
            if global_index % 2 == 1:
                llm_exec = s.measure_function_execution(
                    llm_fn,
                    "max_subarray",
                    case,
                    original_repeats,
                    f"codellama_kadane_recovery_run_{run_num}_llm",
                    recovery_idle["average_power_watts"],
                )

                human_exec = s.measure_function_execution(
                    human_fn,
                    "max_subarray",
                    case,
                    original_repeats,
                    f"codellama_kadane_recovery_run_{run_num}_human",
                    recovery_idle["average_power_watts"],
                )

            else:
                human_exec = s.measure_function_execution(
                    human_fn,
                    "max_subarray",
                    case,
                    original_repeats,
                    f"codellama_kadane_recovery_run_{run_num}_human",
                    recovery_idle["average_power_watts"],
                )

                llm_exec = s.measure_function_execution(
                    llm_fn,
                    "max_subarray",
                    case,
                    original_repeats,
                    f"codellama_kadane_recovery_run_{run_num}_llm",
                    recovery_idle["average_power_watts"],
                )

            for key, value in llm_exec.items():
                result[f"llm_{key}"] = value

            for key, value in human_exec.items():
                result[f"human_{key}"] = value

            raw_llm = llm_exec["execution_energy_joules"]
            raw_human = human_exec["execution_energy_joules"]

            adj_llm = llm_exec[
                "execution_baseline_adjusted_energy_joules"
            ]
            adj_human = human_exec[
                "execution_baseline_adjusted_energy_joules"
            ]

            result["execution_energy_ratio_llm_vs_human"] = (
                raw_llm / raw_human
            )

            result[
                "execution_baseline_adjusted_energy_ratio_llm_vs_human"
            ] = (
                adj_llm / adj_human
                if adj_human > 0
                else None
            )

            result["execution_time_ratio_llm_vs_human"] = (
                llm_exec["execution_elapsed_s"]
                / human_exec["execution_elapsed_s"]
            )

            result.pop("execution_error", None)

            print(
                f"  LLM:   {llm_exec['execution_elapsed_s']:.3f}s | "
                f"{raw_llm:.2f} J raw"
            )

            print(
                f"  Human: {human_exec['execution_elapsed_s']:.3f}s | "
                f"{raw_human:.2f} J raw"
            )

            print(
                f"  raw energy ratio: "
                f"{result['execution_energy_ratio_llm_vs_human']:.3f}x"
            )

            print(
                f"  time ratio: "
                f"{result['execution_time_ratio_llm_vs_human']:.3f}x"
            )

        except Exception as exc:
            result["execution_error"] = (
                f"Execution recovery failed: {exc}"
            )
            save(data)
            raise

        # Save immediately after each recovered run.
        save(data)
        time.sleep(2)


    print("\n========================================")
    print("EXECUTION RECOVERY COMPLETE")
    print("========================================")

    energy_ratios = [
        r["execution_energy_ratio_llm_vs_human"]
        for _, r in targets
    ]

    time_ratios = [
        r["execution_time_ratio_llm_vs_human"]
        for _, r in targets
    ]


    print(
        "Max Subarray Efficiency raw energy ratio "
        f"mean/median: "
        f"{statistics.mean(energy_ratios):.3f}/"
        f"{statistics.median(energy_ratios):.3f}x"
    )

    print(
        "Max Subarray Efficiency time ratio "
        f"mean/median: "
        f"{statistics.mean(time_ratios):.3f}/"
        f"{statistics.median(time_ratios):.3f}x"
    )

    print("\nUpdated:")
    print(PATH)
    print("Inference measurements were NOT changed.")


if __name__ == "__main__":
    main()
