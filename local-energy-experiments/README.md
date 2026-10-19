# Local energy experiment scripts

The scripts for the nine local experiments in the final-output folder.

All Python files are in `scripts/`:

- `stage2_{qwen,deepseek,codellama}_{algorithmic,data_processing,api}_full.py`:
  the nine experiment runners.
- `tasks_core.py` and `tasks_api.py`: shared specifications, prompts, tests,
  and weather mocks, extracted from `research_runner_12.py` and
  `research_runner_3.py`. Unused hosted-provider code has been removed.
- `repair_qwen_algorithmic_measurements.py`: repairs invalid algorithm readings.
- `stage2_qwen_api_execution_recovery.py`: retests saved weather code and
  recovers execution measurements without repeating inference.
- `recover_codellama_kadane_execution.py`: retests the five saved Maximum
  Subarray / Efficiency solutions using the corrected function-name lookup,
  then recovers their execution readings with the original 237 repetitions.
- `export_generated.py`: exports the exact saved generated code from result JSON
  into separate Python files, without calling a model.

Run from this folder with macOS/Apple Silicon, Ollama and PowerMetrics:

```sh
python3 -m pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
ollama pull deepseek-coder-v2:lite
ollama pull codellama:13b
python3 scripts/stage2_qwen_algorithmic_full.py
```

Install the corresponding Ollama model: `qwen2.5-coder:7b`,
`deepseek-coder-v2:lite`, or `codellama:13b`. Historical Python/Ollama versions
were 3.14.6 / 0.33.3. Leave the machine idle during energy measurements.

To run all nine experiments sequentially, with Ollama already running:

```sh
for script in scripts/stage2_*_full.py; do
  python3 "$script" || exit 1
done
```

Each runner uses two tasks, three strategies and five trials per condition.
JSON includes the generated response/code, exact prompts, tokens, correctness,
and inference/execution energy. Historical model digest prefixes were
`dae161e27b0e` (Qwen), `63fb193b3a9b` (DeepSeek), and `9f438cb9cd58`
(CodeLlama). Check installed weights against those identifiers: pulling a model
tag now does not guarantee the same weights. New generations, timing, and energy
measurements can differ from the historical outputs.

Outputs always go to `results/`, regardless of the working directory. Recovery
scripts read the corresponding JSON there; existing results can be overwritten.
No results or local environments are included in this folder.

The final output's Qwen API JSON contains the corrected results. Its recovery
script writes `stage2_qwen_api_full_corrected.json`; that file was renamed
`stage2_qwen_api_full.json` in the final-output folder.

The historical outputs also include selective Qwen algorithm energy repairs and
five recovered CodeLlama `kadane` executions. The current runners already use
the corrected tests; recovery is only needed to repair an existing result. With
the corresponding JSON in `results/`, use the relevant recovery script above.
The CodeLlama recovery is specific to the original 237-repeat calibration.

To recover the Python programs from an existing final-results folder exactly:

```sh
python3 scripts/export_generated.py path/to/final-results --output results/generated
```

The input can also be a single result JSON. The output directory must be new or
empty. Programs are grouped by model/task/strategy/run; extraction preserves the
saved code, including incorrect solutions. It does not rerun inference or repair
the generated program. The final JSON files themselves are not included here.

Cleanup changes are limited to organization, comments, unused hosted-provider
code, result paths, and recovery entry points. Prompts, model settings,
correctness criteria, and measurement calculations are preserved.

No provider keys or environment-variable configuration are needed. Weather
requests use a deterministic mock and an intentional `fake_key` placeholder.
