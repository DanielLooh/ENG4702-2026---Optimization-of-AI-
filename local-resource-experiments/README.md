# Local resource, CPU and memory experiments

Rachana's supporting artifacts for *The True Cost of LLM-Generated Code: Correctness, Generation Cost, and Execution Efficiency of Local Coding Models*.

## Evidence included

| Files | Final-paper use |
| --- | --- |
| `results/exp1_gpu/*_summary.json` and `*_samples.csv` | RQ4 resource monitoring; Table VIII and supplementary Table A9 |
| `results/exp5/*_summary.json` and `*_samples.csv` | RQ4 idle/active model memory; Table VIII and supplementary Table A10 |
| `results/exp4/*_summary.json` and `*_samples.csv` | RQ2 Sales Aggregation CPU measurements; supplementary Table A5 |
| `generated/` | Frozen Sales Aggregation implementations, saved model responses and correctness checks |
| Python scripts and `requirements.txt` | Measurement and generation procedures |

The `exp1`, `exp4`, and `exp5` names are historical experiment identifiers, not research-question numbers. This folder is **not** a complete reproduction package for every result in the paper. The three-model inference/ execution-energy study, memory-scaling study and concurrency study are not included here. Hosted-model scripts and results are in the repository's `research/` directory.

## Provenance and interpretation

The 36 source files were selected from `full-research-paper-reproduction-package.zip` and preserved unchanged. `MANIFEST.json` records their SHA-256 hashes. The original package's outdated results narrative, old README, AI handoff prompt, macOS metadata, Python caches and separate Qwen-only prompt-strategy energy study are excluded.

Resource monitoring includes one generation per model. For example, the saved Qwen resource-monitoring response time is 47.427 seconds (47.43 seconds in the paper). GPU readings are system-level Apple unified-memory telemetry, not per-model VRAM. Do not add GPU-driver memory to process RSS. The idle-memory experiment is a separate run, so its measurements should not be interpreted as simultaneous with resource monitoring.

The CPU experiment includes CodeLlama's measured execution even though its solution failed the non-empty correctness check. Retain that evidence; its runtime is not an efficiency gain for a correct solution. Qwen and DeepSeek also fail an extra empty-input test recorded in the validation files; the measured workload uses non-empty records. `run_sales_workload.py` checks output keys, not full semantic correctness. Consult the validation JSON files separately.

The CPU monitor estimates core use from aggregate process CPU time (100% approximates one fully occupied logical core), rather than directly observing which physical cores executed the program. The original validation JSON files retain machine-local source paths for provenance; they are not portable setup commands.

## Setup

Use Python 3.11 or newer and a running local Ollama server. The GPU telemetry implementation requires macOS/Apple Silicon. Hardware, OS, background activity and Ollama/model versions affect results; reruns are not expected to produce identical measurements. The supplied dependency file gives compatible ranges, not a locked environment.

From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
ollama pull deepseek-coder-v2:16b
ollama pull codellama:13b
```

Keep the original model tags: the saved DeepSeek runs use `deepseek-coder-v2:16b`. Do not silently substitute another tag or quantisation. The recorded generation settings are temperature 0, seed 42, context 2048 and output cap 256. Pulling a tag today does not guarantee the exact historical model bytes.

## Rerun without overwriting the submitted evidence

Run models sequentially and stop other loaded models first. These commands write to the ignored `reruns/` folder, leaving `results/` unchanged:

```sh
for model in qwen2.5-coder:7b deepseek-coder-v2:16b codellama:13b; do
  .venv/bin/python exp1_resource_monitor.py --model "$model" --output-dir reruns/exp1_gpu
  .venv/bin/python exp5_memory_footprint.py --model "$model" --output-dir reruns/exp5
  ollama stop "$model"
done
```

To repeat the CPU measurement on the frozen submitted programs:

```sh
for stem in qwen2.5-coder_7b deepseek-coder-v2_16b codellama_13b; do
  .venv/bin/python exp4_cpu_core_utilisation.py --label "${stem}_sales_aggregation" --output-dir reruns/exp4 -- .venv/bin/python run_sales_workload.py "generated/${stem}_sales_solution.py"
done
```

The workload constructs one million seeded sales records and calls the aggregation function eight times. The CodeLlama run is retained for measurement comparison despite its correctness failure; it must not be presented as a successful solution.

`generate_sales_solutions.py` can generate new candidates into `reruns/generated/`, but new output is not the frozen code underlying the submitted measurements. Do not overwrite `generated/` when reproducing the existing evidence.
