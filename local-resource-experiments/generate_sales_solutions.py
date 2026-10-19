#!/usr/bin/env python3
"""Generate and validate the frozen programs used by Experiment 4.

This separates inference (Experiments 1/5) from generated-code execution
(Experiment 4) while retaining the exact model response as an audit trail.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from common import call_ollama, write_json

PROMPT = """You are an expert software engineer. Provide Python code only, with no Markdown fences or explanation.

Write a function exactly named aggregate_sales(records). records is a list of tuples: (product: str, region: str, revenue: float). Return a dictionary with exactly these keys: total_revenue, top_product, top_region. total_revenue is the sum of all revenue. top_product is the product with the greatest total revenue across all regions; top_region is the region with the greatest total revenue across all products. Resolve ties lexicographically. Use one pass where possible and standard library only."""

VALIDATOR = r'''
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
p=Path(sys.argv[1]); spec=spec_from_file_location("solution",p); m=module_from_spec(spec); spec.loader.exec_module(m)
assert m.aggregate_sales([("A","X",3.0),("A","Y",4.0),("B","X",6.0)]) == {"total_revenue":13.0,"top_product":"A","top_region":"X"}
assert m.aggregate_sales([]) == {"total_revenue":0,"top_product":None,"top_region":None}
'''


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = call_ollama(args.model, PROMPT, keep_alive="0", num_predict=args.num_predict,
                      num_ctx=args.num_ctx, seed=args.seed)
    stem = args.model.replace(":", "_").replace("/", "_")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{stem}_sales_raw.json"
    code_path = args.output_dir / f"{stem}_sales_solution.py"
    write_json(raw_path, raw)
    code_path.write_text(extract_code(raw["response"]), encoding="utf-8")
    check = subprocess.run([sys.executable, "-c", VALIDATOR, str(code_path)], text=True, capture_output=True)
    metadata = {"model": args.model, "prompt": PROMPT, "raw_response": str(raw_path),
                "solution": str(code_path), "valid": check.returncode == 0,
                "validation_stdout": check.stdout, "validation_stderr": check.stderr}
    write_json(args.output_dir / f"{stem}_sales_validation.json", metadata)
    if check.returncode:
        raise SystemExit(f"Generated solution failed validation: {check.stderr}")
    print(code_path)


if __name__ == "__main__":
    main()
