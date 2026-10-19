#!/usr/bin/env python3
"""Deterministic Exp 4 workload for one validated generated sales solution."""

from __future__ import annotations

import argparse
import importlib.util
import random


def load_solution(path: str):
    spec = importlib.util.spec_from_file_location("generated_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.aggregate_sales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("solution")
    parser.add_argument("--records", type=int, default=1_000_000)
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args()
    randomizer = random.Random(20260928)
    products = [f"product_{i:03d}" for i in range(200)]
    regions = [f"region_{i:02d}" for i in range(30)]
    records = [(randomizer.choice(products), randomizer.choice(regions), randomizer.random() * 1000)
               for _ in range(args.records)]
    aggregate_sales = load_solution(args.solution)
    for _ in range(args.repetitions):
        output = aggregate_sales(records)
        assert set(output) == {"total_revenue", "top_product", "top_region"}


if __name__ == "__main__":
    main()
