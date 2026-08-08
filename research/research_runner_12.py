"""
AI Prompt Strategy Research Runner
===================================
Three prompt strategies × 2 LeetCode problems × 5 runs × 2 models.
Total API calls: ~80 (two-step makes 2 calls per run).

Category  : LeetCode — Algorithmic Problems
Strategies: Basic | Efficiency-Focused | Two-Step (generate + optimize)
Models    : OpenAI gpt-5.4-mini  ×  Anthropic claude-haiku-4-5
"""

import os
import re
import time
import json
import statistics
from datetime import datetime
from typing import Optional

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


# ── Configuration ──────────────────────────────────────────────────────────────

OPENAI_API_KEY    = ''
ANTHROPIC_API_KEY = ''

OPENAI_MODEL    = "gpt-5.4-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"
TEMPERATURE     = 0.0
MAX_TOKENS      = 4096
N_RUNS          = 5
CALL_DELAY_S    = 0.4   # polite gap between API calls

OPENAI_PRICING = {
    "gpt-5.5":      {"input": 5.00,  "output": 30.00},
    "gpt-5.4":      {"input": 2.50,  "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75,  "output":  4.50},
    "gpt-5.4-nano": {"input": 0.15,  "output":  0.60},
}
ANTHROPIC_PRICING = {
    "claude-opus-4-5":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5": {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":  {"input":  1.00, "output":  5.00},
}


# ── Problem Definitions ────────────────────────────────────────────────────────

PROBLEMS = {
    "two_sum": {
        "name": "Two Sum",
        "difficulty": "Easy",
        "optimal": "Hash map — O(n) time, O(n) space",
        "baseline_exec_us": 1.42,
        "statement": (
            "Given an array of integers nums and an integer target, return indices of the\n"
            "two numbers such that they add up to target. You may assume that each input\n"
            "would have exactly one solution, and you may not use the same element twice.\n\n"
            "Example:\n  Input:  nums = [2,7,11,15], target = 9\n  Output: [0,1]"
        ),
    },
    "max_subarray": {
        "name": "Maximum Subarray",
        "difficulty": "Medium",
        "optimal": "Kadane's Algorithm — O(n) time, O(1) space",
        "baseline_exec_us": 1.42,
        "statement": (
            "Given an integer array nums, find the contiguous subarray (containing at least\n"
            "one number) which has the largest sum and return its sum.\n\n"
            "Example:\n  Input:  nums = [-2,1,-3,4,-1,2,1,-5,4]\n  Output: 6"
        ),
    },
    "sales_aggregation": {
        "name": "Sales Data Aggregation",
        "difficulty": "Easy",
        "category": "Kaggle Data Processing",
        "optimal": "Single-pass aggregation — O(n) time, O(k) space",
        "baseline_exec_us": None,
        "statement": (
            "Given a list of sales records (each a dict with keys 'product', 'region', 'revenue'),\n"
            "write a function aggregate_sales(records) that performs a single-pass aggregation to:\n"
            "  1. Compute total revenue per product\n"
            "  2. Compute total revenue per region\n"
            "  3. Identify the top-performing product (highest total revenue)\n"
            "  4. Identify the top-performing region (highest total revenue)\n\n"
            "Return a dict with keys:\n"
            "  - 'product_totals': dict mapping product name to total revenue\n"
            "  - 'region_totals':  dict mapping region name to total revenue\n"
            "  - 'top_product': str\n"
            "  - 'top_region':  str\n\n"
            "Sample data:\n"
            "  records = [\n"
            '      {"product": "Widget", "region": "North", "revenue": 500},\n'
            '      {"product": "Gadget", "region": "South", "revenue": 300},\n'
            '      {"product": "Widget", "region": "South", "revenue": 200},\n'
            '      {"product": "Gadget", "region": "North", "revenue": 100},\n'
            '      {"product": "Widget", "region": "North", "revenue": 150},\n'
            "  ]\n"
            "  # Expected:\n"
            '  # {"product_totals": {"Widget": 850, "Gadget": 400},\n'
            '  #  "region_totals":  {"North": 750, "South": 500},\n'
            '  #  "top_product": "Widget", "top_region": "North"}\n\n'
            "Optimal solution: single-pass aggregation — O(n) time, O(k) space."
        ),
    },
    "data_cleaning": {
        "name": "Data Cleaning and Filtering",
        "difficulty": "Medium",
        "category": "Kaggle Data Processing",
        "optimal": "Single-pass with hash set — O(n log n) time, O(n) space",
        "baseline_exec_us": None,
        "statement": (
            "Given a list of records (each a dict), write a function clean_data(records) that:\n"
            "  1. Removes records containing any None/null values or empty strings in any field\n"
            "  2. Removes exact duplicate records (same values for all fields)\n"
            "  3. Filters to only records with 'score' >= 50\n"
            "  4. Returns the results sorted by 'score' in descending order\n\n"
            "Sample data:\n"
            "  records = [\n"
            '      {"name": "Alice", "score": 85, "category": "A"},\n'
            '      {"name": "Bob",   "score": 45, "category": "B"},\n'
            '      {"name": "Alice", "score": 85, "category": "A"},\n'
            '      {"name": "",      "score": 70, "category": "C"},\n'
            '      {"name": "Dave",  "score": None, "category": "D"},\n'
            '      {"name": "Eve",   "score": 60,   "category": "E"},\n'
            '      {"name": "Frank", "score": 75,   "category": "F"},\n'
            "  ]\n"
            "  # Expected:\n"
            '  # [{"name": "Alice", "score": 85, "category": "A"},\n'
            '  #  {"name": "Frank", "score": 75, "category": "F"},\n'
            '  #  {"name": "Eve",   "score": 60, "category": "E"}]\n\n'
            "Optimal solution: single-pass with hash set for duplicate detection — "
            "O(n log n) time, O(n) space."
        ),
    },
}

UNIT_TESTS = {
    "two_sum": [
        ({"nums": [2, 7, 11, 15], "target": 9},  [0, 1]),
        ({"nums": [3, 2, 4],      "target": 6},  [1, 2]),
        ({"nums": [3, 3],         "target": 6},  [0, 1]),
    ],
    "max_subarray": [
        ([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6),
        ([1],                              1),
        ([5, 4, -1, 7, 8],                23),
    ],
}


# ── Prompt Builders ────────────────────────────────────────────────────────────

def prompt_basic(problem: dict):
    return (
        "You are an expert software engineer. Provide a Python solution.",
        f"Solve this:\n\n{problem['statement']}",
    )

def prompt_efficiency(problem: dict):
    return (
        "You are an expert software engineer specializing in algorithm optimization.",
        (
            f"Solve this problem with the most efficient approach:\n\n{problem['statement']}\n\n"
            "Optimize for: runtime and memory\n\n"
            "Provide:\n"
            "- Algorithm name\n"
            "- Time/Space complexity\n"
            "- Python code\n"
            "- Why this is optimal"
        ),
    )

def prompt_two_step_generate(problem: dict):
    return (
        "You are an expert software engineer. Provide a Python solution.",
        f"Solve this:\n\n{problem['statement']}",
    )

def prompt_two_step_optimize(prev_response: str):
    return (
        "You are an expert software engineer specializing in optimization.",
        (
            f"Here's my solution:\n\n{prev_response}\n\n"
            "Optimize this for runtime and memory. Make ONE pass of improvements — "
            "no major rewrites. Return the optimized code with a brief explanation of what changed."
        ),
    )


# ── API Helpers ────────────────────────────────────────────────────────────────

def estimate_cost(model: str, in_tok: int, out_tok: int, table: dict) -> Optional[float]:
    p = table.get(model)
    if not p:
        return None
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000


def call_openai(system: str, user: str) -> dict:
    if openai is None:
        raise RuntimeError("openai package not installed")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
    )
    elapsed = time.time() - t0
    u = resp.usage
    out_tok = u.completion_tokens
    return {
        "text":           resp.choices[0].message.content,
        "model":          resp.model,
        "input_tokens":   u.prompt_tokens,
        "output_tokens":  out_tok,
        "total_tokens":   u.total_tokens,
        "elapsed_s":      round(elapsed, 3),
        "tokens_per_sec": round(out_tok / elapsed, 1) if elapsed > 0 else 0,
        "cost":           estimate_cost(OPENAI_MODEL, u.prompt_tokens, out_tok, OPENAI_PRICING),
    }


def call_anthropic(system: str, user: str) -> dict:
    if anthropic is None:
        raise RuntimeError("anthropic package not installed")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    t0 = time.time()
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    elapsed = time.time() - t0
    u = resp.usage
    out_tok = u.output_tokens
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    return {
        "text":           text,
        "model":          resp.model,
        "input_tokens":   u.input_tokens,
        "output_tokens":  out_tok,
        "total_tokens":   u.input_tokens + out_tok,
        "elapsed_s":      round(elapsed, 3),
        "tokens_per_sec": round(out_tok / elapsed, 1) if elapsed > 0 else 0,
        "cost":           estimate_cost(ANTHROPIC_MODEL, u.input_tokens, out_tok, ANTHROPIC_PRICING),
    }


def call_model(provider: str, system: str, user: str) -> dict:
    r = call_openai(system, user) if provider == "openai" else call_anthropic(system, user)
    time.sleep(CALL_DELAY_S)
    return r


def merge_two_step(r1: dict, r2: dict) -> dict:
    tot_elapsed = r1["elapsed_s"] + r2["elapsed_s"]
    tot_out     = r1["output_tokens"] + r2["output_tokens"]
    return {
        "text":             r2["text"],
        "model":            r1["model"],
        "input_tokens":     r1["input_tokens"] + r2["input_tokens"],
        "output_tokens":    tot_out,
        "total_tokens":     r1["total_tokens"] + r2["total_tokens"],
        "elapsed_s":        round(tot_elapsed, 3),
        "tokens_per_sec":   round(tot_out / tot_elapsed, 1) if tot_elapsed > 0 else 0,
        "cost":             (r1.get("cost") or 0) + (r2.get("cost") or 0),
        "step1_total_tok":  r1["total_tokens"],
        "step2_total_tok":  r2["total_tokens"],
    }


# ── Code Extraction & Unit Testing ────────────────────────────────────────────

def extract_code(text: str) -> Optional[str]:
    """Return the best Python code block from a markdown response."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return None
    for b in blocks:
        if re.search(r"^\s*(def |class )", b, re.MULTILINE):
            return b.strip()
    return blocks[0].strip()


def _find_fn(ns: dict, names: list):
    """Return first callable found in ns matching any name in the list."""
    for name in names:
        if name in ns and callable(ns[name]):
            return ns[name]
    return None


def _method_from_solution(ns: dict, method_names: list):
    """Instantiate Solution() and return first matching method, or None."""
    if "Solution" not in ns:
        return None
    try:
        obj = ns["Solution"]()
        for name in method_names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn
    except Exception:
        pass
    return None


# ── Kaggle test helpers ───────────────────────────────────────────────────────

def _extract_top(result: dict, kind: str):
    """Pull top_product or top_region from a result dict, trying many key patterns."""
    for k in (f"top_{kind}", f"best_{kind}", f"top_performing_{kind}",
              f"highest_{kind}", f"max_{kind}"):
        if k in result:
            return result[k]
    # Fallback: derive from totals dict
    for k in (f"{kind}_totals", f"{kind}s", f"{kind}_revenue"):
        sub = result.get(k)
        if isinstance(sub, dict) and sub:
            return max(sub, key=sub.get)
    return None


def _test_sales_aggregation(ns: dict) -> tuple:
    NAMES = ["aggregate_sales", "process_sales", "analyze_sales",
             "compute_revenue", "sales_aggregation", "aggregate"]
    fn = (_find_fn(ns, NAMES) or
          _method_from_solution(ns, NAMES))
    if fn is None:
        return 0, 3

    cases = [
        (
            [
                {"product": "Widget", "region": "North", "revenue": 500},
                {"product": "Gadget", "region": "South", "revenue": 300},
                {"product": "Widget", "region": "South", "revenue": 200},
                {"product": "Gadget", "region": "North", "revenue": 100},
                {"product": "Widget", "region": "North", "revenue": 150},
            ],
            ("Widget", "North", "Widget", 850),   # top_prod, top_reg, prod_name, prod_total
        ),
        (
            [
                {"product": "Alpha", "region": "East", "revenue": 1000},
                {"product": "Beta",  "region": "West", "revenue": 500},
                {"product": "Alpha", "region": "West", "revenue": 200},
            ],
            ("Alpha", "East", "Alpha", 1200),
        ),
        (
            [{"product": "Solo", "region": "Lone", "revenue": 999}],
            ("Solo", "Lone", "Solo", 999),
        ),
    ]

    passed = 0
    for records, (exp_tp, exp_tr, check_prod, check_val) in cases:
        try:
            r = fn([dict(x) for x in records])
            if not isinstance(r, dict):
                continue
            top_p = _extract_top(r, "product")
            top_r = _extract_top(r, "region")
            # Also verify one exact product total
            prod_totals = r.get("product_totals") or r.get("products") or r.get("product_revenue") or {}
            total_ok = prod_totals.get(check_prod) == check_val
            if top_p == exp_tp and top_r == exp_tr and total_ok:
                passed += 1
        except Exception:
            pass
    return passed, 3


def _test_data_cleaning(ns: dict) -> tuple:
    NAMES = ["clean_data", "clean_and_filter", "process_records", "filter_records",
             "clean_records", "filter_data", "clean_filter", "clean"]
    fn = (_find_fn(ns, NAMES) or
          _method_from_solution(ns, NAMES))
    if fn is None:
        return 0, 3

    cases = [
        (
            [
                {"name": "Alice", "score": 85, "category": "A"},
                {"name": "Bob",   "score": 45, "category": "B"},
                {"name": "Alice", "score": 85, "category": "A"},   # dup
                {"name": "",      "score": 70, "category": "C"},   # empty string
                {"name": "Dave",  "score": None, "category": "D"}, # null
                {"name": "Eve",   "score": 60,   "category": "E"},
                {"name": "Frank", "score": 75,   "category": "F"},
            ],
            [85, 75, 60],
        ),
        (
            [
                {"name": "X", "score": 90, "cat": "a"},
                {"name": "Y", "score": 40, "cat": "b"},  # score < 50
                {"name": "Z", "score": 55, "cat": "c"},
                {"name": "X", "score": 90, "cat": "a"},  # dup
            ],
            [90, 55],
        ),
        (
            [
                {"name": "P", "score": 70, "cat": "x"},
                {"name": "P", "score": 70, "cat": "x"},  # dup
                {"name": "Q", "score": 80, "cat": "y"},
                {"name": "R", "score": 50, "cat": "z"},
            ],
            [80, 70, 50],
        ),
    ]

    passed = 0
    for records, expected_scores in cases:
        try:
            result = fn([dict(x) for x in records])
            scores = [r["score"] for r in result]
            if scores == expected_scores:
                passed += 1
        except Exception:
            pass
    return passed, 3


def run_unit_tests(code: Optional[str], prob_key: str) -> tuple:
    """Execute extracted code against predefined test cases. Returns (passed, total)."""
    if not code:
        return 0, 3

    full_code = "from typing import *\n" + code
    ns = {}
    try:
        exec(compile(full_code, "<generated>", "exec"), ns)
    except Exception:
        return 0, 3

    if prob_key == "two_sum":
        tests = UNIT_TESTS["two_sum"]
        fn = (_find_fn(ns, ["twoSum", "two_sum"]) or
              _method_from_solution(ns, ["twoSum", "two_sum"]))
        if fn is None:
            return 0, len(tests)
        passed = 0
        for kwargs, expected in tests:
            try:
                if sorted(fn(list(kwargs["nums"]), kwargs["target"])) == sorted(expected):
                    passed += 1
            except Exception:
                pass
        return passed, len(tests)

    elif prob_key == "max_subarray":
        tests = UNIT_TESTS["max_subarray"]
        fn = (_find_fn(ns, ["maxSubArray", "maxSubarray", "max_subarray", "maxSub"]) or
              _method_from_solution(ns, ["maxSubArray", "maxSubarray"]))
        if fn is None:
            return 0, len(tests)
        passed = 0
        for nums, expected in tests:
            try:
                if fn(list(nums)) == expected:
                    passed += 1
            except Exception:
                pass
        return passed, len(tests)

    elif prob_key == "sales_aggregation":
        return _test_sales_aggregation(ns)

    elif prob_key == "data_cleaning":
        return _test_data_cleaning(ns)

    return 0, 3


# ── Trial Runner ───────────────────────────────────────────────────────────────

def run_trial(strategy: str, problem: dict, prob_key: str, provider: str) -> dict:
    if strategy == "basic":
        sys_p, usr_p = prompt_basic(problem)
        r = call_model(provider, sys_p, usr_p)

    elif strategy == "efficiency":
        sys_p, usr_p = prompt_efficiency(problem)
        r = call_model(provider, sys_p, usr_p)

    elif strategy == "two_step":
        sys_p1, usr_p1 = prompt_two_step_generate(problem)
        r1 = call_model(provider, sys_p1, usr_p1)
        sys_p2, usr_p2 = prompt_two_step_optimize(r1["text"])
        r2 = call_model(provider, sys_p2, usr_p2)
        r = merge_two_step(r1, r2)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    code = extract_code(r["text"])
    passed, total = run_unit_tests(code, prob_key)
    r["tests_passed"] = passed
    r["tests_total"]  = total
    return r


# ── Report Formatter ───────────────────────────────────────────────────────────

STRATEGIES      = ["basic", "efficiency", "two_step"]
PROVIDERS       = ["openai", "anthropic"]
STRAT_LABELS    = {"basic": "Basic", "efficiency": "Efficiency", "two_step": "Two-Step"}
PROVIDER_LABELS = {
    "openai":    f"OpenAI ({OPENAI_MODEL})",
    "anthropic": f"Anthropic ({ANTHROPIC_MODEL})",
}

def _avg(vals):
    return statistics.mean(vals) if vals else 0

def _ok(runs):
    return [r for r in runs if "error" not in r]


def format_report(all_results: dict, start_time: float) -> str:
    W = 102
    lines = []

    def hr(c="═"):  return c * W
    def sec(t):     lines.extend([f"\n{hr('─')}", f"  {t}", hr("─")])

    lines += [
        f"\n{hr()}",
        "  AI PROMPT STRATEGY RESEARCH REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Wall time : {round(time.time() - start_time, 1)}s",
        hr(),
        "",
        f"  Categories : Cat 1 — LeetCode Algorithmic  |  Cat 2 — Kaggle Data Processing",
        f"  Models     : OpenAI {OPENAI_MODEL}  ×  Anthropic {ANTHROPIC_MODEL}",
        f"  Strategies : Basic  |  Efficiency-Focused  |  Two-Step (generate + optimize)",
        f"  Trials/cell: {N_RUNS}   Temperature: {TEMPERATURE}   Max tokens: {MAX_TOKENS}",
    ]

    # ── Per-problem sections ─────────────────────────────────────────────────
    for prob_key, prob in PROBLEMS.items():
        cat = prob.get("category", "LeetCode Algorithmic")
        baseline = (f"{prob['baseline_exec_us']} µs"
                    if prob.get("baseline_exec_us") is not None else "N/A")
        sec(f"PROBLEM: {prob['name']}  [{prob['difficulty']}]  [{cat}]  —  {prob['optimal']}")
        lines.append(f"  Baseline execution: {baseline}\n")

        for provider in PROVIDERS:
            lines.append(f"  ── {PROVIDER_LABELS[provider]}")
            hdr = (
                f"  {'Strategy':<18} {'Runs':>5}  {'Tests':>8}  "
                f"{'Avg In':>8}  {'Avg Out':>8}  {'Avg Tot':>9}  "
                f"{'Avg Lat(s)':>10}  {'Avg Tok/s':>10}  {'Avg Cost':>12}"
            )
            lines.append(hdr)
            lines.append("  " + "─"*18 + " " + "─"*5 + "  " + "─"*8 + "  " +
                         "─"*8 + "  " + "─"*8 + "  " + "─"*9 + "  " +
                         "─"*10 + "  " + "─"*10 + "  " + "─"*12)

            for strat in STRATEGIES:
                runs = _ok(all_results.get((prob_key, provider, strat), []))
                if not runs:
                    lines.append(f"  {STRAT_LABELS[strat]:<18}  (no successful runs)")
                    continue
                tp     = sum(r["tests_passed"] for r in runs)
                tt     = sum(r["tests_total"]  for r in runs)
                lines.append(
                    f"  {STRAT_LABELS[strat]:<18} {len(runs):>5}  {tp}/{tt:>5}  "
                    f"{_avg([r['input_tokens']   for r in runs]):>8.0f}  "
                    f"{_avg([r['output_tokens']  for r in runs]):>8.0f}  "
                    f"{_avg([r['total_tokens']   for r in runs]):>9.0f}  "
                    f"{_avg([r['elapsed_s']      for r in runs]):>10.3f}  "
                    f"{_avg([r['tokens_per_sec'] for r in runs]):>10.1f}  "
                    f"~${_avg([r['cost'] or 0   for r in runs]):>10.5f}"
                )
            lines.append("")

        # Per-run detail
        lines.append(f"  Detail — {prob['name']} (every run)\n")
        for provider in PROVIDERS:
            lines.append(f"  {PROVIDER_LABELS[provider]}")
            for strat in STRATEGIES:
                runs = all_results.get((prob_key, provider, strat), [])
                lines.append(f"\n    [{STRAT_LABELS[strat]}]  ({len(runs)} runs, "
                             f"{N_RUNS} planned)")
                lines.append(
                    f"    {'Run':<4}  {'In Tok':>7}  {'Out Tok':>7}  {'Total':>7}  "
                    f"{'Lat(s)':>7}  {'Tok/s':>7}  {'Cost':>12}  Tests"
                )
                lines.append(
                    "    " + "─"*4 + "  " + "─"*7 + "  " + "─"*7 + "  " +
                    "─"*7 + "  " + "─"*7 + "  " + "─"*7 + "  " + "─"*12 + "  " + "─"*5
                )
                for i, r in enumerate(runs, 1):
                    if "error" in r:
                        lines.append(f"    {i:<4}  ERROR: {r['error'][:70]}")
                        continue
                    cs = f"~${r['cost']:.5f}" if r.get("cost") is not None else "n/a"
                    lines.append(
                        f"    {i:<4}  {r['input_tokens']:>7}  {r['output_tokens']:>7}  "
                        f"{r['total_tokens']:>7}  {r['elapsed_s']:>7.3f}  "
                        f"{r['tokens_per_sec']:>7.1f}  {cs:>12}  "
                        f"{r['tests_passed']}/{r['tests_total']}"
                    )
            lines.append("")

    # ── Cross-strategy comparison ────────────────────────────────────────────
    sec("CROSS-STRATEGY COMPARISON — Averages Across Both Problems")

    for provider in PROVIDERS:
        lines.append(f"\n  {PROVIDER_LABELS[provider]}")
        lines.append(
            f"  {'Strategy':<18}  {'Avg In':>8}  {'Avg Out':>8}  {'Avg Total':>10}  "
            f"{'Avg Lat':>8}  {'Avg Tok/s':>10}  {'Avg Cost':>12}  Tests"
        )
        lines.append("  " + "─"*18 + "  " + "─"*8 + "  " + "─"*8 + "  " + "─"*10 +
                     "  " + "─"*8 + "  " + "─"*10 + "  " + "─"*12 + "  " + "─"*8)
        for strat in STRATEGIES:
            runs = []
            for pk in PROBLEMS:
                runs += _ok(all_results.get((pk, provider, strat), []))
            if not runs:
                continue
            tp = sum(r["tests_passed"] for r in runs)
            tt = sum(r["tests_total"]  for r in runs)
            lines.append(
                f"  {STRAT_LABELS[strat]:<18}  "
                f"{_avg([r['input_tokens']   for r in runs]):>8.0f}  "
                f"{_avg([r['output_tokens']  for r in runs]):>8.0f}  "
                f"{_avg([r['total_tokens']   for r in runs]):>10.0f}  "
                f"{_avg([r['elapsed_s']      for r in runs]):>8.3f}  "
                f"{_avg([r['tokens_per_sec'] for r in runs]):>10.1f}  "
                f"~${_avg([r['cost'] or 0   for r in runs]):>10.5f}  "
                f"{tp}/{tt}"
            )

    # ── Model comparison ─────────────────────────────────────────────────────
    sec("MODEL COMPARISON — OpenAI vs Anthropic (All Strategies & Problems Combined)")

    all_oai = [r for pk in PROBLEMS for st in STRATEGIES
               for r in _ok(all_results.get((pk, "openai", st), []))]
    all_ant = [r for pk in PROBLEMS for st in STRATEGIES
               for r in _ok(all_results.get((pk, "anthropic", st), []))]

    def cmp_row(label, ov, av, lower=True, fmt=".3f"):
        o, a = _avg(ov), _avg(av)
        if abs(o - a) < 1e-9:
            w = "tie"
        elif lower:
            w = "OpenAI" if o < a else "Anthropic"
        else:
            w = "OpenAI" if o > a else "Anthropic"
        return f"  {label:<26}  {format(o, fmt):<18}  {format(a, fmt):<18}  {w}"

    lines.append(f"\n  {'Metric':<26}  {'OpenAI':<18}  {'Anthropic':<18}  Winner")
    lines.append("  " + "─"*26 + "  " + "─"*18 + "  " + "─"*18 + "  " + "─"*12)
    lines.append(cmp_row("Avg input tokens",
                         [r["input_tokens"]   for r in all_oai],
                         [r["input_tokens"]   for r in all_ant], fmt=".0f"))
    lines.append(cmp_row("Avg output tokens",
                         [r["output_tokens"]  for r in all_oai],
                         [r["output_tokens"]  for r in all_ant], lower=False, fmt=".0f"))
    lines.append(cmp_row("Avg total tokens",
                         [r["total_tokens"]   for r in all_oai],
                         [r["total_tokens"]   for r in all_ant], fmt=".0f"))
    lines.append(cmp_row("Avg latency (s)",
                         [r["elapsed_s"]      for r in all_oai],
                         [r["elapsed_s"]      for r in all_ant]))
    lines.append(cmp_row("Avg throughput (tok/s)",
                         [r["tokens_per_sec"] for r in all_oai],
                         [r["tokens_per_sec"] for r in all_ant], lower=False, fmt=".1f"))
    lines.append(cmp_row("Avg cost/run ($)",
                         [r["cost"] or 0      for r in all_oai],
                         [r["cost"] or 0      for r in all_ant], fmt=".6f"))

    oai_tp = sum(r["tests_passed"] for r in all_oai)
    oai_tt = sum(r["tests_total"]  for r in all_oai)
    ant_tp = sum(r["tests_passed"] for r in all_ant)
    ant_tt = sum(r["tests_total"]  for r in all_ant)
    oai_rate = oai_tp / oai_tt if oai_tt else 0
    ant_rate = ant_tp / ant_tt if ant_tt else 0
    tw = "OpenAI" if oai_rate > ant_rate else ("Anthropic" if ant_rate > oai_rate else "tie")
    lines.append(
        f"  {'Test pass rate':<26}  {oai_tp}/{oai_tt} ({oai_rate:.0%})         "
        f"  {ant_tp}/{ant_tt} ({ant_rate:.0%})         "
        f"  {tw}"
    )

    lines.append(f"\n{hr()}\n")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    total_runs = len(PROBLEMS) * len(PROVIDERS) * len(STRATEGIES) * N_RUNS
    total_api  = total_runs + len(PROBLEMS) * len(PROVIDERS) * N_RUNS  # +1 per two_step run

    print(f"\n{'═'*70}")
    print(f"  Research Experiment  ({len(PROBLEMS)} problems, {len(PROVIDERS)} models, "
          f"{len(STRATEGIES)} strategies, {N_RUNS} runs each)")
    print(f"  Logical runs: {total_runs}   Est. API calls: {total_api}")
    print(f"  Cost estimate: < $0.20 (cheap models)")
    print(f"{'═'*70}\n")

    all_results = {}
    start_time  = time.time()
    run_num     = 0

    for prob_key, problem in PROBLEMS.items():
        for provider in PROVIDERS:
            for strat in STRATEGIES:
                key = (prob_key, provider, strat)
                all_results[key] = []
                for i in range(N_RUNS):
                    run_num += 1
                    tag = (f"[{run_num:>2}/{total_runs}] "
                           f"{problem['name']:<22} | {provider:<10} | "
                           f"{strat:<10} | run {i+1}")
                    print(f"  {tag}", end="\r", flush=True)
                    try:
                        r = run_trial(strat, problem, prob_key, provider)
                        all_results[key].append(r)
                        print(
                            f"  {tag}  "
                            f"lat={r['elapsed_s']}s  "
                            f"out={r['output_tokens']}  "
                            f"tests={r['tests_passed']}/{r['tests_total']}"
                        )
                    except Exception as e:
                        err_r = {
                            "error": str(e),
                            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                            "elapsed_s": 0, "tokens_per_sec": 0, "cost": 0,
                            "tests_passed": 0, "tests_total": 3,
                        }
                        all_results[key].append(err_r)
                        print(f"  {tag}  ERROR: {e}")

    report = format_report(all_results, start_time)
    print(report)

    here      = os.path.dirname(os.path.abspath(__file__))
    rpt_path  = os.path.join(here, "research_report.txt")
    json_path = os.path.join(here, "research_results.json")

    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)

    serializable = {
        f"{pk}__{pv}__{st}": [
            {k: v for k, v in r.items() if k not in ("text",)}
            for r in runs
        ]
        for (pk, pv, st), runs in all_results.items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"  Report saved : {rpt_path}")
    print(f"  Raw JSON     : {json_path}\n")


if __name__ == "__main__":
    main()
