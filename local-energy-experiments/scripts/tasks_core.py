"""Algorithm and data-processing tasks, prompts, and correctness checks."""

import re
from typing import Optional


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
        fn = (_find_fn(ns, ["twoSum", "two_sum", "two_pointer_algorithm"]) or
              _method_from_solution(ns, ["twoSum", "two_sum", "two_pointer_algorithm"]))
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
        fn = (_find_fn(ns, ["maxSubArray", "maxSubarray", "max_subarray", "max_sub_array", "maxSub", "kadane"]) or
              _method_from_solution(ns, ["maxSubArray", "maxSubarray", "max_subarray", "max_sub_array", "maxSub", "kadane"]))
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
