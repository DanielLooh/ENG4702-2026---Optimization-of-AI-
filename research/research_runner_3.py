"""
research_runner_cat3.py — Category 3: API Integration Tasks
============================================================
Task 5: Current Weather Fetch and Parse
Task 6: Multi-City Weather Comparison

5 runs × 3 strategies × 2 models = 30 logical runs (~40 API calls to OpenAI/Anthropic).
Unit tests use a mock requests module — no real OpenWeatherMap key needed.
Appends findings to research_report.txt.
"""

import os, re, sys, time, json, types, statistics, urllib.parse
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


# ── Config ─────────────────────────────────────────────────────────────────────

OPENAI_API_KEY    = ""
ANTHROPIC_API_KEY = ""

OPENAI_MODEL    = "gpt-5.4-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"
TEMPERATURE     = 0.0
MAX_TOKENS      = 4096
N_RUNS          = 5
CALL_DELAY_S    = 0.4

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

HERE        = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(HERE, "research_report.txt")


# ── Category 3 Problems ────────────────────────────────────────────────────────

PROBLEMS = {
    "weather_fetch": {
        "name": "Current Weather Fetch and Parse",
        "difficulty": "Easy",
        "category": "API Integration",
        "optimal": "Single API call with JSON parsing — O(1) time",
        "baseline_exec_us": None,
        "statement": (
            "Write a function fetch_weather(city, api_key) that:\n"
            "  1. Calls the OpenWeatherMap Current Weather API:\n"
            "       GET https://api.openweathermap.org/data/2.5/weather\n"
            "       params: q=city, appid=api_key, units=metric\n"
            "  2. Returns a dict with exactly these keys:\n"
            "       'city'        : str\n"
            "       'temperature' : float  (°C)\n"
            "       'feels_like'  : float  (°C)\n"
            "       'humidity'    : int    (%)\n"
            "       'description' : str\n"
            "       'wind_speed'  : float  (m/s)\n"
            "  3. On any HTTP or connection error returns {'error': <message>, 'city': city}\n\n"
            "Use the requests library. Handle all errors gracefully.\n"
            "Optimal: single API call with structured JSON parsing — O(1) time."
        ),
    },
    "weather_compare": {
        "name": "Multi-City Weather Comparison",
        "difficulty": "Medium",
        "category": "API Integration",
        "optimal": "Sequential API calls with exception handling — O(n) time",
        "baseline_exec_us": None,
        "statement": (
            "Write a function compare_weather(cities, api_key) that:\n"
            "  1. Fetches weather for each city via:\n"
            "       GET https://api.openweathermap.org/data/2.5/weather\n"
            "       params: q=city, appid=api_key, units=metric\n"
            "  2. Silently skips any city that errors (never raises)\n"
            "  3. Returns a dict with keys:\n"
            "       'results'  : list of dicts (city, temperature, feels_like,\n"
            "                    humidity, description, wind_speed)\n"
            "       'hottest'  : str  (city with highest temperature)\n"
            "       'coldest'  : str  (city with lowest temperature)\n"
            "       'avg_temp' : float (mean temperature of successful results)\n"
            "  4. Returns {'error': 'No data available', 'results': []} if all cities fail\n\n"
            "Use the requests library. Handle all errors gracefully.\n"
            "Optimal: sequential calls with exception handling — O(n) time."
        ),
    },
}


# ── Mock Requests Module ───────────────────────────────────────────────────────

def _owm_payload(city, temp, feels_like, humidity, description, wind_speed):
    return {
        "cod": 200, "name": city,
        "main": {"temp": temp, "feels_like": feels_like, "humidity": humidity,
                 "temp_min": temp - 1, "temp_max": temp + 1, "pressure": 1013},
        "weather": [{"id": 800, "main": "Clear", "description": description, "icon": "01d"}],
        "wind": {"speed": wind_speed, "deg": 200},
        "sys": {"country": "XX"}, "visibility": 10000,
    }

_CITY_DB = {
    "London": _owm_payload("London", 15.5, 14.0, 70, "light rain",  4.2),
    "Paris":  _owm_payload("Paris",  22.3, 21.8, 52, "clear sky",   2.1),
    "Berlin": _owm_payload("Berlin",  9.8,  8.5, 82, "overcast",    5.7),
    "Tokyo":  _owm_payload("Tokyo",  28.1, 30.0, 78, "few clouds",  3.3),
    "Sydney": _owm_payload("Sydney", 18.6, 18.0, 65, "sunny",       4.8),
}

class _HTTPError(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)
        self.response = None

class _RequestException(Exception): pass
class _ConnectionError(_RequestException): pass
class _Timeout(_RequestException): pass


def _build_mock_requests(city_data: dict, error_cities: set = None):
    error_cities = error_cities or set()

    class _Resp:
        def __init__(self, data, status_code=200):
            self._data   = data
            self.status_code = status_code
            self.text    = json.dumps(data)
            self.ok      = status_code < 400
        def json(self):          return self._data
        def raise_for_status(self):
            if self.status_code >= 400:
                raise _HTTPError(f"HTTP {self.status_code}")

    def _get(url, params=None, timeout=None, **kw):
        # Extract city from params dict or inline URL query string
        city = None
        if isinstance(params, dict):
            city = params.get("q") or params.get("city")
        if city is None:
            qs   = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            city = (qs.get("q") or qs.get("city") or [""])[0]
        city = str(city).strip()
        if city in error_cities or city not in city_data:
            return _Resp({"cod": "404", "message": "city not found"}, 404)
        return _Resp(city_data[city], 200)

    mod              = types.ModuleType("requests")
    mod.get          = _get
    mod.post         = lambda *a, **kw: _Resp({}, 200)
    exc              = types.ModuleType("requests.exceptions")
    exc.HTTPError    = _HTTPError
    exc.ConnectionError = _ConnectionError
    exc.Timeout      = _Timeout
    exc.RequestException = _RequestException
    mod.exceptions   = exc
    # flat aliases (from requests import HTTPError)
    mod.HTTPError    = _HTTPError
    mod.ConnectionError = _ConnectionError
    mod.Timeout      = _Timeout
    mod.RequestException = _RequestException
    return mod


def _exec_with_mock(code: str, mock) -> dict:
    """Exec code with mock injected into sys.modules so all import styles work."""
    saved     = {k: sys.modules[k] for k in ("requests", "requests.exceptions")
                 if k in sys.modules}
    sys.modules["requests"]            = mock
    sys.modules["requests.exceptions"] = mock.exceptions
    ns = {}
    try:
        exec(compile("from typing import *\n" + code, "<generated>", "exec"), ns)
    finally:
        for k, v in saved.items():
            sys.modules[k] = v
        for k in ("requests", "requests.exceptions"):
            if k not in saved:
                sys.modules.pop(k, None)
    return ns


# ── Unit Test Helpers ──────────────────────────────────────────────────────────

def _find_fn(ns, names):
    for n in names:
        if n in ns and callable(ns[n]):
            return ns[n]
    return None

def _method(ns, method_names):
    if "Solution" not in ns:
        return None
    try:
        obj = ns["Solution"]()
        for n in method_names:
            fn = getattr(obj, n, None)
            if callable(fn): return fn
    except Exception:
        pass
    return None

def _set_requests(ns, mock):
    """Update all request-related names in the namespace."""
    ns["requests"] = mock
    if "get" in ns and callable(ns.get("get")):
        ns["get"] = mock.get
    for exc_name in ("HTTPError", "ConnectionError", "Timeout", "RequestException"):
        if exc_name in ns:
            ns[exc_name] = getattr(mock, exc_name)


def _test_weather_fetch(code: str) -> tuple:
    NAMES = ["fetch_weather", "get_weather", "weather_fetch",
             "current_weather", "get_current_weather"]

    # ── Test 1: valid city, all 5 fields present ──────────────────────────────
    mock = _build_mock_requests(_CITY_DB)
    ns   = _exec_with_mock(code, mock)
    fn   = _find_fn(ns, NAMES) or _method(ns, NAMES)
    if fn is None:
        return 0, 3

    p1 = 0
    _set_requests(ns, mock)
    try:
        r    = fn("London", "fake_key")
        temp = r.get("temperature") or r.get("temp") or 0
        if (isinstance(r, dict) and "error" not in r
                and abs(temp - 15.5) < 0.5
                and r.get("humidity") == 70
                and r.get("feels_like") is not None
                and r.get("description") is not None
                and r.get("wind_speed") is not None):
            p1 = 1
    except Exception:
        pass

    # ── Test 2: different city — temperature correct ───────────────────────────
    p2 = 0
    _set_requests(ns, mock)
    try:
        r    = fn("Tokyo", "fake_key")
        temp = r.get("temperature") or r.get("temp") or 0
        if isinstance(r, dict) and "error" not in r and abs(temp - 28.1) < 0.5:
            p2 = 1
    except Exception:
        pass

    # ── Test 3: invalid city — graceful error (no exception raised) ────────────
    mock3 = _build_mock_requests(_CITY_DB, error_cities={"Atlantis"})
    _set_requests(ns, mock3)
    p3 = 0
    try:
        r = fn("Atlantis", "fake_key")
        if isinstance(r, dict):   # returned gracefully (with or without 'error' key)
            p3 = 1
    except Exception:
        pass  # crashed — test fails

    return p1 + p2 + p3, 3


def _test_weather_compare(code: str) -> tuple:
    NAMES = ["compare_weather", "fetch_multiple", "multi_weather", "compare_cities",
             "weather_comparison", "get_weather_comparison", "fetch_cities_weather"]

    mock = _build_mock_requests(_CITY_DB, error_cities={"BadCity", "X", "Y", "Z"})
    ns   = _exec_with_mock(code, mock)
    fn   = _find_fn(ns, NAMES) or _method(ns, NAMES)
    if fn is None:
        return 0, 3

    # ── Test 1: 3 valid cities — hottest / coldest correct ────────────────────
    # London=15.5, Paris=22.3, Berlin=9.8
    _set_requests(ns, mock)
    p1 = 0
    try:
        r       = fn(["London", "Paris", "Berlin"], "fake_key")
        hottest = r.get("hottest") or r.get("hottest_city") or r.get("max_city") or ""
        coldest = r.get("coldest") or r.get("coldest_city") or r.get("min_city") or ""
        if isinstance(r, dict) and hottest == "Paris" and coldest == "Berlin":
            p1 = 1
    except Exception:
        pass

    # ── Test 2: one failing city skipped, rest compared ───────────────────────
    # London=15.5, BadCity=error, Tokyo=28.1 → hot=Tokyo, cold=London, results=2
    _set_requests(ns, mock)
    p2 = 0
    try:
        r       = fn(["London", "BadCity", "Tokyo"], "fake_key")
        hottest = r.get("hottest") or r.get("hottest_city") or ""
        coldest = r.get("coldest") or r.get("coldest_city") or ""
        results = r.get("results", [])
        if (isinstance(r, dict) and hottest == "Tokyo"
                and coldest == "London" and len(results) == 2):
            p2 = 1
    except Exception:
        pass

    # ── Test 3: all cities fail — must return gracefully ──────────────────────
    _set_requests(ns, mock)
    p3 = 0
    try:
        r = fn(["X", "Y", "Z"], "fake_key")
        if isinstance(r, dict):   # didn't crash
            p3 = 1
    except Exception:
        pass

    return p1 + p2 + p3, 3


def run_unit_tests(code: Optional[str], prob_key: str) -> tuple:
    if not code:
        return 0, 3
    if prob_key == "weather_fetch":
        return _test_weather_fetch(code)
    elif prob_key == "weather_compare":
        return _test_weather_compare(code)
    return 0, 3


# ── Code Extractor ────────────────────────────────────────────────────────────

def extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return None
    for b in blocks:
        if re.search(r"^\s*(def |class )", b, re.MULTILINE):
            return b.strip()
    return blocks[0].strip()


# ── Prompt Builders ───────────────────────────────────────────────────────────

def prompt_basic(problem):
    return (
        "You are an expert software engineer. Provide a Python solution.",
        f"Solve this:\n\n{problem['statement']}",
    )

def prompt_efficiency(problem):
    return (
        "You are an expert software engineer specializing in API design and optimization.",
        (f"Solve this with the most efficient approach:\n\n{problem['statement']}\n\n"
         "Optimize for: minimal API calls, robust error handling, clean code\n\n"
         "Provide:\n- Algorithm/approach name\n- Time complexity\n- Python code\n"
         "- Why this is optimal"),
    )

def prompt_two_step_generate(problem):
    return (
        "You are an expert software engineer. Provide a Python solution.",
        f"Solve this:\n\n{problem['statement']}",
    )

def prompt_two_step_optimize(prev):
    return (
        "You are an expert software engineer specializing in API integration and error handling.",
        (f"Here's my solution:\n\n{prev}\n\n"
         "Optimize this for robustness and efficiency. Make ONE pass of improvements — "
         "no major rewrites. Focus on: error handling completeness, response parsing "
         "correctness, edge cases. Return optimized code with a brief explanation."),
    )


# ── API Callers ───────────────────────────────────────────────────────────────

def estimate_cost(model, in_tok, out_tok, table):
    p = table.get(model)
    if not p: return None
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000

def _make_result(text, model, in_tok, out_tok, total_tok, elapsed):
    out_tok = out_tok or 0
    return {
        "text":           text,
        "model":          model,
        "input_tokens":   in_tok,
        "output_tokens":  out_tok,
        "total_tokens":   total_tok,
        "elapsed_s":      round(elapsed, 3),
        "tokens_per_sec": round(out_tok / elapsed, 1) if elapsed > 0 else 0,
        "cost":           None,
    }

def call_openai(system, user):
    if openai is None: raise RuntimeError("openai not installed")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    t0   = time.time()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE, max_completion_tokens=MAX_TOKENS,
    )
    elapsed = time.time() - t0
    u = resp.usage
    r = _make_result(resp.choices[0].message.content, resp.model,
                     u.prompt_tokens, u.completion_tokens, u.total_tokens, elapsed)
    r["cost"] = estimate_cost(OPENAI_MODEL, u.prompt_tokens, u.completion_tokens, OPENAI_PRICING)
    return r

def call_anthropic(system, user):
    if anthropic is None: raise RuntimeError("anthropic not installed")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    t0   = time.time()
    resp = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        system=system, messages=[{"role": "user", "content": user}],
    )
    elapsed  = time.time() - t0
    u        = resp.usage
    text     = "".join(b.text for b in resp.content if hasattr(b, "text"))
    out_tok  = u.output_tokens
    r = _make_result(text, resp.model, u.input_tokens, out_tok,
                     u.input_tokens + out_tok, elapsed)
    r["cost"] = estimate_cost(ANTHROPIC_MODEL, u.input_tokens, out_tok, ANTHROPIC_PRICING)
    return r

def call_model(provider, system, user):
    r = call_openai(system, user) if provider == "openai" else call_anthropic(system, user)
    time.sleep(CALL_DELAY_S)
    return r

def merge_two_step(r1, r2):
    tot_e   = r1["elapsed_s"] + r2["elapsed_s"]
    tot_out = r1["output_tokens"] + r2["output_tokens"]
    return {
        "text":           r2["text"],
        "model":          r1["model"],
        "input_tokens":   r1["input_tokens"] + r2["input_tokens"],
        "output_tokens":  tot_out,
        "total_tokens":   r1["total_tokens"] + r2["total_tokens"],
        "elapsed_s":      round(tot_e, 3),
        "tokens_per_sec": round(tot_out / tot_e, 1) if tot_e > 0 else 0,
        "cost":           (r1.get("cost") or 0) + (r2.get("cost") or 0),
    }


# ── Trial Runner ──────────────────────────────────────────────────────────────

STRATEGIES   = ["basic", "efficiency", "two_step"]
PROVIDERS    = ["openai", "anthropic"]
STRAT_LABELS = {"basic": "Basic", "efficiency": "Efficiency", "two_step": "Two-Step"}
PROV_LABELS  = {
    "openai":    f"OpenAI ({OPENAI_MODEL})",
    "anthropic": f"Anthropic ({ANTHROPIC_MODEL})",
}

def run_trial(strategy, problem, prob_key, provider):
    if strategy == "basic":
        sp, up = prompt_basic(problem)
        r = call_model(provider, sp, up)
    elif strategy == "efficiency":
        sp, up = prompt_efficiency(problem)
        r = call_model(provider, sp, up)
    elif strategy == "two_step":
        sp1, up1 = prompt_two_step_generate(problem)
        r1 = call_model(provider, sp1, up1)
        sp2, up2 = prompt_two_step_optimize(r1["text"])
        r2 = call_model(provider, sp2, up2)
        r  = merge_two_step(r1, r2)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    code = extract_code(r["text"])
    passed, total = run_unit_tests(code, prob_key)
    r["tests_passed"] = passed
    r["tests_total"]  = total
    return r


# ── Report Section Builder ────────────────────────────────────────────────────

def _avg(vals): return statistics.mean(vals) if vals else 0
def _ok(runs):  return [r for r in runs if "error" not in r]

def build_cat3_section(all_results, start_time):
    W = 102
    lines = []

    def hr(c="═"): return c * W
    def sec(t):    lines.extend([f"\n{hr('─')}", f"  {t}", hr("─")])

    lines += [
        "",
        hr(),
        "  CATEGORY 3 — API Integration Tasks",
        f"  Appended  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Wall time : {round(time.time() - start_time, 1)}s  "
        f"(Cat 3 only — Tasks 5 & 6)",
        hr(),
        "",
        f"  Models     : OpenAI {OPENAI_MODEL}  ×  Anthropic {ANTHROPIC_MODEL}",
        f"  Strategies : Basic  |  Efficiency-Focused  |  Two-Step (generate + optimize)",
        f"  Trials/cell: {N_RUNS}   Temperature: {TEMPERATURE}   Max tokens: {MAX_TOKENS}",
        f"  Unit tests : mock requests module (no real OWM key used)",
    ]

    for prob_key, prob in PROBLEMS.items():
        sec(f"PROBLEM: {prob['name']}  [{prob['difficulty']}]  [{prob['category']}]"
            f"  —  {prob['optimal']}")
        lines.append(f"  Baseline execution: N/A\n")

        for provider in PROVIDERS:
            lines.append(f"  ── {PROV_LABELS[provider]}")
            lines.append(
                f"  {'Strategy':<18} {'Runs':>5}  {'Tests':>8}  "
                f"{'Avg In':>8}  {'Avg Out':>8}  {'Avg Tot':>9}  "
                f"{'Avg Lat(s)':>10}  {'Avg Tok/s':>10}  {'Avg Cost':>12}"
            )
            lines.append("  " + "─"*18 + " " + "─"*5 + "  " + "─"*8 + "  " +
                         "─"*8 + "  " + "─"*8 + "  " + "─"*9 + "  " +
                         "─"*10 + "  " + "─"*10 + "  " + "─"*12)
            for strat in STRATEGIES:
                runs = _ok(all_results.get((prob_key, provider, strat), []))
                if not runs:
                    lines.append(f"  {STRAT_LABELS[strat]:<18}  (no successful runs)")
                    continue
                tp = sum(r["tests_passed"] for r in runs)
                tt = sum(r["tests_total"]  for r in runs)
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
            lines.append(f"  {PROV_LABELS[provider]}")
            for strat in STRATEGIES:
                runs = all_results.get((prob_key, provider, strat), [])
                lines.append(f"\n    [{STRAT_LABELS[strat]}]  ({len(runs)} runs, {N_RUNS} planned)")
                lines.append(
                    f"    {'Run':<4}  {'In Tok':>7}  {'Out Tok':>7}  {'Total':>7}  "
                    f"{'Lat(s)':>7}  {'Tok/s':>7}  {'Cost':>12}  Tests"
                )
                lines.append("    " + "─"*4 + "  " + "─"*7 + "  " + "─"*7 + "  " +
                              "─"*7 + "  " + "─"*7 + "  " + "─"*7 + "  " + "─"*12 + "  " + "─"*5)
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

    # ── Cross-strategy comparison (Cat 3 only) ──────────────────────────────
    sec("CAT 3 — CROSS-STRATEGY COMPARISON (Both API Tasks Combined)")
    for provider in PROVIDERS:
        lines.append(f"\n  {PROV_LABELS[provider]}")
        lines.append(
            f"  {'Strategy':<18}  {'Avg In':>8}  {'Avg Out':>8}  {'Avg Total':>10}  "
            f"{'Avg Lat':>8}  {'Avg Tok/s':>10}  {'Avg Cost':>12}  Tests"
        )
        lines.append("  " + "─"*18 + "  " + "─"*8 + "  " + "─"*8 + "  " + "─"*10 +
                     "  " + "─"*8 + "  " + "─"*10 + "  " + "─"*12 + "  " + "─"*8)
        for strat in STRATEGIES:
            runs = [r for pk in PROBLEMS for r in _ok(all_results.get((pk, provider, strat), []))]
            if not runs: continue
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

    # ── Model comparison (Cat 3 only) ──────────────────────────────────────
    sec("CAT 3 — MODEL COMPARISON (OpenAI vs Anthropic, Both Tasks Combined)")
    all_oai = [r for pk in PROBLEMS for st in STRATEGIES
               for r in _ok(all_results.get((pk, "openai", st), []))]
    all_ant = [r for pk in PROBLEMS for st in STRATEGIES
               for r in _ok(all_results.get((pk, "anthropic", st), []))]

    def cmp(label, ov, av, lower=True, fmt=".3f"):
        o, a = _avg(ov), _avg(av)
        w = ("tie" if abs(o - a) < 1e-9
             else ("OpenAI" if o < a else "Anthropic") if lower
             else ("OpenAI" if o > a else "Anthropic"))
        return f"  {label:<26}  {format(o, fmt):<18}  {format(a, fmt):<18}  {w}"

    lines.append(f"\n  {'Metric':<26}  {'OpenAI':<18}  {'Anthropic':<18}  Winner")
    lines.append("  " + "─"*26 + "  " + "─"*18 + "  " + "─"*18 + "  " + "─"*12)
    lines.append(cmp("Avg input tokens",
                     [r["input_tokens"]   for r in all_oai], [r["input_tokens"]   for r in all_ant], fmt=".0f"))
    lines.append(cmp("Avg output tokens",
                     [r["output_tokens"]  for r in all_oai], [r["output_tokens"]  for r in all_ant], lower=False, fmt=".0f"))
    lines.append(cmp("Avg total tokens",
                     [r["total_tokens"]   for r in all_oai], [r["total_tokens"]   for r in all_ant], fmt=".0f"))
    lines.append(cmp("Avg latency (s)",
                     [r["elapsed_s"]      for r in all_oai], [r["elapsed_s"]      for r in all_ant]))
    lines.append(cmp("Avg throughput (tok/s)",
                     [r["tokens_per_sec"] for r in all_oai], [r["tokens_per_sec"] for r in all_ant], lower=False, fmt=".1f"))
    lines.append(cmp("Avg cost/run ($)",
                     [r["cost"] or 0      for r in all_oai], [r["cost"] or 0      for r in all_ant], fmt=".6f"))

    oai_tp = sum(r["tests_passed"] for r in all_oai)
    oai_tt = sum(r["tests_total"]  for r in all_oai)
    ant_tp = sum(r["tests_passed"] for r in all_ant)
    ant_tt = sum(r["tests_total"]  for r in all_ant)
    oai_rate = oai_tp / oai_tt if oai_tt else 0
    ant_rate = ant_tp / ant_tt if ant_tt else 0
    tw = ("OpenAI" if oai_rate > ant_rate else
          "Anthropic" if ant_rate > oai_rate else "tie")
    lines.append(
        f"  {'Test pass rate':<26}  {oai_tp}/{oai_tt} ({oai_rate:.0%})         "
        f"  {ant_tp}/{ant_tt} ({ant_rate:.0%})         "
        f"  {tw}"
    )

    lines.append(f"\n{hr()}\n")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    total_runs = len(PROBLEMS) * len(PROVIDERS) * len(STRATEGIES) * N_RUNS
    total_api  = total_runs + len(PROBLEMS) * len(PROVIDERS) * N_RUNS

    print(f"\n{'═'*70}")
    print(f"  Category 3 — API Integration  ({len(PROBLEMS)} problems, "
          f"{len(PROVIDERS)} models, {len(STRATEGIES)} strategies, {N_RUNS} runs each)")
    print(f"  Logical runs: {total_runs}   Est. API calls: {total_api}")
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
                           f"{problem['name']:<34} | {provider:<10} | "
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
                            "error": str(e), "input_tokens": 0, "output_tokens": 0,
                            "total_tokens": 0, "elapsed_s": 0, "tokens_per_sec": 0,
                            "cost": 0, "tests_passed": 0, "tests_total": 3,
                        }
                        all_results[key].append(err_r)
                        print(f"  {tag}  ERROR: {e}")

    section = build_cat3_section(all_results, start_time)
    print(section)

    # Append to existing report
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"  Appended to: {REPORT_PATH}")

    # Save raw JSON for Cat 3
    json_path = os.path.join(HERE, "research_results_cat3.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {f"{pk}__{pv}__{st}": [{k: v for k, v in r.items() if k != "text"} for r in runs]
             for (pk, pv, st), runs in all_results.items()},
            f, indent=2
        )
    print(f"  Raw JSON   : {json_path}\n")


if __name__ == "__main__":
    main()
