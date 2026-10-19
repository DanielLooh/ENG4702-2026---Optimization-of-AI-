"""Weather tasks, prompts, deterministic request mocks, and correctness checks."""

import json
import re
import sys
import types
import urllib.parse
from typing import Optional


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


class _RequestException(Exception):
    pass


class _HTTPError(_RequestException):
    def __init__(self, msg=""):
        super().__init__(msg)
        self.response = None


class _ConnectionError(_RequestException):
    pass


class _Timeout(_RequestException):
    pass


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


def extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return None
    for b in blocks:
        if re.search(r"^\s*(def |class )", b, re.MULTILINE):
            return b.strip()
    return blocks[0].strip()


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
