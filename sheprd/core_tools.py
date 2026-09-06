"""
Core Built-in Tools for Sheprd Agents.
Provides zero-setup, zero-dependency native tools in pure Python:
- get_weather: Real-time weather via wttr.in (JSON format)
- wikipedia_search: Factual lookup & summaries via Wikipedia REST API
- fetch_url: Fetch and extract clean article text from any URL
- calculate: Safe mathematical and date expressions
- get_current_time: System time, date, and timezone
"""

import ast
import ipaddress
import json
import logging
import math
import operator
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("sheprd.tools.core")


# Safe expression evaluator for `calculate`
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if op_type is ast.Pow:
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Exponents and bases must be numeric.")
            if abs(right) > 100 or abs(left) > 10000:
                raise ValueError("Exponent or base is too large to safely compute (exponent bomb guard).")
            if abs(right) > 20 and abs(left) > 10:
                raise ValueError("Calculation magnitude exceeds memory limits (exponent bomb guard).")
        return _ALLOWED_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unary operator {op_type.__name__} is not allowed")
        operand = _safe_eval_node(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Name):
        if node.id in _ALLOWED_FUNCTIONS:
            return _ALLOWED_FUNCTIONS[node.id]
        raise ValueError(f"Variable '{node.id}' is not defined")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Function call '{getattr(node.func, 'id', 'unknown')}' is not allowed")
        fn = _ALLOWED_FUNCTIONS[node.func.id]
        if not callable(fn):
            return fn
        args = [_safe_eval_node(a) for a in node.args]
        return fn(*args)
    else:
        raise ValueError(f"Expression type '{type(node).__name__}' is not supported")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self._ignore = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self._ignore = True
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "li", "br", "tr"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg"):
            self._ignore = False

    def handle_data(self, data):
        if not self._ignore:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self.text_parts)
        # Collapse multiple blank lines
        return re.sub(r"\n\s*\n+", "\n\n", raw).strip()


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def tool_get_weather(location: str) -> str:
    """Fetches real-time weather data for a given city or location."""
    clean_loc = location.strip()
    if not clean_loc:
        return "Error: Location cannot be empty."
    encoded = urllib.parse.quote(clean_loc)
    url = f"https://wttr.in/{encoded}?format=j1"
    req = urllib.request.Request(url, headers={"User-Agent": "Sheprd-Agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            curr = data.get("current_condition", [{}])[0]
            nearest = data.get("nearest_area", [{}])[0]
            area_name = nearest.get("areaName", [{}])[0].get("value", clean_loc)
            country = nearest.get("country", [{}])[0].get("value", "")

            temp_c = curr.get("temp_C", "N/A")
            temp_f = curr.get("temp_F", "N/A")
            feels_c = curr.get("FeelsLikeC", "N/A")
            condition = curr.get("weatherDesc", [{}])[0].get("value", "Unknown")
            humidity = curr.get("humidity", "N/A")
            wind_kmh = curr.get("windspeedKmph", "N/A")
            wind_dir = curr.get("winddir16Point", "")

            return json.dumps({
                "location": f"{area_name}, {country}".strip(", "),
                "condition": condition,
                "temperature": f"{temp_c}°C ({temp_f}°F)",
                "feels_like": f"{feels_c}°C",
                "humidity": f"{humidity}%",
                "wind": f"{wind_kmh} km/h {wind_dir}".strip(),
            }, indent=2)
    except Exception as e:
        logger.warning("Weather fetch error for '%s': %s", clean_loc, e)
        return f"Unable to retrieve weather for '{clean_loc}': {e}"


def tool_wikipedia_search(query: str) -> str:
    """Searches Wikipedia and retrieves factual summaries."""
    clean_query = query.strip()
    if not clean_query:
        return "Error: Query cannot be empty."

    # 1. Search for matching titles
    search_url = (
        f"https://en.wikipedia.org/w/api.php?action=opensearch"
        f"&search={urllib.parse.quote(clean_query)}&limit=3&namespace=0&format=json"
    )
    req = urllib.request.Request(search_url, headers={"User-Agent": "Sheprd-Agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            titles = data[1] if len(data) > 1 else []
            if not titles:
                return f"No Wikipedia articles found for '{clean_query}'."

            # Fetch summary of top match
            top_title = titles[0]
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(top_title)}"
            req_sum = urllib.request.Request(summary_url, headers={"User-Agent": "Sheprd-Agent/1.0"})
            with urllib.request.urlopen(req_sum, timeout=8) as resp_sum:
                sum_data = json.loads(resp_sum.read().decode("utf-8"))
                extract = sum_data.get("extract", "No summary available.")
                url_link = sum_data.get("content_urls", {}).get("desktop", {}).get("page", "")

                return json.dumps({
                    "title": top_title,
                    "summary": extract,
                    "url": url_link,
                    "other_matches": titles[1:],
                }, indent=2)
    except Exception as e:
        logger.warning("Wikipedia error for '%s': %s", clean_query, e)
        return f"Error querying Wikipedia for '{clean_query}': {e}"


def _is_safe_public_url(url: str) -> Tuple[bool, str]:
    """
    Validates that a URL does not resolve to private, loopback, link-local or cloud metadata IP.
    
    Security note: A theoretical time-of-check to time-of-use (TOCTOU) DNS rebinding window
    exists between this pre-flight socket.getaddrinfo check and the subsequent urllib request,
    if an attacker controls an authoritative nameserver that returns a public IP initially and
    a private/loopback IP with 0s TTL immediately thereafter. In zero-trust environments,
    an egress firewall or forward proxy should provide strict perimeter defense.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, "Only http and https protocols are supported."
        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL: hostname missing."
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        addr_infos = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, socktype, proto, canonname, sockaddr in addr_infos:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_loopback:
                return False, f"Access to loopback address {ip} is prohibited (SSRF guard)."
            if ip.is_private:
                return False, f"Access to private subnet address {ip} is prohibited (SSRF guard)."
            if ip.is_link_local:
                return False, f"Access to link-local / cloud metadata address {ip} is prohibited (SSRF guard)."
            if ip.is_reserved or ip.is_multicast:
                return False, f"Access to reserved address {ip} is prohibited (SSRF guard)."
        return True, ""
    except Exception as e:
        return False, f"Failed to resolve URL destination: {e}"


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Enforces SSRF boundary checks on HTTP redirect targets."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        is_safe, reason = _is_safe_public_url(newurl)
        if not is_safe:
            raise urllib.error.HTTPError(newurl, 403, f"SSRF redirection blocked: {reason}", headers, None)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def tool_fetch_url(url: str) -> str:
    """Fetches text content from a public web URL with strict SSRF defense."""
    clean_url = url.strip()
    is_safe, reason = _is_safe_public_url(clean_url)
    if not is_safe:
        return f"Error: URL blocked by security policy: {reason}"

    opener = urllib.request.build_opener(_SafeRedirectHandler())
    req = urllib.request.Request(clean_url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    try:
        with opener.open(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            # Cap read to 64KB to prevent multi-GB memory exhaustion bomb (N2)
            raw_bytes = resp.read(65536)
            raw = raw_bytes.decode("utf-8", errors="ignore")

            if "text/html" in content_type or "<html" in raw.lower():
                parser = _HTMLTextExtractor()
                parser.feed(raw)
                text = parser.get_text()
            else:
                text = raw

            max_len = 3000
            if len(text) > max_len:
                text = text[:max_len] + f"\n... [Truncated: {len(text) - max_len} remaining characters]"

            return text or "Empty page content."
    except Exception as e:
        logger.warning("URL fetch failed for '%s': %s", clean_url, e)
        return f"Failed to fetch '{clean_url}': {e}"


def tool_calculate(expression: str) -> str:
    """Safely evaluates a mathematical expression."""
    clean_expr = expression.strip()
    try:
        tree = ast.parse(clean_expr, mode="eval")
        result = _safe_eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def tool_get_current_time(timezone_name: str = "UTC") -> str:
    """Returns the current date, time, and UTC timestamp in requested timezone."""
    clean_tz = (timezone_name or "UTC").strip()
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(clean_tz)
    except Exception:
        tz = timezone.utc

    now_tz = datetime.now(tz)
    now_utc = datetime.now(timezone.utc)
    return json.dumps({
        "time": now_tz.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "timezone": str(tz),
        "utc_time": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "timestamp": int(now_utc.timestamp()),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool Schemas for llama.cpp / OpenAI function-calling
# ---------------------------------------------------------------------------

CORE_TOOLS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_weather": {
        "definition": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get real-time weather information (temperature, conditions, humidity, wind) for any city or location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or region name, e.g. 'Petawawa', 'Tokyo', 'San Francisco, CA'."
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        "handler": lambda args: tool_get_weather(args.get("location", ""))
    },
    "wikipedia_search": {
        "definition": {
            "type": "function",
            "function": {
                "name": "wikipedia_search",
                "description": "Search Wikipedia for factual background, biographies, places, historical events, scientific concepts, and technology.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search topic or term."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        "handler": lambda args: tool_wikipedia_search(args.get("query", ""))
    },
    "fetch_url": {
        "definition": {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch text content and documentation from a public web URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full HTTP or HTTPS URL to read."
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        "handler": lambda args: tool_fetch_url(args.get("url", ""))
    },
    "calculate": {
        "definition": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate mathematical expressions (arithmetic, trigonometry, exponents, square roots, logarithms).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The math expression to evaluate, e.g. '2 ** 16', 'sqrt(144) + 42', 'round(sin(pi/4), 4)'."
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        "handler": lambda args: tool_calculate(args.get("expression", ""))
    },
    "get_current_time": {
        "definition": {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get current live system date, time, and UTC timestamp.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone_name": {
                            "type": "string",
                            "description": "Optional timezone name (defaults to UTC)."
                        }
                    }
                }
            }
        },
        "handler": lambda args: tool_get_current_time(args.get("timezone_name", "UTC"))
    },
}


def get_core_tool_definitions() -> List[Dict[str, Any]]:
    """Returns OpenAI-compatible function definitions for all core tools."""
    return [t["definition"] for t in CORE_TOOLS_REGISTRY.values()]


def execute_core_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Executes a core tool by name with provided arguments dictionary."""
    if name not in CORE_TOOLS_REGISTRY:
        return f"Unknown tool '{name}'."
    handler = CORE_TOOLS_REGISTRY[name]["handler"]
    try:
        return handler(arguments)
    except Exception as e:
        logger.error("Core tool '%s' execution failed: %s", name, e)
        return f"Error executing tool '{name}': {e}"


# Convenience aliases
calculate = tool_calculate
get_current_time = tool_get_current_time
get_weather = tool_get_weather
wikipedia_search = tool_wikipedia_search
fetch_url = tool_fetch_url
