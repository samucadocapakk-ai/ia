import asyncio
import ast
import ipaddress
import json
import math
import os
import re
import socket
import time
import secrets
from datetime import datetime, timezone
from contextlib import aclosing
from html import unescape
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request as UrlRequest, urlopen
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
from provider_load import install_load_meter
from browser_bridge import install, browser_call, BROWSER_TOOLS, BROWSER_PROMPT, activity_sources


app = FastAPI()
install(app)
install_load_meter(app)

client = AsyncOpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.environ.get("FEATHERLESS_API_KEY"),
)

# Required Space secret. No hardcoded fallback model.
MODEL_ID = os.environ["MODEL_ID"]

# Blank or missing means: do not inject an app-level system message.
SYSTEM_MESSAGE = os.environ.get("SYSTEM_MESSAGE", "").strip()


def current_time_context(now=None):
    """Current UTC calendar date; no time of day is sent to the model."""
    utc_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "role": "system",
        "content": (
            f"Current date (UTC): {utc_now:%Y-%m-%d}.\n"
            "Use this server-provided date for relative dates such as today, yesterday, "
            "and tomorrow, unless the user specifies another timezone or date context. "
            "This date does not mean your knowledge is current. Use web tools to verify "
            "time-sensitive information."
        ),
    }

# Keep the existing concurrency limit behavior.
FEATHERLESS_SEMAPHORE = asyncio.Semaphore(
    int(os.environ.get("MAX_CONCURRENT_REQUESTS", "3"))
)

# Provider/context settings. Defaults are safe for a 32K provider window.
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "32768"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "4096"))
CONTEXT_SAFETY_TOKENS = int(os.environ.get("CONTEXT_SAFETY_TOKENS", "1024"))
MAX_INPUT_TOKENS = max(
    1024,
    MAX_CONTEXT_TOKENS - MAX_OUTPUT_TOKENS - CONTEXT_SAFETY_TOKENS,
)

# Generation settings requested for both chat and agent model calls.
TEMPERATURE = 1.4
TOP_K = 70

DDG_TIMEOUT_SECONDS = float(os.environ.get("DDG_TIMEOUT_SECONDS", "3.5"))
DDG_MAX_CONTEXT_CHARS = int(os.environ.get("DDG_MAX_CONTEXT_CHARS", "3500"))
AGENT_MAX_ROUNDS = int(os.environ.get("AGENT_MAX_ROUNDS", "6"))
AGENT_RESULT_LIMIT = int(os.environ.get("AGENT_RESULT_LIMIT", "18000"))


class ThoughtFilter:
    """Buffer streamed chunks and hide content inside reasoning tags."""

    def __init__(self) -> None:
        self.buffer = ""
        self.in_thought = False
        self.current_end_marker: str | None = None
        self.first_token_received = False
        self.markers = [
            ("<think>", "</think>"),
            ("<|channel>thought", "<channel|>"),
            ("<|channel>", "<channel|>"),
        ]

    def process(self, chunk_text: str) -> str:
        self.buffer += chunk_text
        output = ""

        while self.buffer:
            if not self.in_thought:
                earliest_start_idx = -1
                matched_start = None
                matched_end = None

                for start, end in self.markers:
                    idx = self.buffer.find(start)
                    if idx != -1 and (
                        earliest_start_idx == -1 or idx < earliest_start_idx
                    ):
                        earliest_start_idx = idx
                        matched_start = start
                        matched_end = end

                if earliest_start_idx != -1:
                    output += self.buffer[:earliest_start_idx]
                    self.buffer = self.buffer[
                        earliest_start_idx + len(matched_start) :
                    ]
                    self.in_thought = True
                    self.current_end_marker = matched_end
                else:
                    max_possible_partial = 0
                    for start, _ in self.markers:
                        for i in range(1, len(start)):
                            if self.buffer.endswith(start[:i]):
                                max_possible_partial = max(max_possible_partial, i)
                                break

                    if max_possible_partial > 0:
                        output += self.buffer[:-max_possible_partial]
                        self.buffer = self.buffer[-max_possible_partial:]
                        break

                    output += self.buffer
                    self.buffer = ""
            else:
                if not self.current_end_marker:
                    self.in_thought = False
                    continue

                idx = self.buffer.find(self.current_end_marker)
                if idx != -1:
                    self.buffer = self.buffer[idx + len(self.current_end_marker) :]
                    self.in_thought = False
                    self.current_end_marker = None
                else:
                    max_possible_partial = 0
                    for i in range(1, len(self.current_end_marker)):
                        if self.buffer.endswith(self.current_end_marker[:i]):
                            max_possible_partial = max(max_possible_partial, i)
                            break

                    self.buffer = (
                        self.buffer[-max_possible_partial:]
                        if max_possible_partial > 0
                        else ""
                    )
                    break

        if output and not self.first_token_received:
            cleaned_output = output.lstrip()
            if cleaned_output:
                self.first_token_received = True
                return cleaned_output
            return ""

        return output

    def flush(self) -> str:
        if not self.in_thought and self.buffer:
            result = self.buffer
            self.buffer = ""
            if result and not self.first_token_received:
                cleaned = result.lstrip()
                if cleaned:
                    self.first_token_received = True
                    return cleaned
                return ""
            return result
        return ""



class ToolStreamFilter:
    """
    Stream normal text while swallowing <tool_call>...</tool_call> anywhere in
    the response, including when the tags are split across provider chunks.
    """

    START = "<tool_call>"
    END = "</tool_call>"

    def __init__(self):
        self.pending = ""
        self.in_tool = False
        self.saw_tool = False
        self.closed_tool = False
        self.tool_payload = ""

    @staticmethod
    def _partial_suffix(value: str, marker: str) -> int:
        low_value = value.lower()
        low_marker = marker.lower()
        max_len = min(len(value), len(marker) - 1)

        for size in range(max_len, 0, -1):
            if low_value.endswith(low_marker[:size]):
                return size

        return 0

    def process(self, text: str) -> str:
        if not text or self.closed_tool:
            return ""

        self.pending += text
        output = ""

        while self.pending and not self.closed_tool:
            if not self.in_tool:
                low = self.pending.lower()
                idx = low.find(self.START)

                if idx != -1:
                    output += self.pending[:idx]
                    self.pending = self.pending[idx + len(self.START):]
                    self.in_tool = True
                    self.saw_tool = True
                    continue

                keep = self._partial_suffix(self.pending, self.START)

                if keep:
                    output += self.pending[:-keep]
                    self.pending = self.pending[-keep:]
                else:
                    output += self.pending
                    self.pending = ""

                break

            low = self.pending.lower()
            idx = low.find(self.END)

            if idx != -1:
                self.tool_payload += self.pending[:idx]
                self.pending = ""
                self.in_tool = False
                self.closed_tool = True
                break

            keep = self._partial_suffix(self.pending, self.END)

            if keep:
                self.tool_payload += self.pending[:-keep]
                self.pending = self.pending[-keep:]
            else:
                self.tool_payload += self.pending
                self.pending = ""

            break

        return output

    def flush_visible(self) -> str:
        if self.saw_tool:
            # Never show incomplete or malformed tool markup.
            self.pending = ""
            return ""

        result = self.pending
        self.pending = ""
        return result

    def get_call(self):
        if not self.saw_tool:
            return None
        return _parse_tool_payload(self.tool_payload)


def _content_text(content) -> str:
    """Extract text from normal or multimodal OpenAI message content."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)

    return ""


def _latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return _content_text(message.get("content")).strip()
    return ""


# ---------------------------------------------------------------------------
# Context-window management
# ---------------------------------------------------------------------------


def _estimate_text_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate that works across changing models."""
    if not text:
        return 0
    # ~3.3 chars/token is intentionally conservative for mixed code/text.
    return max(1, math.ceil(len(text) / 3.3))


def _estimate_content_tokens(content) -> int:
    if isinstance(content, str):
        return _estimate_text_tokens(content)

    if isinstance(content, list):
        total = 0
        for item in content:
            if not isinstance(item, dict):
                total += 8
                continue

            item_type = item.get("type")
            if item_type == "text":
                total += _estimate_text_tokens(str(item.get("text") or ""))
            elif item_type == "image_url":
                # Do not count base64 characters as language tokens.
                total += 1200
            else:
                total += _estimate_text_tokens(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                )
        return total

    return _estimate_text_tokens(str(content or ""))


def _estimate_message_tokens(message: dict) -> int:
    if not isinstance(message, dict):
        return 8

    role = str(message.get("role") or "")
    return 12 + _estimate_text_tokens(role) + _estimate_content_tokens(
        message.get("content")
    )


def _estimate_messages_tokens(messages: list) -> int:
    return 32 + sum(_estimate_message_tokens(m) for m in messages)


def _truncate_text(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if _estimate_text_tokens(text) <= max_tokens:
        return text

    max_chars = max(1, int(max_tokens * 3.0))
    if len(text) <= max_chars:
        return text

    marker = "\n\n[Earlier content truncated to fit the provider context window.]\n\n"
    marker_chars = len(marker)

    if max_chars <= marker_chars + 64:
        return text[-max_chars:]

    return marker + text[-(max_chars - marker_chars) :]


def _truncate_message(message: dict, max_tokens: int) -> dict:
    msg = dict(message)
    content = msg.get("content")

    if isinstance(content, str):
        msg["content"] = _truncate_text(content, max(1, max_tokens - 16))
        return msg

    if isinstance(content, list):
        copied = []
        image_cost = 0
        text_positions = []

        for item in content:
            if not isinstance(item, dict):
                copied.append(item)
                continue

            new_item = dict(item)
            copied.append(new_item)

            if new_item.get("type") == "image_url":
                image_cost += 1200
            elif new_item.get("type") == "text":
                text_positions.append(len(copied) - 1)

        text_budget = max(64, max_tokens - image_cost - 32)
        if text_positions:
            per_text = max(32, text_budget // len(text_positions))
            for pos in text_positions:
                copied[pos]["text"] = _truncate_text(
                    str(copied[pos].get("text") or ""), per_text
                )

        msg["content"] = copied
        return msg

    return msg


def fit_context(messages: list, budget: int = MAX_INPUT_TOKENS) -> list:
    """
    Keep the app/system preamble and newest user request, then retain as much
    recent history as will fit. Oldest history is discarded first.
    """
    source = [dict(m) for m in messages if isinstance(m, dict)]
    if not source:
        return []

    if _estimate_messages_tokens(source) <= budget:
        return source

    # Preserve only the contiguous leading system preamble as fixed context.
    leading_indices = []
    for i, msg in enumerate(source):
        if msg.get("role") == "system":
            leading_indices.append(i)
        else:
            break

    latest_user_idx = None
    for i in range(len(source) - 1, -1, -1):
        if source[i].get("role") == "user":
            latest_user_idx = i
            break

    fixed = set(leading_indices)
    if latest_user_idx is not None:
        fixed.add(latest_user_idx)

    fixed_messages = [source[i] for i in sorted(fixed)]
    fixed_cost = _estimate_messages_tokens(fixed_messages)

    if fixed_cost > budget:
        min_user_budget = 768 if latest_user_idx is not None else 0

        if latest_user_idx is not None:
            system_cost = _estimate_messages_tokens(
                [source[i] for i in leading_indices]
            )
            user_budget = max(128, budget - system_cost - 64)
            source[latest_user_idx] = _truncate_message(
                source[latest_user_idx], user_budget
            )

        fixed_messages = [source[i] for i in sorted(fixed)]
        if _estimate_messages_tokens(fixed_messages) > budget and leading_indices:
            available_for_system = max(
                256,
                budget
                - min_user_budget
                - (
                    _estimate_message_tokens(source[latest_user_idx])
                    if latest_user_idx is not None
                    else 0
                ),
            )
            per_system = max(128, available_for_system // len(leading_indices))
            for idx in leading_indices:
                source[idx] = _truncate_message(source[idx], per_system)

    selected = set(fixed)

    for i in range(len(source) - 1, -1, -1):
        if i in selected:
            continue

        candidate_indices = sorted(selected | {i})
        candidate = [source[j] for j in candidate_indices]
        if _estimate_messages_tokens(candidate) <= budget:
            selected.add(i)

    result = [source[i] for i in sorted(selected)]

    while len(result) > 1 and _estimate_messages_tokens(result) > budget:
        removed = False
        for i, msg in enumerate(result):
            if msg.get("role") == "system" and i == 0:
                continue
            if latest_user_idx is not None and msg.get("role") == "user":
                if _content_text(msg.get("content")) == _latest_user_text(result):
                    continue
            result.pop(i)
            removed = True
            break
        if not removed:
            break

    return result


# ---------------------------------------------------------------------------
# Automatic DuckDuckGo context
# ---------------------------------------------------------------------------


def _should_try_ddg(text: str) -> bool:
    """Conservative heuristic: only send likely factual/current questions to DDG."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text or len(text) > 1200:
        return False

    lower = text.lower()
    non_search_starts = (
        "write ",
        "rewrite ",
        "draft ",
        "compose ",
        "translate ",
        "summarize ",
        "proofread ",
        "roleplay ",
        "role-play ",
        "create a story",
        "make a story",
        "debug ",
        "fix this code",
        "refactor ",
        "generate code",
        "write code",
    )
    if lower.startswith(non_search_starts):
        return False

    freshness_terms = (
        "latest",
        "current",
        "currently",
        "today",
        "tonight",
        "right now",
        "recent",
        "recently",
        "news",
        "price",
        "score",
        "status",
        "weather",
        "forecast",
        "flight status",
        "who is the",
        "what is the current",
        "as of",
        "this week",
        "this month",
        "this year",
    )
    if any(term in lower for term in freshness_terms):
        return True

    factual_question = re.match(
        r"^(who|what|when|where|which|how old|how many|how much|define|meaning of|capital of|population of)\b",
        lower,
    )
    return bool(factual_question and ("?" in text or len(text) <= 220))


def _flatten_related_topics(items, limit: int = 4) -> list[dict]:
    out: list[dict] = []

    def walk(values):
        for item in values or []:
            if len(out) >= limit:
                return
            if not isinstance(item, dict):
                continue

            nested = item.get("Topics")
            if isinstance(nested, list):
                walk(nested)
                continue

            text = str(item.get("Text") or "").strip()
            url = str(item.get("FirstURL") or "").strip()
            if text:
                out.append({"text": text, "url": url})

    walk(items)
    return out


def _ddg_request_sync(query: str) -> dict | None:
    params = urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
            "skip_disambig": "1",
            "t": "xortron_chat",
        }
    )

    request = UrlRequest(
        f"https://api.duckduckgo.com/?{params}",
        headers={
            "Accept": "application/json",
            "User-Agent": "XORTRON-HF-Space/1.0",
        },
    )

    with urlopen(request, timeout=DDG_TIMEOUT_SECONDS) as response:
        raw = response.read(1_000_000)

    data = json.loads(raw.decode("utf-8", errors="replace"))
    return data if isinstance(data, dict) else None


async def duckduckgo_context(query: str) -> str | None:
    if not _should_try_ddg(query):
        return None

    try:
        data = await asyncio.to_thread(_ddg_request_sync, query[:500])
    except Exception:
        return None

    if not data:
        return None

    answer = str(data.get("Answer") or "").strip()
    abstract = str(data.get("AbstractText") or data.get("Abstract") or "").strip()
    definition = str(data.get("Definition") or "").strip()
    heading = str(data.get("Heading") or "").strip()
    source_name = str(
        data.get("AbstractSource")
        or data.get("DefinitionSource")
        or "DuckDuckGo Instant Answers"
    ).strip()
    source_url = str(
        data.get("AbstractURL") or data.get("DefinitionURL") or ""
    ).strip()
    related = _flatten_related_topics(data.get("RelatedTopics"), limit=4)

    substantive = answer or abstract or definition
    if not substantive and len(related) < 2:
        return None

    lines = [
        "DuckDuckGo Instant Answer context (automatically retrieved because this query appears factual or time-sensitive):",
        f"Query: {query[:500]}",
    ]

    if heading:
        lines.append(f"Topic: {heading}")
    if answer:
        lines.append(f"Instant answer: {answer}")
    if abstract:
        lines.append(f"Summary: {abstract}")
    if definition:
        lines.append(f"Definition: {definition}")
    if source_name:
        lines.append(f"Source: {source_name}")
    if source_url:
        lines.append(f"Source URL: {source_url}")

    if related:
        lines.append("Related Instant Answer items:")
        for item in related:
            suffix = f" — {item['url']}" if item.get("url") else ""
            lines.append(f"- {item['text']}{suffix}")

    lines.append(
        "Use this context only when it is relevant. DuckDuckGo Instant Answers is not a full web-search results API; "
        "do not imply that a broader web search occurred. If relying on this context, mention the source naturally."
    )

    return "\n".join(lines)[:DDG_MAX_CONTEXT_CHARS]


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


TOOL_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.I)

AGENT_PROMPT = """You have server-side tools.

Use a tool whenever it materially improves the answer, especially for:
- current or recently changed facts
- death/alive status
- dates, news, prices, scores, schedules, weather, releases, or other freshness-sensitive facts
- calculations
- current time
- Wikipedia lookups
- reading a URL

To call a tool, output ONLY this exact wrapper:
<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>

Available tools:
calculator {"expression":"sqrt(144)+5"}
get_current_time {"time_zone":"America/New_York"}
web_search {"query":"...","limit":8}
search_wikipedia {"query":"...","limit":6}
read_webpage {"url":"https://...","max_chars":12000}

web_search performs a real multi-source no-key lookup using DuckDuckGo web
results, Wikipedia live article extracts, and GDELT news results.

After receiving a TOOL RESULT, either use another tool if truly needed or
answer the user normally. Never claim a tool ran unless a TOOL RESULT was
provided. Never print tool markup as part of the final answer."""


def _calc(expr):
    expr = str(expr or "").strip()
    if not expr or len(expr) > 300:
        raise ValueError("Invalid expression length.")

    funcs = {
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
        "floor": math.floor,
        "ceil": math.ceil,
        "pow": pow,
    }
    names = {"pi": math.pi, "e": math.e, **funcs}

    tree = ast.parse(expr, mode="eval")
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("Unsupported expression.")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError("Unsupported name.")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in funcs
        ):
            raise ValueError("Unsupported function.")

    return str(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, names))


def _time_tool(args):
    name = str(args.get("time_zone") or "").strip()
    try:
        tz = ZoneInfo(name) if name else timezone.utc
    except Exception:
        raise ValueError("Invalid IANA timezone.")

    d = datetime.now(tz)
    return json.dumps(
        {
            "iso": d.isoformat(),
            "time_zone": name or "UTC",
            "formatted": d.strftime("%A, %B %d, %Y %I:%M:%S %p %Z"),
        }
    )


async def _wiki(query, limit=6):
    """Wikipedia search that includes live article intros/extracts."""
    query = str(query or "").strip()
    if not query:
        raise ValueError("Wikipedia query required.")

    limit = max(1, min(int(limit or 6), 10))

    def run():
        params = urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": limit,
                "prop": "extracts|info",
                "exintro": "1",
                "explaintext": "1",
                "exsentences": "5",
                "inprop": "url",
                "format": "json",
                "utf8": "1",
                "origin": "*",
            }
        )

        req = UrlRequest(
            "https://en.wikipedia.org/w/api.php?" + params,
            headers={"User-Agent": "XORTRON-HF-Space/1.0"},
        )

        with urlopen(req, timeout=8) as response:
            data = json.loads(response.read(2_000_000).decode("utf-8", "replace"))

        pages = list(data.get("query", {}).get("pages", {}).values())
        pages.sort(key=lambda p: p.get("index", 999999))

        out = []
        for item in pages[:limit]:
            title = str(item.get("title") or "")
            out.append(
                {
                    "title": title,
                    "extract": str(item.get("extract") or "").strip(),
                    "url": str(
                        item.get("fullurl")
                        or ("https://en.wikipedia.org/wiki/" + title.replace(" ", "_"))
                    ),
                }
            )

        return json.dumps({"query": query, "results": out}, ensure_ascii=False)

    try:
        return await asyncio.to_thread(run)
    except Exception:
        return ""

def _safe_url(raw):
    url = str(raw or "").strip()
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Public HTTP/HTTPS URL required.")

    if parsed.hostname.lower() == "localhost" or parsed.hostname.lower().endswith(
        ".local"
    ):
        raise ValueError("Private/local URLs blocked.")

    try:
        for info in socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        ):
            if not ipaddress.ip_address(info[4][0]).is_global:
                raise ValueError("Private/local URLs blocked.")
    except socket.gaierror:
        raise ValueError("Could not resolve host.")

    return url


async def _read_url(raw, max_chars=12000):
    url = _safe_url(raw)
    max_chars = max(500, min(int(max_chars or 12000), 30000))

    def run():
        req = UrlRequest(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 XORTRON-Agent/1.0",
                "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.5",
            },
        )

        with urlopen(req, timeout=8) as response:
            final = _safe_url(response.geturl())
            ctype = response.headers.get("content-type", "")
            raw_text = response.read(max_chars * 4).decode("utf-8", "replace")

        if "html" in ctype.lower():
            raw_text = re.sub(
                r"(?is)<(script|style|noscript|svg|canvas|iframe|nav|footer|form).*?>.*?</\1>",
                " ",
                raw_text,
            )
            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_text)
            title = (
                re.sub(
                    r"\s+", " ", re.sub(r"<[^>]+>", "", title_match.group(1))
                ).strip()
                if title_match
                else ""
            )
            text = re.sub(
                r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", raw_text)
            ).strip()
            return json.dumps(
                {"url": final, "title": title, "content": text[:max_chars]},
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "url": final,
                "content_type": ctype,
                "content": raw_text[:max_chars],
            },
            ensure_ascii=False,
        )

    return await asyncio.to_thread(run)


async def _web(query, limit=6):
    query = str(query or "").strip()
    if not query:
        raise ValueError("Search query required.")

    try:
        data = await asyncio.to_thread(_ddg_request_sync, query[:500])
    except Exception as exc:
        return "DuckDuckGo request failed: " + str(exc)

    if not data:
        return "No DuckDuckGo Instant Answer results."

    out = []

    if data.get("Answer"):
        out.append(
            {
                "title": "Answer",
                "text": data["Answer"],
                "url": data.get("AbstractURL", ""),
            }
        )

    if data.get("AbstractText"):
        out.append(
            {
                "title": data.get("Heading") or "Overview",
                "text": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
            }
        )

    if data.get("Definition"):
        out.append(
            {
                "title": "Definition",
                "text": data["Definition"],
                "url": data.get("DefinitionURL", ""),
            }
        )

    out.extend(
        _flatten_related_topics(
            data.get("RelatedTopics"), limit=max(1, min(int(limit or 6), 10))
        )
    )

    limit = max(1, min(int(limit or 6), 10))
    return json.dumps({"query": query, "results": out[:limit]}, ensure_ascii=False)


async def _ddg_html_search(query: str, limit: int = 8) -> str:
    """Unofficial no-key DuckDuckGo HTML results fallback."""
    query = str(query or "").strip()
    if not query:
        return ""

    limit = max(1, min(int(limit or 8), 10))

    def clean_html(value: str) -> str:
        value = re.sub(r"(?is)<[^>]+>", " ", value or "")
        return re.sub(r"\s+", " ", unescape(value)).strip()

    def decode_ddg_url(href: str) -> str:
        href = unescape(href or "")
        if href.startswith("//"):
            href = "https:" + href

        try:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
        except Exception:
            pass

        return href

    def run():
        url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query[:500]})
        req = UrlRequest(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
            },
        )

        with urlopen(req, timeout=8) as response:
            raw = response.read(2_000_000).decode("utf-8", "replace")

        blocks = re.findall(
            r'(?is)<div[^>]+class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*</div>',
            raw,
        )
        results = []

        for block in blocks:
            link = re.search(
                r'(?is)<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
            )
            if not link:
                continue

            snippet = re.search(
                r'(?is)<(?:a|div)[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
                block,
            )

            results.append(
                {
                    "title": clean_html(link.group(2)),
                    "url": decode_ddg_url(link.group(1)),
                    "snippet": clean_html(snippet.group(1)) if snippet else "",
                }
            )

            if len(results) >= limit:
                break

        if not results:
            links = re.findall(
                r'(?is)<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                raw,
            )
            snippets = re.findall(
                r'(?is)<(?:a|div)[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
                raw,
            )

            for i, link in enumerate(links[:limit]):
                results.append(
                    {
                        "title": clean_html(link[1]),
                        "url": decode_ddg_url(link[0]),
                        "snippet": clean_html(snippets[i]) if i < len(snippets) else "",
                    }
                )

        return (
            json.dumps({"query": query, "results": results}, ensure_ascii=False)
            if results
            else ""
        )

    return await asyncio.to_thread(run)


async def _gdelt_search(query: str, limit: int = 8) -> str:
    query = str(query or "").strip()
    if not query:
        return ""

    limit = max(1, min(int(limit or 8), 20))

    def run():
        params = urlencode(
            {
                "query": query[:500],
                "mode": "ArtList",
                "maxrecords": limit,
                "format": "json",
                "sort": "HybridRel",
            }
        )

        req = UrlRequest(
            "https://api.gdeltproject.org/api/v2/doc/doc?" + params,
            headers={
                "User-Agent": "XORTRON-HF-Space/1.0",
                "Accept": "application/json",
            },
        )

        with urlopen(req, timeout=8) as response:
            data = json.loads(response.read(2_000_000).decode("utf-8", "replace"))

        articles = []
        for item in data.get("articles", [])[:limit]:
            articles.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("domain", ""),
                    "seen": item.get("seendate", ""),
                    "language": item.get("language", ""),
                }
            )

        return (
            json.dumps({"query": query, "results": articles}, ensure_ascii=False)
            if articles
            else ""
        )

    return await asyncio.to_thread(run)


async def _web_search_bundle(query: str, limit: int = 8) -> str:
    """Full no-key search used by forced lookup and the agent web_search tool."""
    query = str(query or "").strip()
    if not query:
        raise ValueError("Search query required.")

    limit = max(1, min(int(limit or 8), 10))

    instant, ddg_html, wiki, gdelt = await asyncio.gather(
        _web(query, limit),
        _ddg_html_search(query, limit),
        _wiki(query, min(limit, 6)),
        _gdelt_search(query, limit),
        return_exceptions=True,
    )

    sections = []

    if isinstance(wiki, str) and wiki and '"results": []' not in wiki:
        sections.append("WIKIPEDIA LIVE RESULTS:\n" + wiki)

    if isinstance(ddg_html, str) and ddg_html:
        sections.append("DUCKDUCKGO WEB RESULTS:\n" + ddg_html)

    if isinstance(gdelt, str) and gdelt:
        sections.append("GDELT NEWS RESULTS:\n" + gdelt)

    if (
        isinstance(instant, str)
        and instant
        and "No DuckDuckGo Instant Answer results" not in instant
        and "request failed" not in instant
    ):
        sections.append("DUCKDUCKGO INSTANT ANSWERS:\n" + instant)

    stamp = datetime.now(timezone.utc).isoformat()

    if not sections:
        return json.dumps(
            {
                "query": query,
                "retrieved_at": stamp,
                "results": [],
                "note": "All configured no-key search sources returned no usable results.",
            },
            ensure_ascii=False,
        )

    return (
        json.dumps(
            {"query": query, "retrieved_at": stamp},
            ensure_ascii=False,
        )
        + "\n\n"
        + "\n\n".join(sections)
    )[:20000]


async def _tool(name, args):
    args = args if isinstance(args, dict) else {}

    if name == "calculator":
        return _calc(args.get("expression", ""))

    if name == "get_current_time":
        return _time_tool(args)

    if name == "web_search":
        return await _web_search_bundle(
            args.get("query", ""),
            args.get("limit", 8),
        )

    if name == "search_wikipedia":
        return await _wiki(
            args.get("query", ""),
            args.get("limit", 6),
        )

    if name == "read_webpage":
        return await _read_url(
            args.get("url", ""),
            args.get("max_chars", 12000),
        )

    raise ValueError("Unknown tool: " + str(name))


def _parse_tool_payload(payload):
    raw = str(payload or "").strip()
    if not raw:
        return None

    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            return None

    if not isinstance(obj, dict):
        return None

    if isinstance(obj.get("function"), dict):
        fn = obj["function"]
        name = str(fn.get("name") or "").strip()
        args = fn.get("arguments", {})
    else:
        name = str(
            obj.get("name")
            or obj.get("tool")
            or obj.get("tool_name")
            or ""
        ).strip()
        args = obj.get("arguments", obj.get("args", {}))

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            try:
                args = ast.literal_eval(args)
            except Exception:
                args = {}

    if not isinstance(args, dict):
        args = {}

    return (name, args) if name else None


def _tool_call(text):
    value = str(text or "")

    match = re.search(
        r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
        value,
        re.I,
    )
    if match:
        return _parse_tool_payload(match.group(1))

    # Tolerate a missing closing tag when the response ended immediately after
    # the JSON payload.
    match = re.search(r"<tool_call>\s*([\s\S]+)$", value, re.I)
    if match:
        return _parse_tool_payload(match.group(1))

    return None

def _needs_forced_web_lookup(text: str) -> bool:
    value = (text or "").lower().strip()
    if not value:
        return False

    freshness = (
        "today",
        "tonight",
        "tomorrow",
        "yesterday",
        "latest",
        "recent",
        "recently",
        "current",
        "currently",
        "right now",
        "this week",
        "this weekend",
        "next week",
        "breaking",
        "happening",
        "going on",
        "open now",
        "schedule",
        "score",
        "scores",
        "weather",
        "forecast",
        "price",
        "prices",
        "stock",
        "news",
        "event",
        "events",
        "died",
        "die",
        "dead",
        "death",
        "alive",
        "still alive",
        "when did",
        "who won",
        "release date",
    )
    return any(term in value for term in freshness)



VERIFY_TERMS = (
    "look it up",
    "look that up",
    "check it",
    "check that",
    "check again",
    "search it",
    "search that",
    "verify it",
    "verify that",
    "are you sure",
    "you sure",
    "that's wrong",
    "thats wrong",
    "you're wrong",
    "youre wrong",
    "no he's not",
    "no hes not",
    "no she's not",
    "no shes not",
    "no it isn't",
    "no it isnt",
    "not true",
    "wrong",
)


def _is_verify_followup(text: str) -> bool:
    value = re.sub(r"\s+", " ", (text or "")).lower().strip()
    return bool(value and any(term in value for term in VERIFY_TERMS))


def _is_elliptical_followup(text: str) -> bool:
    value = re.sub(r"\s+", " ", (text or "")).lower().strip()
    if not value or len(value) > 180:
        return False

    if re.match(r"^(what|how)\s+about\b", value):
        return True

    if re.match(r"^and\s+.{1,80}\??$", value):
        return True

    if re.match(r"^(him|her|them|it|that one|this one)\??$", value):
        return True

    return False


def _recent_text_messages(messages: list, limit: int = 8):
    out = []
    for message in messages[-limit:]:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "")
        text = _content_text(message.get("content")).strip()
        if text:
            out.append((role, text))

    return out


def _previous_user_text(messages: list) -> str:
    current_seen = False

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        text = _content_text(message.get("content")).strip()
        if not text:
            continue

        if not current_seen:
            current_seen = True
            continue

        return text

    return ""


def _recent_live_theme(messages: list) -> str:
    recent = " ".join(
        text.lower()
        for _, text in _recent_text_messages(messages, limit=8)
    )

    if any(term in recent for term in ("died", " die", "dead", "death", "alive")):
        return "death died date alive obituary"

    if any(term in recent for term in ("weather", "forecast", "temperature", "rain", "snow")):
        return "current weather forecast"

    if any(term in recent for term in ("score", "scores", "game", "match", "won", "standings")):
        return "latest score result"

    if any(term in recent for term in ("price", "stock", "market", "cost")):
        return "current price latest"

    if any(term in recent for term in ("news", "latest", "recent", "breaking")):
        return "latest news current"

    return "current latest"


def _extract_followup_subject(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "")).strip()

    match = re.match(r"(?i)^(?:what|how)\s+about\s+(.+?)[?.!]*$", value)
    if match:
        subject = match.group(1).strip(" ?.!,:;")
        if subject:
            return subject

    match = re.match(r"(?i)^and\s+(.+?)[?.!]*$", value)
    if match:
        subject = match.group(1).strip(" ?.!,:;")
        if subject:
            return subject

    return ""


def _extract_named_subject(text: str) -> str:
    if not text:
        return ""

    candidates = re.findall(
        r"\b([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,3})\b",
        text,
    )

    reject = {
        "Duck Duck Go",
        "Tool Result",
        "Server Tool Execution",
        "Web Results Received",
    }

    for candidate in candidates:
        candidate = candidate.strip()
        if candidate not in reject and len(candidate) <= 80:
            return candidate

    return ""


def _recent_subject(messages: list) -> str:
    current_user_seen = False

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "")
        text = _content_text(message.get("content")).strip()
        if not text:
            continue

        if role == "user" and not current_user_seen:
            current_user_seen = True
            continue

        if not current_user_seen:
            continue

        subject = _extract_named_subject(text)
        if subject:
            return subject

    previous = _previous_user_text(messages)
    if previous:
        cleaned = re.sub(
            r"(?i)\b(when did|did|does|is|was|who is|who was|what happened to|what about|how about)\b",
            " ",
            previous,
        )
        cleaned = re.sub(
            r"(?i)\b(die|died|dead|death|alive|still alive|today|current|latest)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.!,:;")
        if cleaned:
            return cleaned[:100]

    return ""


def _build_contextual_search_query(messages: list):
    """
    Carry live-search intent across conversational follow-ups.

    Example:
      "when did dolly parton die?"
      "what about ozzy?"

    The second turn is converted to a live search such as:
      "ozzy death died date alive obituary"
    instead of falling through to stale model memory.
    """
    latest = _latest_user_text(messages)
    if not latest:
        return None

    if _needs_forced_web_lookup(latest):
        return latest[:500]

    previous_user = _previous_user_text(messages)
    recent_context_is_live = _needs_forced_web_lookup(previous_user)

    # The last assistant answer can also carry the live topic.
    recent_before_current = _recent_text_messages(messages, limit=6)[:-1]
    if any(_needs_forced_web_lookup(text) for _, text in recent_before_current):
        recent_context_is_live = True

    if _is_elliptical_followup(latest) and recent_context_is_live:
        subject = _extract_followup_subject(latest) or _recent_subject(messages)
        if subject:
            return f"{subject} {_recent_live_theme(messages)}"[:500]

    if _is_verify_followup(latest):
        subject = _recent_subject(messages)
        if subject:
            return f"{subject} {_recent_live_theme(messages)}"[:500]

        for role, text in reversed(recent_before_current):
            if role in ("assistant", "user"):
                return f"{text[:300]} {_recent_live_theme(messages)}"[:500]

    return None


def _is_local_event_query(text: str) -> bool:
    value = (text or "").lower()
    return any(
        term in value
        for term in (
            "what's happening",
            "whats happening",
            "what is happening",
            "what's going on",
            "whats going on",
            "what is going on",
            "things to do",
            "events",
            "event near",
            "happening in",
        )
    )


async def _forced_web_context(query: str) -> str:
    """Mandatory multi-source live retrieval for an already-resolved search query."""
    query = str(query or "").strip()
    stamp = datetime.now(timezone.utc).isoformat()

    if not query:
        return (
            "SERVER TOOL EXECUTION NOTICE\n"
            f"Retrieval time: {stamp}\n"
            "No usable search query could be constructed."
        )

    try:
        results = await _web_search_bundle(query, 8)
    except Exception as exc:
        return (
            "SERVER TOOL EXECUTION NOTICE\n"
            f"Retrieval time: {stamp}\n"
            f"Live search failed: {exc}\n"
            "Do not present stale model memory as a verified current fact. "
            "Tell the user the live lookup failed if the answer depends on it."
        )

    return (
        "SERVER TOOL EXECUTION NOTICE\n"
        f"Retrieval time: {stamp}\n"
        f"Search query actually executed: {query}\n"
        "The server performed a live multi-source lookup. Treat the retrieved "
        "material below as evidence that overrides stale model memory for current "
        "facts. If sources disagree, say so. Do not claim that browsing was not "
        "available. Do not invent a fact that is absent from the retrieved data.\n\n"
        + results
    )[:20000]


# ---------------------------------------------------------------------------
# TRUE streaming agent
# ---------------------------------------------------------------------------


def _delta_event(text: str) -> str:
    payload = {"choices": [{"delta": {"content": text}}]}
    return f"data: {json.dumps(payload)}\n\n"


async def _agent_stream(messages, thinking, browser_enabled=False):
    """
    True token-streaming agent.

    Normal answer text streams as it arrives. Tool markup is detected anywhere
    in the response, hidden from the UI, executed server-side, and followed by
    another model round containing the tool result.
    """
    work = list(messages)
    work.insert(
        1 if work and work[0].get("role") == "system" else 0,
        {"role": "system", "content": AGENT_PROMPT + (BROWSER_PROMPT if browser_enabled else "")},
    )

    for _ in range(max(1, min(AGENT_MAX_ROUNDS, 10))):
        stream = await client.chat.completions.create(
            model=MODEL_ID,
            messages=fit_context(work),
            stream=True,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            extra_body={
                "top_k": TOP_K,
                "chat_template_kwargs": {"enable_thinking": thinking},
            },
        )

        thought_filter = ThoughtFilter()
        tool_filter = ToolStreamFilter()
        visible_prefix = ""

        async for chunk in stream:
            if not chunk.choices:
                continue

            token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            piece = thought_filter.process(token)
            if not piece:
                continue

            safe = tool_filter.process(piece)
            if safe:
                visible_prefix += safe
                yield _delta_event(safe)

        leftover = thought_filter.flush()
        if leftover:
            safe = tool_filter.process(leftover)
            if safe:
                visible_prefix += safe
                yield _delta_event(safe)

        call = tool_filter.get_call()

        if not call:
            tail = tool_filter.flush_visible()
            if tail:
                yield _delta_event(tail)
            return

        name, args = call

        label = {
            "browser_python": "Analyzing and calculating",
            "browser_write_file": "Creating file",
            "browser_read_file": "Reading file",
            "browser_list_files": "Checking files",
            "calculator": "Calculating",
            "get_current_time": "Checking time",
            "web_search": "Searching web",
            "search_wikipedia": "Searching Wikipedia",
            "read_webpage": "Reading webpage",
        }.get(name, f"Using {name}")

        activity_id = secrets.token_hex(8)
        activity_start = time.monotonic()
        activity = {"id": activity_id, "name": name, "label": label,
                    "query": str(args.get("query") or args.get("url") or args.get("name") or "")[:400],
                    "code": str(args.get("code") or "")[:12000], "status": "running"}
        yield f"data: {json.dumps({'tool_activity': activity})}\n\n"
        yield f"data: {json.dumps({'agent_status': label})}\n\n"
        await asyncio.sleep(0)

        try:
            if name in BROWSER_TOOLS and browser_enabled:
                async with aclosing(browser_call(name, args)) as browser_steps:
                    async for step in browser_steps:
                        if "browser_tool" in step:
                            yield f"data: {json.dumps(step)}\n\n"
                        else:
                            result = step["result"]
            else:
                result = await _tool(name, args)
        except Exception as exc:
            result = "Tool error: " + str(exc)

        activity.update({"status": "error" if "Tool error:" in str(result) or "Python error" in str(result) else "complete",
                         "seconds": round(time.monotonic() - activity_start, 1),
                         "result": str(result)[:14000],
                         "sources": activity_sources(result) if name in {"web_search", "read_webpage", "search_wikipedia"} else []})
        yield f"data: {json.dumps({'tool_activity': activity})}\n\n"

        canonical = (
            "<tool_call>"
            + json.dumps(
                {"name": name, "arguments": args},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "</tool_call>"
        )

        work += [
            {
                "role": "assistant",
                "content": (
                    (visible_prefix + "\n" if visible_prefix.strip() else "")
                    + canonical
                ),
            },
            {
                "role": "system",
                "content": (
                    f"TOOL RESULT ({name}):\n"
                    f"{str(result)[:AGENT_RESULT_LIMIT]}"
                ),
            },
        ]

    work.append(
        {
            "role": "system",
            "content": (
                "Tool-round limit reached. Answer now using the available tool "
                "results. Do not emit another tool call."
            ),
        }
    )

    stream = await client.chat.completions.create(
        model=MODEL_ID,
        messages=fit_context(work),
        stream=True,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body={
            "top_k": TOP_K,
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
    )

    thought_filter = ThoughtFilter()
    tool_filter = ToolStreamFilter()

    async for chunk in stream:
        if not chunk.choices:
            continue

        token = chunk.choices[0].delta.content or ""
        if not token:
            continue

        piece = thought_filter.process(token)
        if piece:
            safe = tool_filter.process(piece)
            if safe:
                yield _delta_event(safe)

    leftover = thought_filter.flush()
    if leftover:
        safe = tool_filter.process(leftover)
        if safe:
            yield _delta_event(safe)

    tail = tool_filter.flush_visible()
    if tail:
        yield _delta_event(tail)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def get_ui() -> str:
    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.post("/v1/chat/completions")
async def chat_stream(request: Request) -> StreamingResponse:
    body = await request.json()
    raw_messages = body.get("messages", [])
    base_messages = list(raw_messages) if isinstance(raw_messages, list) else []

    fast_mode = body.get("fast", True) is True
    enable_thinking = not fast_mode
    agent_enabled = body.get("agent", False) is True

    # Only inject an app-level system message when the secret actually contains text.
    if SYSTEM_MESSAGE and not any(
        isinstance(message, dict) and message.get("role") == "system"
        for message in base_messages
    ):
        base_messages.insert(0, {"role": "system", "content": SYSTEM_MESSAGE})

    async def send_model_stream(messages):
        model_messages = fit_context(messages)
        stream = await client.chat.completions.create(
            model=MODEL_ID,
            messages=model_messages,
            stream=True,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            extra_body={
                "top_k": TOP_K,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )

        thought_filter = ThoughtFilter()
        tool_filter = ToolStreamFilter()

        async for chunk in stream:
            if not chunk.choices:
                continue

            token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            filtered = thought_filter.process(token)
            if not filtered:
                continue

            safe = tool_filter.process(filtered)
            if safe:
                yield _delta_event(safe)

        leftover = thought_filter.flush()
        if leftover:
            safe = tool_filter.process(leftover)
            if safe:
                yield _delta_event(safe)

        tail = tool_filter.flush_visible()
        if tail:
            yield _delta_event(tail)

    async def event_generator():
        if FEATHERLESS_SEMAPHORE.locked():
            queue_message = {
                "choices": [
                    {
                        "delta": {
                            "content": "⏳ High traffic. Your request is queued and will process shortly...\n\n"
                        }
                    }
                ]
            }
            yield f"data: {json.dumps(queue_message)}\n\n"
            await asyncio.sleep(0.05)

        async with FEATHERLESS_SEMAPHORE:
            try:
                # Refresh after queueing so midnight or a long wait cannot leave an old date.
                messages = [current_time_context(), *base_messages]
                latest_text = _latest_user_text(messages)

                if agent_enabled:
                    # Deterministic live lookup. This includes short contextual
                    # follow-ups such as "what about ozzy?" after a death/current
                    # question, plus corrections like "no he's not" and "look it up".
                    search_query = _build_contextual_search_query(messages)

                    if search_query:
                        search_activity_id = secrets.token_hex(8)
                        search_started = time.monotonic()
                        yield f"data: {json.dumps({'tool_activity': {'id': search_activity_id, 'label': 'Searching web', 'query': search_query, 'status': 'running'}})}\n\n"
                        yield f"data: {json.dumps({'agent_status': 'Searching web'})}\n\n"
                        await asyncio.sleep(0)

                        try:
                            forced_context = await _forced_web_context(search_query)
                        except Exception as exc:
                            forced_context = (
                                "SERVER TOOL EXECUTION NOTICE\n"
                                f"A live search was attempted but failed: {exc}. "
                                "Do not claim that no browsing capability exists; report that this lookup failed."
                            )

                        yield f"data: {json.dumps({'tool_activity': {'id': search_activity_id, 'label': 'Searched web', 'query': search_query, 'status': 'complete', 'seconds': round(time.monotonic() - search_started, 1), 'sources': activity_sources(forced_context), 'result': forced_context[:14000]}})}\n\n"
                        insert_at = (
                            1
                            if messages and messages[0].get("role") == "system"
                            else 0
                        )
                        messages.insert(
                            insert_at,
                            {"role": "system", "content": forced_context},
                        )
                        messages = fit_context(messages)

                        yield f"data: {json.dumps({'agent_status': 'Web results received'})}\n\n"
                        await asyncio.sleep(0)

                        async for event in _agent_stream(messages, enable_thinking, body.get("browser_tools") is True):
                            yield event

                        yield "data: [DONE]\n\n"
                        return

                    # Explicit URLs are read server-side first, then the model answer streams.
                    urls = re.findall(
                        r"https?://[^\s<>()\]\[\"']+", latest_text or ""
                    )
                    if urls and body.get("browser_tools") is not True:
                        yield f"data: {json.dumps({'agent_status': 'Reading webpage'})}\n\n"
                        await asyncio.sleep(0)

                        insert_at = (
                            1
                            if messages and messages[0].get("role") == "system"
                            else 0
                        )

                        try:
                            page_result = await _read_url(urls[0], 16000)
                            messages.insert(
                                insert_at,
                                {
                                    "role": "system",
                                    "content": (
                                        "SERVER TOOL EXECUTION NOTICE\n"
                                        "The server successfully executed read_webpage for the URL in the user request. "
                                        "Do not say you cannot open links. Use the retrieved page below.\n\n"
                                        + page_result
                                    ),
                                },
                            )
                            yield f"data: {json.dumps({'agent_status': 'Webpage read'})}\n\n"
                            await asyncio.sleep(0)
                        except Exception as exc:
                            messages.insert(
                                insert_at,
                                {
                                    "role": "system",
                                    "content": (
                                        "SERVER TOOL EXECUTION NOTICE\n"
                                        f"read_webpage was attempted but failed: {exc}. "
                                        "Do not claim no tool exists; say this page could not be retrieved."
                                    ),
                                },
                            )

                        messages = fit_context(messages)

                        async for event in _agent_stream(messages, enable_thinking, body.get("browser_tools") is True):
                            yield event

                        yield "data: [DONE]\n\n"
                        return

                    # FIX: The ordinary agent path now truly streams from Featherless.
                    # Actual <tool_call> responses are buffered only long enough to run
                    # the tool; normal/final answers stream token-by-token.
                    async for event in _agent_stream(messages, enable_thinking, body.get("browser_tools") is True):
                        yield event

                    yield "data: [DONE]\n\n"
                    return

                # Agent off: keep lightweight web assistance, but carry
                # freshness intent across short follow-up turns too.
                if body.get("web", True) is not False:
                    search_query = _build_contextual_search_query(messages)

                    if search_query:
                        live_context = await _forced_web_context(search_query)
                        insert_at = (
                            1
                            if messages and messages[0].get("role") == "system"
                            else 0
                        )
                        messages.insert(
                            insert_at,
                            {"role": "system", "content": live_context},
                        )
                    else:
                        ddg_context = await duckduckgo_context(latest_text)
                        if ddg_context:
                            insert_at = (
                                1
                                if messages and messages[0].get("role") == "system"
                                else 0
                            )
                            messages.insert(
                                insert_at,
                                {"role": "system", "content": ddg_context},
                            )

                messages = fit_context(messages)

                async for event in send_model_stream(messages):
                    yield event

                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                raise
            except Exception as error:
                error_text = str(error)
                lower_error = error_text.lower()

                if any(
                    term in lower_error
                    for term in ("vision", "image_url", "image input", "multimodal")
                ):
                    error_text = (
                        "This Featherless model does not appear to support image input. "
                        "Use a vision-capable MODEL_ID, then try the photo again. "
                        f"Provider detail: {error_text}"
                    )

                payload = {
                    "choices": [
                        {"delta": {"content": f"⚠️ Error: {error_text}"}}
                    ]
                }
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
