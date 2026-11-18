"""Chat backends: Nebius Token Factory, local llama.cpp, or neither.

Both backends speak the OpenAI chat-completions dialect, so the transport is
one function and the only real difference is the base URL, the auth header and
whether a model name has to be named explicitly.

Why two: the local path keeps the "nothing leaves the machine" guarantee for
unpublished structures, and the hosted path gives the pipeline a frontier-scale
Nemotron that can actually hold a tool-calling loop together. Which one is in
use is reported to the UI rather than inferred, because a screening tool that
silently changes where its molecules are sent is a tool nobody should trust.

Neither backend is load-bearing for the science. Every caller here has a
deterministic fallback, and the numeric pipeline never consults this module.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")

# Nebius Token Factory is OpenAI-compatible, so it needs no separate client.
NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_MODEL = os.environ.get("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b")

# Hosted inference can stall on a cold route; local llama.cpp can stall on a
# long prompt. Both are bounded so a hung backend degrades to the rule-based
# path instead of hanging the request.
TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "90"))

# Nemotron 3 is a reasoning model: it emits `reasoning_content` before
# `content`, and both are charged against max_tokens. A budget sized for a
# non-reasoning model gets spent entirely on reasoning and returns empty
# content, which looks exactly like a backend failure. Callers pass generous
# budgets for that reason -- see DEFAULT_MAX_TOKENS.
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2000"))


@dataclass(frozen=True)
class Backend:
    """A resolved chat endpoint. `name` is what the UI shows the user."""

    name: str
    url: str
    model: str | None
    api_key: str | None

    @property
    def configured(self) -> bool:
        return bool(self.url) and (self.name != "nebius" or bool(self.api_key))


def nebius_backend() -> Backend:
    return Backend("nebius", f"{NEBIUS_BASE_URL.rstrip('/')}/chat/completions",
                   NEBIUS_MODEL, NEBIUS_API_KEY or None)


def llama_backend() -> Backend:
    return Backend("llama.cpp", f"{LLAMA_URL.rstrip('/')}/v1/chat/completions",
                   None, None)


def resolve(prefer: str) -> list[Backend]:
    """Map the caller's `llm` preference onto an ordered list to try.

    "auto" prefers Nebius when a key is present and falls back to local, so a
    judge running the public demo gets the hosted path and a chemist running it
    offline gets the local one, from the same default.
    """
    nebius, llama = nebius_backend(), llama_backend()
    if prefer == "rules":
        return []
    if prefer == "nebius":
        return [nebius] if nebius.configured else []
    if prefer == "llama":
        return [llama]
    return ([nebius] if nebius.configured else []) + [llama]


def _headers(backend: Backend) -> dict:
    if backend.api_key:
        return {"Authorization": f"Bearer {backend.api_key}"}
    return {}


def _body(backend: Backend, messages: list[dict], max_tokens: int,
          tools: list[dict] | None, temperature: float) -> dict:
    body: dict = {"messages": messages, "temperature": temperature,
                  "max_tokens": max_tokens}
    if backend.model:
        body["model"] = backend.model
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


def chat(messages: list[dict], prefer: str = "auto",
         max_tokens: int = DEFAULT_MAX_TOKENS,
         tools: list[dict] | None = None,
         temperature: float = 0.1) -> tuple[dict | None, str | None]:
    """Try each resolved backend in turn.

    Returns (assistant_message, backend_name), or (None, None) if every
    backend failed. The assistant message is returned whole rather than as a
    string because tool-calling callers need `tool_calls` alongside `content`.
    """
    for backend in resolve(prefer):
        try:
            r = httpx.post(backend.url, json=_body(backend, messages, max_tokens,
                                                   tools, temperature),
                           headers=_headers(backend), timeout=TIMEOUT_SECONDS)
            r.raise_for_status()
            return r.json()["choices"][0]["message"], backend.name
        except Exception:
            continue  # next backend, then the caller's deterministic floor
    return None, None


def complete(system: str, user: str, prefer: str = "auto",
             max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[str | None, str | None]:
    """Single-turn convenience wrapper. Returns (text, backend_name)."""
    message, name = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        prefer=prefer, max_tokens=max_tokens)
    if message is None:
        return None, None
    return (message.get("content") or "").strip() or None, name


# The Nebius reachability probe is a real (tiny) completion, so it is billed
# and rate-limited like any other. Health is polled by the UI, so the result is
# cached briefly -- long enough that a page refresh is free, short enough that
# an outage surfaces quickly.
_PROBE_TTL_SECONDS = 60.0
_probe_cache: tuple[float, bool] | None = None


def _nebius_reachable(backend: Backend) -> bool:
    global _probe_cache
    now = time.monotonic()
    if _probe_cache and now - _probe_cache[0] < _PROBE_TTL_SECONDS:
        return _probe_cache[1]
    try:
        # A 1-token completion is the only honest reachability check: the
        # models endpoint can answer while inference is unavailable.
        r = httpx.post(backend.url,
                       json={"model": backend.model, "max_tokens": 1,
                             "messages": [{"role": "user", "content": "ping"}]},
                       headers=_headers(backend), timeout=10.0)
        ok = r.status_code == 200
    except Exception:
        ok = False
    _probe_cache = (now, ok)
    return ok


def status() -> dict:
    """Which backends are actually reachable right now.

    Probed rather than assumed: a key in the environment is not the same as a
    working endpoint, and the UI should not claim a backend it cannot use.
    """
    out = {"llama_cpp": False, "nebius": False, "nebius_configured": False,
           "nebius_model": NEBIUS_MODEL, "rules": True}
    try:
        out["llama_cpp"] = httpx.get(f"{LLAMA_URL.rstrip('/')}/health",
                                     timeout=2.0).status_code == 200
    except Exception:
        pass

    backend = nebius_backend()
    out["nebius_configured"] = backend.configured
    if backend.configured:
        out["nebius"] = _nebius_reachable(backend)
    return out
