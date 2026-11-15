"""One OpenAI-compatible chat client, no SDK.

The repo installs with `numpy` and nothing else, so the HTTP call is `urllib`. That is
also the point: the same 120 lines drive DeepSeek, OpenAI, OpenRouter, vLLM and Ollama,
because all of them speak `/chat/completions` with the same tool-calling shape. Swapping
providers is a base URL and a model name, which is what "hands-on with several providers"
should mean in practice.

Cost is computed from the tokens the provider reports, not estimated from the transcript.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True)
class Price:
    """USD per million tokens. DeepSeek bills cache hits separately."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None

    def cost(self, prompt: int, completion: int, cached: int = 0) -> float:
        cached = min(max(cached, 0), prompt)
        cached_price = self.input_per_mtok if self.cached_input_per_mtok is None else self.cached_input_per_mtok
        return (
            (prompt - cached) * self.input_per_mtok
            + cached * cached_price
            + completion * self.output_per_mtok
        ) / 1_000_000


# List prices as of 2026-11, USD per million tokens. Override with FORECAST_LOG_PRICE_*.
PRICES = {
    "deepseek-chat": Price(0.28, 0.42, 0.028),
    "deepseek-reasoner": Price(0.28, 0.42, 0.028),
    "deepseek-flash": Price(0.28, 0.42, 0.028),
    "qwen3.8-flash": Price(0.110, 0.370),
}
DEFAULT_PRICE = Price(0.0, 0.0, 0.0)  # unknown model: report tokens, not pretend dollars


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""
    parse_error: str = ""


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
        )

    def to_json(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_s": round(self.latency_s, 3),
        }


@dataclass(frozen=True)
class Reply:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    # 思考型模型（DeepSeek V4、o 系列）在多轮工具调用里要求把上一轮的思考原样传回，
    # 否则下一轮直接 400。丢掉它就等于把这类模型挡在门外。
    reasoning: str = ""


class LLM(Protocol):
    name: str

    def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1600,
        tool_choice: dict | None = None,
    ) -> Reply: ...


class LLMError(RuntimeError):
    pass


class OpenAICompatLLM:
    """Any `/chat/completions` endpoint that supports tools."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        price: Price | None = None,
        timeout: float = 120.0,
        max_attempts: int = 4,
        name: str | None = None,
    ):
        if not api_key:
            raise LLMError("api key is empty")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.price = PRICES.get(model, DEFAULT_PRICE) if price is None else price
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.name = name or f"{model}@{base_url.split('//')[-1].split('/')[0]}"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> OpenAICompatLLM:
        env = os.environ if env is None else env
        model = env.get("FORECAST_LOG_MODEL", "deepseek-chat")
        key = (
            env.get("FORECAST_LOG_API_KEY")
            or env.get("DEEPSEEK_API_KEY")
            or env.get("OPENAI_API_KEY")
            or ""
        )
        base_url = env.get("FORECAST_LOG_BASE_URL", "https://api.deepseek.com/v1")
        price = None
        if env.get("FORECAST_LOG_PRICE_INPUT") and env.get("FORECAST_LOG_PRICE_OUTPUT"):
            price = Price(
                float(env["FORECAST_LOG_PRICE_INPUT"]),
                float(env["FORECAST_LOG_PRICE_OUTPUT"]),
                float(env["FORECAST_LOG_PRICE_CACHED"]) if env.get("FORECAST_LOG_PRICE_CACHED") else None,
            )
        return cls(model=model, api_key=key, base_url=base_url, price=price)

    def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1600,
        tool_choice: dict | None = None,
    ) -> Reply:
        payload: dict = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = tool_choice or "auto"
        body, latency = self._chat_post(payload, forced=bool(tool_choice))
        reply = self._parse(body, latency)
        # 思考型模型把推理也算进 max_tokens：预算用完会返回空 content、没有 tool_calls、
        # finish_reason=length。这时候不是模型没有答案，是它没被给够空间。翻倍重试。
        budget = int(payload.get("max_tokens") or 1600)
        while (
            not reply.content
            and not reply.tool_calls
            and reply.finish_reason == "length"
            and budget < 16000
        ):
            budget *= 2
            payload["max_tokens"] = budget
            body, latency = self._chat_post(payload, forced=False)
            reply = self._parse(body, latency)
        return reply

    def _chat_post(self, payload: dict, forced: bool) -> tuple[dict, float]:
        """思考型模型（DeepSeek V4、o 系列）不接受 tool_choice，连 "auto" 都不行，直接 400。

        `tools` 字段本身是支持的，去掉 `tool_choice` 就正常。不这么兜，一次 400 就会变成
        「这道题没有答案」，把整次评测的题集切碎。
        """
        try:
            return self._post(payload)
        except LLMError as error:
            if "tool_choice" not in str(error):
                raise
            return self._post({key: value for key, value in payload.items() if key != "tool_choice"})

    def _post(self, payload: dict) -> tuple[dict, float]:
        data = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        last = ""
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode()), time.perf_counter() - started
            except urllib.error.HTTPError as error:
                last = f"HTTP {error.code}: {error.read()[:400].decode(errors='replace')}"
                retryable = error.code in (408, 409, 429, 500, 502, 503, 504)
                if not retryable or attempt == self.max_attempts:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last = f"{type(error).__name__}: {error}"
                if attempt == self.max_attempts:
                    break
            time.sleep(min(2 ** (attempt - 1), 8))
        raise LLMError(f"{self.name} failed after {self.max_attempts} attempts: {last}")

    def _parse(self, body: dict, latency: float) -> Reply:
        usage = body.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cached = int(
            usage.get("prompt_cache_hit_tokens")
            or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            or 0
        )
        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"{self.name} returned no choices: {json.dumps(body)[:400]}")
        choice = choices[0]
        message = choice.get("message") or {}
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw = function.get("arguments") or ""
            parse_error = ""
            try:
                arguments = json.loads(raw) if raw.strip() else {}
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
            except (json.JSONDecodeError, ValueError) as error:
                arguments, parse_error = {}, f"unparseable arguments: {error}"
            calls.append(
                ToolCall(
                    id=str(call.get("id") or f"call-{len(calls)}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                    raw_arguments=raw,
                    parse_error=parse_error,
                )
            )
        return Reply(
            content=message.get("content"),
            tool_calls=tuple(calls),
            usage=Usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_tokens=cached,
                cost_usd=self.price.cost(prompt, completion, cached),
                latency_s=latency,
            ),
            finish_reason=str(choice.get("finish_reason") or ""),
            reasoning=str(message.get("reasoning_content") or ""),
        )


class ScriptedLLM:
    """Deterministic stand-in for tests and for dry runs without a key.

    Accepts either a list of `Reply` (played in order) or a callable that sees the
    transcript and returns one, which is how the tests drive specific branches.
    """

    def __init__(self, script: Sequence[Reply] | Callable[[list[dict]], Reply], name: str = "scripted"):
        self.name = name
        self.script = script
        self.calls: list[list[dict]] = []
        self.tool_names_seen: list[list[str]] = []
        self.tool_choice_seen: list[dict | None] = []

    def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 0,
        tool_choice: dict | None = None,
    ) -> Reply:
        self.calls.append([dict(message) for message in messages])
        self.tool_names_seen.append([spec["function"]["name"] for spec in (tools or [])])
        self.tool_choice_seen.append(tool_choice)
        if callable(self.script):
            return self.script(self.calls[-1])
        if not self.script:
            raise LLMError("script is empty")
        return self.script[0] if len(self.script) == 1 else self.script.pop(0)
