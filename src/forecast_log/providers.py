"""Name a labeler, get a client. Kept separate so an A/B can hold two vendors at once.

`FORECAST_LOG_*` describes **one** provider; that is right for a run, wrong for a
comparison. Here each labeler resolves from its own block in `.env`:

    deepseek-flash  ->  DEEPSEEK_API_KEY,  https://api.deepseek.com/v1
    qwen3.8-flash   ->  QWEN_API_KEY, QWEN_BASE_URL, QWEN_PRICE_INPUT/OUTPUT

A prefix with no key in the environment raises instead of silently falling back to the
other vendor: two "different" labelers that are secretly the same model is the one bug
that would make this whole file worthless.
"""

from __future__ import annotations

import os
from pathlib import Path

from forecast_log.agent.env import merged_env
from forecast_log.agent.llm import LLMError, OpenAICompatLLM, Price

VENDORS = {
    "deepseek": {
        "key": ("DEEPSEEK_API_KEY",),
        "base_url": ("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "price": ("DEEPSEEK_PRICE_INPUT", "DEEPSEEK_PRICE_OUTPUT", "DEEPSEEK_PRICE_CACHED"),
        "timeout": 120.0,
    },
    "qwen": {
        "key": ("QWEN_API_KEY",),
        "base_url": ("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "price": ("QWEN_PRICE_INPUT", "QWEN_PRICE_OUTPUT", None),
        # 思考型模型读 10 天标题要 100 秒上下，120 秒的默认超时会把正常请求判成失败，
        # 然后 4 次重试各等 120 秒——看起来像卡死。
        "timeout": 300.0,
    },
}


def vendor_of(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith("qwen") or lowered.startswith("tongyi"):
        return "qwen"
    if lowered.startswith("deepseek"):
        return "deepseek"
    return "deepseek"


def build(model: str, root: Path, env: dict[str, str] | None = None) -> OpenAICompatLLM:
    env = merged_env(root) if env is None else env
    spec = VENDORS[vendor_of(model)]
    key = next((env[name] for name in spec["key"] if env.get(name)), "")
    if not key:
        raise LLMError(f"{model}: 环境里没有 {spec['key'][0]}，不能拿另一家顶上")
    base_name, base_default = spec["base_url"]
    price = None
    in_name, out_name, cached_name = spec["price"]
    if env.get(in_name) and env.get(out_name):
        price = Price(
            float(env[in_name]),
            float(env[out_name]),
            float(env[cached_name]) if cached_name and env.get(cached_name) else None,
        )
    return OpenAICompatLLM(
        model=model,
        api_key=key,
        base_url=env.get(base_name, base_default),
        price=price,
        timeout=spec.get("timeout", 120.0),
    )


def describe(model: str, env: dict[str, str] | None = None) -> dict:
    """For the report: which host and which price a labeler actually ran on."""
    env = merged_env(None) if env is None else env
    spec = VENDORS[vendor_of(model)]
    base_name, base_default = spec["base_url"]
    return {
        "model": model,
        "vendor": vendor_of(model),
        "base_url": env.get(base_name, base_default),
        "key_present": bool(next((env[name] for name in spec["key"] if env.get(name)), "")),
        "cwd_env_override": bool(os.environ.get("FORECAST_LOG_MODEL")),
    }
