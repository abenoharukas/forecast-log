"""双模型群聊研判（DeepSeek × Qwen）：基于当前、近 5 天、近 30 天、近 3 个月行情上下文轮流对话。

核心流程：
1. `build_market_context(root, market)`：从本地 `bars` 与 `signals` 提取当前（最新交易日）、最近 5 天、
   最近 30 天、最近 3 个月的客观行情摘要，不引入外部慢请求。
2. `run_group_chat_stream(...)`：
   - 新会话首条写入 `context` 消息落库；若有用户输入则追加 `user` 消息落库。
   - 默认 3 轮/人（共 6 条 AI 消息）：DeepSeek 与 Qwen 交替发言，围绕「今天偏买入、卖出还是观望」辩论收敛。
   - 每生成一条消息立即写入 `data/forecast.db` 的 `chat_messages` 表，并通过生成器实时推送给前端。
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from forecast_log import providers
from forecast_log.agent.env import merged_env
from forecast_log.agent.llm import LLM
from forecast_log.artifacts_db import DEFAULT_INSTRUMENTS, load_cross_market
from forecast_log.db import append_chat_message, get_chat_thread, get_series

MARKET_LABELS = {
    "ndx": "Nasdaq-100 (^NDX)",
    "qqq": "QQQ (纳指100 ETF)",
    "ssec": "A股 · 上证综指 (000001.SS)",
}

STANCE_RE = re.compile(r"【\s*(买入|卖出|观望|BUY|SELL|HOLD)\s*】", re.IGNORECASE)


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.2f}%"


def _window_stats(bars: list, count: int) -> dict | None:
    if not bars:
        return None
    sub = bars[-count:] if len(bars) >= count else list(bars)
    if not sub:
        return None
    first_bar = sub[0]
    last_bar = sub[-1]
    # 区间基准价优先取窗口前一日收盘价，若已在序列最前端则取首日开盘/收盘价
    start_idx = len(bars) - len(sub)
    base_close = (
        bars[start_idx - 1].close
        if start_idx > 0
        else (first_bar.open if first_bar.open is not None else first_bar.close)
    )
    ret = (last_bar.close / base_close - 1.0) if base_close else 0.0
    highs = [b.high if b.high is not None else b.close for b in sub]
    lows = [b.low if b.low is not None else b.close for b in sub]
    hi = max(highs)
    lo = min(lows)
    pos = ((last_bar.close - lo) / (hi - lo)) if hi > lo else 0.5
    up_days = 0
    for i, b in enumerate(sub):
        prev = bars[start_idx + i - 1].close if (start_idx + i) > 0 else (b.open or b.close)
        if b.close >= prev:
            up_days += 1
    avg_close = sum(b.close for b in sub) / len(sub)
    return {
        "sessions": len(sub),
        "from_date": first_bar.date.isoformat(),
        "to_date": last_bar.date.isoformat(),
        "return": ret,
        "high": hi,
        "low": lo,
        "range_pos": pos,
        "up_days": up_days,
        "avg_close": avg_close,
    }


def build_market_context(root: Path, market: str) -> dict:
    """构造精简四档上下文：当前（最新交易日）、最近 5 天、最近 30 天（约 22 交易日）、最近 3 个月（约 63 交易日）。"""
    mkt = market if market in DEFAULT_INSTRUMENTS else "ndx"
    symbol = DEFAULT_INSTRUMENTS[mkt]
    label = MARKET_LABELS.get(mkt, symbol)

    series = get_series(root, symbol)
    if series is None or not series.bars:
        return {
            "market": mkt,
            "symbol": symbol,
            "as_of": "",
            "text": f"标的：{label}\n暂无本地行情数据，请先点击顶部「刷新行情」。",
        }

    bars = list(series.bars)
    latest = bars[-1]
    prev_close = bars[-2].close if len(bars) >= 2 else (latest.open or latest.close)
    day_ret = (latest.close / prev_close - 1.0) if prev_close else 0.0
    o = latest.open if latest.open is not None else latest.close
    h = latest.high if latest.high is not None else latest.close
    l = latest.low if latest.low is not None else latest.close
    status_str = "已收盘" if latest.is_closed else "盘中未收盘"

    # 读取最新规则信号（辅助参考）
    sig_line = ""
    cross = load_cross_market(root, mkt, _include_cross_ref=False)
    if cross and cross.get("latest"):
        lat_sig = cross["latest"]
        flag_str = "触发买入信号 (BUY)" if lat_sig.get("flag") else "观望 / 无信号 (NO SIGNAL)"
        score = lat_sig.get("score")
        thresh = lat_sig.get("threshold")
        if score is not None and thresh is not None:
            sig_line = f"量化因子参考：{flag_str}（合成得分 {score:.3f} / 阈值 {thresh:.3f}）"
        else:
            sig_line = f"量化因子参考：{flag_str}"

    w5 = _window_stats(bars, 5)
    w30 = _window_stats(bars, 22)   # 最近约 30 个自然日 ≈ 22 个交易日
    w90 = _window_stats(bars, 63)   # 最近约 3 个月 ≈ 63 个交易日

    # 最近 5 个交易日逐日表现
    recent5_parts = []
    start5 = max(0, len(bars) - 5)
    for idx in range(start5, len(bars)):
        b = bars[idx]
        pc = bars[idx - 1].close if idx > 0 else (b.open or b.close)
        r = (b.close / pc - 1.0) if pc else 0.0
        recent5_parts.append(f"{b.date.isoformat()[5:]} {b.close:.2f}({_fmt_pct(r)})")

    lines = [
        f"【标的】{label} · 基准日 {latest.date.isoformat()}（{status_str}）",
        (
            f"1. 当前盘面：{latest.date.isoformat()} 收/现价 {latest.close:.2f}（当日 {_fmt_pct(day_ret)}），"
            f"开 {o:.2f} / 高 {h:.2f} / 低 {l:.2f}"
            + (f"；{sig_line}" if sig_line else "")
        ),
    ]
    if w5:
        lines.append(
            f"2. 最近 5 天（{w5['from_date']} → {w5['to_date']}，{w5['sessions']} 个交易日）："
            f"累计 {_fmt_pct(w5['return'])}，高 {w5['high']:.2f} / 低 {w5['low']:.2f}，"
            f"收涨 {w5['up_days']}/{w5['sessions']} 天；逐日：{' · '.join(recent5_parts)}"
        )
    if w30:
        vs_ma20 = (latest.close / w30["avg_close"] - 1.0) if w30["avg_close"] else 0.0
        lines.append(
            f"3. 最近 30 天（{w30['from_date']} → {w30['to_date']}，{w30['sessions']} 个交易日）："
            f"累计 {_fmt_pct(w30['return'])}，区间高 {w30['high']:.2f} / 低 {w30['low']:.2f}，"
            f"均价 {w30['avg_close']:.2f}（现价偏离 {_fmt_pct(vs_ma20)}），处于区间 {w30['range_pos'] * 100:.0f}% 分位"
        )
    if w90:
        lines.append(
            f"4. 最近 3 个月（{w90['from_date']} → {w90['to_date']}，{w90['sessions']} 个交易日）："
            f"累计 {_fmt_pct(w90['return'])}，区间高 {w90['high']:.2f} / 低 {w90['low']:.2f}，"
            f"处于 3 个月区间 {w90['range_pos'] * 100:.0f}% 分位"
        )

    return {
        "market": mkt,
        "symbol": symbol,
        "label": label,
        "as_of": latest.date.isoformat(),
        "text": "\n".join(lines),
    }


def extract_stance(text: str) -> str:
    m = STANCE_RE.search(text or "")
    if m:
        raw = m.group(1).upper()
        if raw in ("买入", "BUY"):
            return "BUY"
        if raw in ("卖出", "SELL"):
            return "SELL"
        if raw in ("观望", "HOLD"):
            return "HOLD"
    # 兜底检测首行倾向
    head = (text or "")[:60]
    if "偏向买入" in head or "建议买入" in head:
        return "BUY"
    if "偏向卖出" in head or "建议卖出" in head or "减仓" in head:
        return "SELL"
    if "观望" in head:
        return "HOLD"
    return ""


def _resolve_models(root: Path) -> tuple[str, str]:
    env = merged_env(root)
    deepseek_model = env.get("DEEPSEEK_MODEL") or "deepseek-chat"
    qwen_model = env.get("QWEN_MODEL") or "qwen3.8-flash"
    return deepseek_model, qwen_model


def _build_llm_messages(
    role: str,
    round_idx: int,
    total_rounds: int,
    history: list[dict],
) -> list[dict]:
    my_name = "DeepSeek" if role == "deepseek" else "Qwen"
    peer_name = "Qwen" if role == "deepseek" else "DeepSeek"
    persona = (
        "你注重中期趋势结构、30天/3个月分位与盈亏比纪律，敢于指出追高或抄底陷阱。"
        if role == "deepseek"
        else "你注重短线动量（当前与近5天节奏）、均值回归与仓位时机，善于捕捉转折信号与反驳盲点。"
    )
    is_final = round_idx == total_rounds

    system_prompt = (
        f"你是量化交易研讨群中的 {my_name}，正在与另一位 AI 分析师 {peer_name} 及人类交易员共同群聊研判。\n"
        f"角色定位：{persona}\n"
        "讨论目标：结合【当前、最近5天、最近30天、最近3个月】行情上下文以及群内消息，真正评判今天操作上**买入**好还是**卖出**好（若风险收益极不对称也可判**观望**，但请优先给出明确倾向）。\n"
        "发言要求：\n"
        "1. 开头必须用 `【买入】`、`【卖出】` 或 `【观望】` 标明你本轮的明确立场。\n"
        f"2. 这是第 {round_idx}/{total_rounds} 轮发言：直接回应 {peer_name} 或用户的最新观点（赞同点、分歧点或补充证据），不要重复抄写原始数据，控制在 100～180 字以内，干脆利落像真实交易台群聊。"
        + ("\n3. 本轮是最后一轮，请给出你最终拍板的操作结论、触发条件或止损/止盈锚点。" if is_final else "")
    )

    transcript_lines = []
    for item in history:
        r = item["role"]
        if r == "context":
            transcript_lines.append(f"[盘面上下文]\n{item['content']}")
        elif r == "user":
            transcript_lines.append(f"[用户补充]：{item['content']}")
        elif r == "deepseek":
            transcript_lines.append(f"[DeepSeek]：{item['content']}")
        elif r == "qwen":
            transcript_lines.append(f"[Qwen]：{item['content']}")

    user_prompt = (
        "\n\n".join(transcript_lines)
        + f"\n\n请以 {my_name} 身份发表第 {round_idx}/{total_rounds} 轮群聊观点（开头带【买入】/【卖出】/【观望】）："
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_group_chat_stream(
    root: Path,
    market: str,
    thread_id: str | None = None,
    user_message: str = "",
    rounds: int = 3,
    llms: dict[str, LLM] | None = None,
) -> Iterator[dict]:
    """执行多模型轮流群聊并实时写入 SQLite，按事件流逐步产出字典。"""
    mkt = market if market in DEFAULT_INSTRUMENTS else "ndx"
    rounds = max(1, min(int(rounds or 3), 5))
    user_text = (user_message or "").strip()

    ctx = build_market_context(root, mkt)
    as_of = ctx["as_of"]

    is_new = not thread_id
    tid = thread_id or f"{mkt}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    existing = [] if is_new else get_chat_thread(root, tid)
    if not existing:
        is_new = True

    yield {
        "type": "thread",
        "thread_id": tid,
        "market": mkt,
        "as_of": as_of,
        "is_new": is_new,
    }

    if is_new:
        ctx_msg = append_chat_message(
            root,
            thread_id=tid,
            market=mkt,
            as_of=as_of,
            role="context",
            content=ctx["text"],
            model="system",
        )
        existing.append(ctx_msg)
        yield {"type": "message", "message": ctx_msg}

    if user_text:
        u_msg = append_chat_message(
            root,
            thread_id=tid,
            market=mkt,
            as_of=as_of,
            role="user",
            content=user_text,
            model="user",
        )
        existing.append(u_msg)
        yield {"type": "message", "message": u_msg}

    deepseek_model, qwen_model = _resolve_models(root)
    if llms is not None:
        ds_client = llms["deepseek"]
        qw_client = llms["qwen"]
    else:
        ds_client = providers.build(deepseek_model, root)
        qw_client = providers.build(qwen_model, root)

    speakers = (
        ("deepseek", deepseek_model, ds_client),
        ("qwen", qwen_model, qw_client),
    )
    total_steps = rounds * len(speakers)
    step = 0

    for r_idx in range(1, rounds + 1):
        for role, model_name, client in speakers:
            step += 1
            yield {
                "type": "thinking",
                "role": role,
                "model": model_name,
                "round": r_idx,
                "rounds": rounds,
                "step": step,
                "total_steps": total_steps,
            }
            prompt_msgs = _build_llm_messages(role, r_idx, rounds, existing)
            reply = client.chat(prompt_msgs, temperature=0.6, max_tokens=600)
            content = (reply.content or "").strip() or "【观望】（模型未返回有效文本）"
            stance = extract_stance(content)
            saved = append_chat_message(
                root,
                thread_id=tid,
                market=mkt,
                as_of=as_of,
                role=role,
                content=content,
                model=model_name,
                stance=stance,
            )
            existing.append(saved)
            yield {"type": "message", "message": saved}

    yield {"type": "done", "thread_id": tid, "total_messages": len(existing)}
