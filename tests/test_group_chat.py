from datetime import date, timedelta

from forecast_log.agent.llm import Reply, ScriptedLLM
from forecast_log.bars import Bar, Series
from forecast_log.db import get_chat_thread, list_chat_threads, upsert_series
from forecast_log.group_chat import build_market_context, run_group_chat_stream


def _seed_bars(tmp_path, symbol="^NDX", days=75):
    start = date(2026, 8, 1)
    bars = []
    for i in range(days):
        d = start + timedelta(days=i)
        c = 20000.0 + i * 25.0
        bars.append(Bar(d, c, open=c - 15.0, high=c + 30.0, low=c - 40.0, volume=1000 + i, is_closed=True))
    upsert_series(tmp_path, Series(symbol, "America/New_York", "2026-11-16T22:00:00Z", tuple(bars)))


def test_build_market_context_includes_four_windows(tmp_path):
    _seed_bars(tmp_path, "^NDX", days=75)
    ctx = build_market_context(tmp_path, "ndx")
    assert ctx["market"] == "ndx"
    assert ctx["symbol"] == "^NDX"
    text = ctx["text"]
    assert "1. 当前盘面" in text
    assert "2. 最近 5 天" in text
    assert "3. 最近 30 天" in text
    assert "4. 最近 3 个月" in text


def test_group_chat_three_rounds_and_user_followup_saved_to_sqlite(tmp_path):
    _seed_bars(tmp_path, "^NDX", days=75)
    ds = ScriptedLLM(
        [
            Reply(content="【买入】趋势稳步向上，30天与3个月均处于高位强势区，回调即是买点。"),
            Reply(content="【买入】同意短线有超买，但未破5日低点前不宜盲目做空，维持偏多。"),
            Reply(content="【买入】最终结论：逢盘中回踩分批买入，跌破近5日低点止损。"),
            Reply(content="【观望】结合用户补充的短线止盈约束，当前追高空间有限，改为等回踩再买。"),
        ],
        name="deepseek-chat",
    )
    qw = ScriptedLLM(
        [
            Reply(content="【观望】近5天连涨斜率偏陡，短线追高盈亏比一般，建议观察开盘承接。"),
            Reply(content="【买入】若以5日均价做防守锚点，轻仓顺势试多可以接受。"),
            Reply(content="【买入】最终结论：轻仓买入，盈利3%附近分批止盈。"),
            Reply(content="【卖出】若已有仓位且目标仅3%，当前高位建议先锁利卖出半仓。"),
        ],
        name="qwen3.8-flash",
    )

    # 第一阶段：默认 3 轮/人 = 6 条 AI 消息 + 1 条盘面上下文 = 7 条入库消息
    events = list(
        run_group_chat_stream(
            tmp_path,
            market="ndx",
            rounds=3,
            llms={"deepseek": ds, "qwen": qw},
        )
    )
    tid = events[0]["thread_id"]
    assert events[-1]["type"] == "done"

    saved = get_chat_thread(tmp_path, tid)
    assert len(saved) == 7
    assert [m["role"] for m in saved] == [
        "context",
        "deepseek",
        "qwen",
        "deepseek",
        "qwen",
        "deepseek",
        "qwen",
    ]
    assert saved[1]["stance"] == "BUY"
    assert saved[2]["stance"] == "HOLD"
    assert saved[6]["stance"] == "BUY"

    # 第二阶段：用户在群聊中补充信息并继续 1 轮/人（+1 user +2 AI = 共 10 条消息）
    followup_events = list(
        run_group_chat_stream(
            tmp_path,
            market="ndx",
            thread_id=tid,
            user_message="我目前半仓，只做一周内的短线（3%止盈），今天要减仓吗？",
            rounds=1,
            llms={"deepseek": ds, "qwen": qw},
        )
    )
    assert followup_events[-1]["type"] == "done"

    saved_after = get_chat_thread(tmp_path, tid)
    assert len(saved_after) == 10
    assert [m["role"] for m in saved_after[7:]] == ["user", "deepseek", "qwen"]
    assert "半仓" in saved_after[7]["content"]
    assert saved_after[8]["stance"] == "HOLD"
    assert saved_after[9]["stance"] == "SELL"

    threads = list_chat_threads(tmp_path, "ndx")
    assert len(threads) == 1
    assert threads[0]["thread_id"] == tid
    assert threads[0]["message_count"] == 10
