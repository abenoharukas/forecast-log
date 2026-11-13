from __future__ import annotations

import argparse

from forecast_log.agent_cli import agent_eval_command, agent_forecast_command, agent_report_command
from forecast_log.commands import (
    backtest_command,
    cross_command,
    fetch_command,
    lock_command,
    resolve_command,
    score_command,
)
from forecast_log import events
from forecast_log.paths import repo_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="forecast-log",
        description="事先锁定概率，事后用 Brier 分数打分。",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="下载 ^VXN 和 ^NDX 的日线收盘")
    sub.add_parser("backtest", help="从过去每个交易日推演，再对照后来的收盘")
    sub.add_parser("lock", help="给三道题各写一次概率。写过就不能改")
    sub.add_parser("resolve", help="结算日收盘之后写入 resolutions.json")
    sub.add_parser("score", help="计算已结算题目的 Brier 分数")
    sub.add_parser("cross", help="跨资产风险偏好偏多时的 5 天上涨点")
    sub.add_parser("events", help="让 LLM 拉 FOMC / CPI / NFP 日历（data/events.json）")

    agent = sub.add_parser("agent-forecast", help="让 agent 给三道题写影子预测（不碰 forecasts/）")
    agent.add_argument("--runs", type=int, default=1, help="独立跑几次，log-odds 池化")
    agent.add_argument("--model", default=None, help="覆盖 FORECAST_LOG_MODEL")
    agent.add_argument("--temperature", type=float, default=0.7)
    agent.add_argument("--force", action="store_true", help="覆盖已有影子记录")

    bench = sub.add_parser("agent-eval", help="在已结算的历史题上跑评测台")
    bench.add_argument("--symbol", default="^VXN")
    bench.add_argument("--stride", type=int, default=20, help="每多少个交易日出一道题")
    bench.add_argument("--limit", type=int, default=12, help="最多几道题")
    bench.add_argument("--runs", type=int, default=1)
    bench.add_argument("--model", default=None)
    bench.add_argument("--blind", action="store_true", help="不给 agent 任何工具，做消融")
    bench.add_argument("--no-agent", action="store_true", help="只跑常数、基率、类比模型")
    bench.add_argument("--modes", default="level,quantile", help="题型，逗号分隔：level,quantile,band_up,band_down")
    bench.add_argument("--also-blind", action="store_true", help="再加一个不给工具的 agent，做消融")
    bench.add_argument("--out", default="artifacts/agent_eval.json")
    bench.add_argument("--workers", type=int, default=1, help="并发跑几道题")

    report = sub.add_parser("agent-report", help="把 agent_eval.json 打成 markdown 表")
    report.add_argument("--path", default="artifacts/agent_eval.json")

    args = parser.parse_args(argv)
    root = repo_root()
    if args.command == "fetch":
        fetch_command(root)
    elif args.command == "backtest":
        backtest_command(root)
    elif args.command == "lock":
        lock_command(root)
    elif args.command == "resolve":
        resolve_command(root)
    elif args.command == "score":
        score_command(root)
    elif args.command == "cross":
        cross_command(root)
    elif args.command == "events":
        record = events.fetch(root)
        print(
            f"{record['model']}  成本 ${record['usage']['cost_usd']:.4f}  "
            f"{record['counts']}\n校验：{record['problems'] or '无'}\n写在 {events.path_of(root)}"
        )
    elif args.command == "agent-forecast":
        agent_forecast_command(
            root, runs=args.runs, model=args.model, temperature=args.temperature, force=args.force
        )
    elif args.command == "agent-report":
        agent_report_command(root, args.path)
    elif args.command == "agent-eval":
        agent_eval_command(
            root,
            symbol=args.symbol,
            stride=args.stride,
            limit=args.limit,
            runs=args.runs,
            model=args.model,
            blind=args.blind,
            with_agent=not args.no_agent,
            also_blind=args.also_blind,
            modes=tuple(part.strip() for part in args.modes.split(",") if part.strip()),
            out=args.out,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
