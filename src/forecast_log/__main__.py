from __future__ import annotations

import argparse

from forecast_log.commands import (
    backtest_command,
    fetch_command,
    lock_command,
    resolve_command,
    score_command,
)
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


if __name__ == "__main__":
    main()
