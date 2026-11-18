from __future__ import annotations

import argparse
import json
from datetime import date

from forecast_log.agent_cli import agent_eval_command, agent_forecast_command, agent_report_command
from forecast_log.commands import (
    backtest_command,
    cross_command,
    fetch_command,
    lock_command,
    resolve_command,
    score_command,
)
from forecast_log import daily, events, wiki_news
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
    sub.add_parser("cross", help="跨资产风险偏好偏多时，之后 5 个交易日内的偏涨点")
    serve_parser = sub.add_parser("serve", help="起本地看板（默认 http://127.0.0.1:8731/）")
    serve_parser.add_argument("port", nargs="?", type=int, default=None, help="端口，默认 8731")
    sub.add_parser("events", help="让 LLM 拉 FOMC / CPI / NFP 日历（data/events.json）")
    daily_parser = sub.add_parser("daily", help="收盘后跑一次：抓当天新闻，LLM 给当天风险偏好标签")
    daily_parser.add_argument("--day", default=None, help="as-of 日期，默认今天")
    daily_parser.add_argument("--no-cache", action="store_true", help="忽略当天的新闻缓存，重新抓")

    news_parser = sub.add_parser("news-backfill", help="补历史新闻（Wikipedia Portal:Current events）")
    news_parser.add_argument("--start", default="2013-01-01")
    news_parser.add_argument("--end", default=None, help="默认到最后一个交易日")
    news_parser.add_argument("--batch", type=int, default=40, help="一次请求取多少天")
    sub.add_parser("news-coverage", help="新闻覆盖率：按年、按交易日")

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
    bench.add_argument("--merge", action="store_true", help="把新模型并进上一次同题集的结果，做多模型对比")

    ab = sub.add_parser("news-ab", help="两家模型读同一批历史新闻，只比新闻解读能力")
    ab.add_argument("--per-year", type=int, default=30, help="每个自然年抽几天")
    ab.add_argument("--seed", type=int, default=7)
    ab.add_argument("--models", default="deepseek-flash,qwen3.8-flash", help="逗号分隔，至少两个")
    ab.add_argument("--workers", type=int, default=4)
    ab.add_argument("--sample-only", action="store_true", help="只抽样本、打印日期，不发请求")
    ab.add_argument("--label-only", action="store_true", help="只标注不比较（可只给一个模型，便于两家并行跑）")

    veto = sub.add_parser("news-veto", help="新闻否决用哪一路：off / 单模型 / 两家都同意")
    veto.add_argument("--models", default="deepseek-flash,qwen3.8-flash")
    veto.add_argument("--threshold", type=float, default=0.6, help="强利空的严重度门槛")

    report = sub.add_parser("agent-report", help="把 agent_eval.json 打成 markdown 表")
    report.add_argument("--path", default="artifacts/agent_eval.json")

    sim_p = sub.add_parser("sim", help="在显式市场状态方程中运行异质性多主体政策冲击仿真")
    sim_p.add_argument(
        "--market",
        choices=("ndx", "ssec"),
        default="ndx",
        help="目标市场：ndx (^NDX / ^VXN) 或 ssec (A股上证综指 000001.SS / RV20)",
    )
    sim_p.add_argument(
        "--scenario",
        default=None,
        help="预设情景（如美股 tariff_25 / fomc_hawkish_25；A股 pboc_924_bazooka / huijin_stabilization / us_china_tariff_35）",
    )
    sim_p.add_argument("--day", default=None, help="基准交易日 YYYY-MM-DD，默认最新交易日")
    sim_p.add_argument("--steps", type=int, default=5, help="推演交易日步数，默认 5")
    sim_p.add_argument("--mode", choices=("rule", "homogeneous", "llm"), default="rule", help="角色决策模式")
    sim_p.add_argument("--models", default=None, help="LLM 模式下模型列表，逗号分隔（如 deepseek-chat,qwen3.8-flash）")
    sim_p.add_argument("--rate-bps", type=float, default=None, help="覆盖利率冲击（bps）")
    sim_p.add_argument("--fx-pct", type=float, default=None, help="覆盖美元指数外生冲击（百分点，%%）")
    sim_p.add_argument("--inflation-z", type=float, default=None, help="覆盖通胀意外（标准差）")
    sim_p.add_argument("--growth-z", type=float, default=None, help="覆盖增长/财政刺激意外（标准差）")
    sim_p.add_argument("--tariff-pct", type=float, default=None, help="覆盖关税税率变动（百分点）")
    sim_p.add_argument("--uncertainty-z", type=float, default=None, help="覆盖政策不确定性冲击（标准差）")
    sim_p.add_argument("--title", default=None, help="自定义情景标题")
    sim_p.add_argument("--out", default="artifacts/sim/latest_scenario.json", help="仿真结果输出路径")

    sim_bt = sub.add_parser("sim-backtest", help="在历史宏观事件上回测多主体仿真引擎（支持美股 ndx 与 A股 ssec）")
    sim_bt.add_argument("--market", choices=("ndx", "ssec"), default="ndx", help="回测市场：ndx 或 ssec")
    sim_bt.add_argument("--kinds", default="fomc,cpi,nfp", help="事件类型，逗号分隔")
    sim_bt.add_argument("--horizon", type=int, default=5, help="事件后推演窗口（交易日）")
    sim_bt.add_argument("--limit", type=int, default=None, help="均匀抽样多少次历史事件（便于快速跑 LLM 对照）")
    sim_bt.add_argument("--with-llm", action="store_true", help="同时跑真实 LLM 多主体对照")
    sim_bt.add_argument("--models", default=None, help="LLM 模型列表，逗号分隔")
    sim_bt.add_argument("--out", default=None, help="回测报告输出路径")

    args = parser.parse_args(argv)
    if args.command == "serve":
        from forecast_log.server import main as serve_main

        serve_main([str(args.port)] if args.port is not None else [])
        return
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
    elif args.command == "news-backfill":
        start = date.fromisoformat(args.start)
        days = [day for day in wiki_news.trading_days(root) if day >= start]
        if args.end:
            end = date.fromisoformat(args.end)
            days = [day for day in days if day <= end]
        print(f"{len(days)} 个交易日，{days[0]} → {days[-1]}")
        stats = wiki_news.backfill_batch(root, days, batch=args.batch)
        print(f"新补 {stats['ok']}，已有 {stats['already']}，空页 {stats['empty']}，失败 {stats['failed']}")
        for error in stats["errors"][:5]:
            print("  ", error)
        print(wiki_news.coverage(root, days))
    elif args.command == "news-coverage":
        days = [day for day in wiki_news.trading_days(root) if day >= date(2013, 1, 1)]
        report = wiki_news.coverage(root, days)
        print(json.dumps(report, ensure_ascii=False, indent=1))
    elif args.command == "daily":
        record = daily.run(
            root,
            day=date.fromisoformat(args.day) if args.day else None,
            use_cache=not args.no_cache,
        )
        label = record["label"]
        news = record["news"]
        print(
            f"{record['as_of']}  {record['model']}  {label['tone']} {label['confidence']:.2f}  "
            f"标题 {news['used']}/{news['retrieved']} 条（丢弃 {news['dropped_after_as_of']} 条越界）  "
            f"${record['usage']['cost_usd']:.5f}"
        )
        for driver in label["drivers"]:
            print("  -", driver)
        if record["contested"]:
            print("  与信号相左：页面会把它标成 contested")
        for error in news["errors"]:
            print("  抓取失败:", error)
        print("写在", daily.artifact_path(root))
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
    elif args.command == "news-ab":
        from forecast_log.news_ab import command as news_ab_command

        news_ab_command(
            root,
            per_year=args.per_year,
            seed=args.seed,
            models=args.models,
            workers=args.workers,
            sample_only=args.sample_only,
            label_only=args.label_only,
        )

    elif args.command == "news-veto":
        from forecast_log.news_veto import command as news_veto_command

        news_veto_command(root, models=args.models, threshold=args.threshold)

    elif args.command == "agent-report":
        agent_report_command(root, args.path)
    elif args.command == "sim":
        from forecast_log import sim

        sim.sim_command(
            root,
            market=args.market,
            scenario=args.scenario,
            day=args.day,
            steps=args.steps,
            mode=args.mode,
            models=args.models,
            rate_bps=args.rate_bps,
            fx_pct=args.fx_pct,
            inflation_z=args.inflation_z,
            growth_z=args.growth_z,
            tariff_pct=args.tariff_pct,
            uncertainty_z=args.uncertainty_z,
            title=args.title,
            out=args.out,
        )
    elif args.command == "sim-backtest":
        from forecast_log import sim

        sim.sim_backtest_command(
            root,
            market=args.market,
            kinds=args.kinds,
            horizon=args.horizon,
            limit=args.limit,
            with_llm=args.with_llm,
            models=args.models,
            out=args.out,
        )
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
            merge=args.merge,
        )


if __name__ == "__main__":
    main()
