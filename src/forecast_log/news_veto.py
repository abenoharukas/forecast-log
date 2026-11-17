"""新闻否决：四个取值，其它条件全部固定。

开关矩阵里唯一缺的一维。`cross.py` 的规则把「新闻否决」写死成一个模型，这里把它
拆成 4 个互斥取值，一次只改这一件事：

    off          不用新闻否决
    <模型 A>     只信 A 的 severity
    <模型 B>     只信 B 的 severity
    both         两家都判强利空才撤（交集）

被固定的：信号集（`cross.json` 的 `flag_price_only`，即价格侧原值）、阈值 0.6、
判定口径（第 5 个交易日收盘是否高于基准收盘）、样本（两家都有标签的那些天）。
产物单独一份：`artifacts/news_veto/report.json`，不写进 `cross.json`。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from forecast_log.news_ab import load_labeler


def variants(root: Path, slugs: list[str], threshold: float = 0.6) -> dict[str, dict]:
    from forecast_log.artifacts_db import load_cross_market

    cross = load_cross_market(root, "ndx")
    if cross is None:
        raise FileNotFoundError("数据库中缺少 ndx 信号，请先运行 forecast-log cross")
    time = cross["time"]
    price_flag = dict(zip(time, cross["flag_price_only"]))
    outcome = dict(zip(time, cross["outcome"]))
    labels = {slug: load_labeler(root, slug) for slug in slugs}
    shared = sorted(set.intersection(*(set(rows) for rows in labels.values())))

    def veto_days(slug: str) -> set[str]:
        return {day for day in shared if labels[slug][day]["label"]["severity"] >= threshold}

    sets = {slug: veto_days(slug) for slug in slugs}
    plan: dict[str, set[str]] = {"off": set()}
    for slug in slugs:
        plan[slug] = sets[slug]
    plan["both"] = set.intersection(*sets.values()) if sets else set()

    base_signals = [day for day in shared if price_flag.get(day) == 1]
    out: dict[str, dict] = {}
    for name, veto in plan.items():
        kept = [day for day in base_signals if day not in veto]
        hits = [day for day in kept if outcome.get(day) == 1]
        out[name] = {
            "signals": len(base_signals),
            "vetoed": len(base_signals) - len(kept),
            "kept": len(kept),
            "hits": len(hits),
            "precision": round(len(hits) / len(kept), 4) if kept else None,
            "vetoed_days_would_have_hit": len([day for day in base_signals if day in veto and outcome.get(day) == 1]),
        }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": "新闻否决用哪一路：off / 单模型 / 两家都同意？其它条件固定",
        "threshold": threshold,
        "shared_days": len(shared),
        "first_day": shared[0] if shared else None,
        "last_day": shared[-1] if shared else None,
        "base_signals": len(base_signals),
        "variants": out,
    }
    days = [
        {
            "date": day,
            "price_signal": price_flag.get(day),
            "outcome": outcome.get(day),
        }
        | {slug: round(labels[slug][day]["label"]["severity"], 3) for slug in slugs}
        for day in shared
        if price_flag.get(day) is not None
    ]
    target = root / "artifacts" / "news_veto"
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (target / "report.md").write_text(render(report) + "\n")
    (target / "days.json").write_text(json.dumps({"threshold": threshold, "days": days}, ensure_ascii=False) + "\n")
    return report


def render(report: dict) -> str:
    lines = [
        f"{report['shared_days']} 天（{report['first_day']} → {report['last_day']}），"
        f"原始信号 {report['base_signals']} 个，阈值 {report['threshold']}",
        "",
        "| 新闻否决 | 撤掉 | 留下 | 命中 | 精确率 | 被撤掉的里面本来会命中的 |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in report["variants"].items():
        precision = "—" if row["precision"] is None else f"{row['precision']:.1%}"
        lines.append(
            f"| {name} | {row['vetoed']} | {row['kept']} | {row['hits']} | {precision} | "
            f"{row['vetoed_days_would_have_hit']} |"
        )
    return "\n".join(lines)


def command(root: Path, *, models: str = "deepseek-flash,qwen3.8-flash", threshold: float = 0.6) -> dict:
    slugs = [name.strip() for name in models.split(",") if name.strip()]
    if len(slugs) < 2:
        raise SystemExit("给两个模型：--models A,B")
    report = variants(root, slugs, threshold=threshold)
    print(render(report))
    print(f"\n完整结果写在 {root / 'artifacts' / 'news_veto' / 'report.json'}")
    return report
