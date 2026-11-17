"""A/B 新闻解读：两个模型读同一批标题，只比这一件事。

为什么单独一个文件：现有的 `news_labels.py` 只认一个模型，把结果写进生产目录。
要做对比就必须同时留住两家的输出、并且证明**输入完全一样**。混在一起跑，事后没人
能说清差异是模型带来的还是题带来的。

考卷是历史某一天的 25 条标题，模型只做一件事：把标题读成 `regime / severity /
novelty / events`。同一份 SYSTEM 提示、同一个温度、同一个 25 条上限、同一天。

打分分三层，写死在 `PROTOCOL` 里，跑之前就定下来，不含事后挑选：

1. **一致性**：regime 一致率、Cohen's kappa（扣掉碰巧一致的成分）；severity 的
   平均绝对差和 Spearman 秩相关；事件类型的 Jaccard 重合度。
2. **决策影响**：按生产阈值（severity ≥ 0.6 = 强利空 → 撤掉多头信号），两家各自
   会撤掉哪些天，只有一家想撤的天有多少。
3. **下游用处**：severity 与「之后 5 个交易日涨跌幅绝对值」的相关（幅度，不是方向），
   以及 regime=risk_off 之后 5 日的上涨率。这两项只报数，**不作为选谁的标准**——
   方向那一层在 README 里已经被判过不显著，再用它选模型就是拿噪声当证据。
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from forecast_log.agent.llm import LLM
from forecast_log.bars import load_series
from forecast_log.news_labels import (
    BATCH_DAYS,
    MAX_TITLES_PER_DAY,
    PROMPT_VERSION,
    label_batch,
    read_day_titles,
)

PROTOCOL = {
    "prompt_version": PROMPT_VERSION,
    "temperature": 0.1,
    "max_tokens": 8000,
    "max_titles_per_day": MAX_TITLES_PER_DAY,
    "batch_days": BATCH_DAYS,
    "veto_threshold": 0.6,
    "sample": "每个自然年等量抽取，随机种子写进报告",
    "scored_on": "两家都成功标出标签的那些天（交集）",
    "decision_metric": "一致性与决策影响为主；下游相关性只报数，不参与选择",
}


def ab_dir(root: Path, slug: str | None = None) -> Path:
    base = root / "artifacts" / "news_ab"
    return base if slug is None else base / slug


def slug_of(model: str) -> str:
    return model.replace("/", "_").replace(" ", "")


def sample_days(root: Path, per_year: int = 30, seed: int = 7) -> list[date]:
    """按年分层抽样。分层是为了避免 2020 那种单年占据大半样本。"""
    by_year: dict[str, list[date]] = {}
    for path in sorted((root / "data" / "news").glob("????-??-??.json")):
        try:
            day = date.fromisoformat(path.stem)
        except ValueError:
            continue
        by_year.setdefault(str(day.year), []).append(day)
    rng = random.Random(seed)
    picked: list[date] = []
    for year in sorted(by_year):
        pool = by_year[year]
        rng.shuffle(pool)
        picked.extend(sorted(pool[:per_year]))
    return sorted(picked)


def load_labeler(root: Path, slug: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in ab_dir(root, slug).glob("????-??-??.json"):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out[record["as_of"]] = record
    return out


def label(
    root: Path,
    days: list[date],
    model: str,
    llm: LLM,
    *,
    workers: int = 4,
    limit_denominator: int | None = None,
) -> dict:
    """标这批天，一天一个文件。已有文件的跳过，中断重跑只补缺的。"""
    slug = slug_of(model)
    target = ab_dir(root, slug)
    target.mkdir(parents=True, exist_ok=True)
    todo = [day for day in days if not (target / f"{day.isoformat()}.json").is_file()]
    stats = {"model": model, "todo": len(todo), "days": 0, "failed": 0, "cost_usd": 0.0, "errors": []}
    if not todo:
        return stats
    titles = read_day_titles(root, todo)
    keys = list(titles)
    chunks = [
        {key: titles[key] for key in keys[start : start + BATCH_DAYS]}
        for start in range(0, len(keys), BATCH_DAYS)
    ]

    from concurrent.futures import ThreadPoolExecutor

    def one(chunk: dict[str, list[str]]):
        """一批解析失败就劈成两半重来。

        10 天 × 3 个事件 + note 很容易超过 8000 token，截断的 JSON 解析不了。劈半比
        调大预算更稳：它同时也解决了「模型偶尔多写一行」这种不确定的失败。
        """
        try:
            return label_batch(root, chunk, llm), []
        except Exception as error:
            if len(chunk) > 1:
                keys = list(chunk)
                middle = len(keys) // 2
                halves = (
                    {key: chunk[key] for key in keys[:middle]},
                    {key: chunk[key] for key in keys[middle:]},
                )
                labeled: list[dict] = []
                usage: dict = {}
                errors: list[str] = []
                for half in halves:
                    try:
                        entries, used = label_batch(root, half, llm)
                    except Exception as inner:
                        errors.append(f"{min(half)}..: {type(inner).__name__}: {inner}")
                        continue
                    labeled.extend(entries)
                    usage = used or usage
                return (labeled, usage), errors
            return ([], {}), [f"{min(chunk)}..: {type(error).__name__}: {error}"]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for chunk, ((labeled, usage), errors) in zip(chunks, pool.map(one, chunks)):
            stats["errors"].extend(errors)
            stats["failed"] += sum(1 for _ in errors)
            if not labeled:
                continue
            stats["cost_usd"] += usage.get("cost_usd", 0.0)
            for entry in labeled:
                record = {
                    "as_of": entry["as_of"],
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "prompt_version": PROMPT_VERSION,
                    "model": model,
                    "label": {key: value for key, value in entry.items() if key != "as_of"},
                    "headlines_used": len(chunk.get(entry["as_of"], [])),
                    "usage": usage,
                }
                (target / f"{entry['as_of']}.json").write_text(json.dumps(record, ensure_ascii=False))
                stats["days"] += 1
    return stats


def _ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    expected = sum((a.count(k) / len(a)) * (b.count(k) / len(b)) for k in labels)
    return 0.0 if expected == 1 else (observed - expected) / (1 - expected)


def _forward_moves(root: Path, days: list[str], horizon: int = 5) -> dict[str, float]:
    try:
        series = load_series(root, "^NDX")
    except FileNotFoundError:  # 没有行情也能比一致性，只是不出下游那两列
        return {}
    ordered = [str(bar.date) for bar in series.bars]
    closes = {str(bar.date): bar.close for bar in series.bars}
    index = {day: position for position, day in enumerate(ordered)}
    out: dict[str, float] = {}
    for day in days:
        position = index.get(day)
        if position is None or position + horizon >= len(ordered):
            continue
        out[day] = closes[ordered[position + horizon]] / closes[day] - 1
    return out


def compare(root: Path, slugs: list[str], *, horizon: int = 5) -> dict:
    loaded = {slug: load_labeler(root, slug) for slug in slugs}
    shared = sorted(set.intersection(*(set(rows) for rows in loaded.values()))) if loaded else []
    moves = _forward_moves(root, shared, horizon)
    threshold = PROTOCOL["veto_threshold"]

    per_labeler: dict[str, dict] = {}
    for slug, rows in loaded.items():
        days = [day for day in shared if day in rows]
        severity = [rows[day]["label"]["severity"] for day in days]
        regime = [rows[day]["label"]["regime"] for day in days]
        per_labeler[slug] = {
            "model": next((rows[day]["model"] for day in days), slug),
            "days": len(days),
            "severity_mean": round(float(np.mean(severity)), 3) if severity else None,
            "severity_ge_0.6": sum(1 for value in severity if value >= threshold),
            "risk_off": sum(1 for value in regime if value == "risk_off"),
            "cost_usd": round(sum(rows[day].get("usage", {}).get("cost_usd", 0.0) for day in days), 4),
            "per_day_usd": round(
                float(np.mean([rows[day].get("usage", {}).get("cost_usd", 0.0) for day in days])), 6
            )
            if days
            else None,
            "latency_p50_s": round(
                float(np.median([rows[day].get("usage", {}).get("latency_s", 0.0) for day in days])), 2
            )
            if days
            else None,
        }
        pairs = [(rows[day]["label"]["severity"], abs(moves[day])) for day in days if day in moves]
        if len(pairs) > 2:
            sev = [pair[0] for pair in pairs]
            mag = [pair[1] for pair in pairs]
            per_labeler[slug]["sev_vs_abs_move"] = round(_spearman(sev, mag), 3)
            up = [1.0 if moves[day] > 0 else 0.0 for day in days if day in moves and rows[day]["label"]["regime"] == "risk_off"]
            per_labeler[slug]["risk_off_up_rate"] = round(float(np.mean(up)), 3) if up else None
            per_labeler[slug]["risk_off_days_with_move"] = len(up)

    pairs_out: dict[str, dict] = {}
    for i, left in enumerate(slugs):
        for right in slugs[i + 1 :]:
            days = [day for day in shared if day in loaded[left] and day in loaded[right]]
            if not days:
                continue
            a = [loaded[left][day]["label"] for day in days]
            b = [loaded[right][day]["label"] for day in days]
            regime_a = [row["regime"] for row in a]
            regime_b = [row["regime"] for row in b]
            sev_a = [row["severity"] for row in a]
            sev_b = [row["severity"] for row in b]
            veto_a = {day for day, row in zip(days, a) if row["severity"] >= threshold}
            veto_b = {day for day, row in zip(days, b) if row["severity"] >= threshold}
            jaccard = [
                len(set(x["type"] for x in row_a["events"]) & set(x["type"] for x in row_b["events"]))
                / max(1, len(set(x["type"] for x in row_a["events"]) | set(x["type"] for x in row_b["events"])))
                for row_a, row_b in zip(a, b)
            ]
            pairs_out[f"{left} vs {right}"] = {
                "days": len(days),
                "regime_agreement": round(sum(1 for x, y in zip(regime_a, regime_b) if x == y) / len(days), 3),
                "kappa": round(_kappa(regime_a, regime_b), 3),
                "severity_mae": round(float(np.mean([abs(x - y) for x, y in zip(sev_a, sev_b)])), 3),
                "severity_spearman": round(_spearman(sev_a, sev_b), 3),
                "event_type_jaccard": round(float(np.mean(jaccard)), 3),
                "veto_same_days": len(veto_a & veto_b),
                "veto_only_left": len(veto_a - veto_b),
                "veto_only_right": len(veto_b - veto_a),
                "veto_union": len(veto_a | veto_b),
            }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": PROTOCOL,
        "slugs": slugs,
        "shared_days": len(shared),
        "first_day": shared[0] if shared else None,
        "last_day": shared[-1] if shared else None,
        "per_labeler": per_labeler,
        "pairs": pairs_out,
    }
    out = ab_dir(root) / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out.parent / "report.md").write_text(render(report) + "\n")
    return report


def render(report: dict) -> str:
    if not report["shared_days"]:
        return "没有两家都标上的日子。"
    lines = [
        f"{report['shared_days']} 天（{report['first_day']} → {report['last_day']}），"
        f"提示词 {report['protocol']['prompt_version']}，温度 {report['protocol']['temperature']}",
        "",
        "| 标签者 | 天数 | severity 均值 | ≥0.6 的天 | risk_off 的天 | $/天 | p50 秒 | severity↔5日涨跌绝对值 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for slug, row in report["per_labeler"].items():
        lines.append(
            f"| {row['model']} | {row['days']} | {row['severity_mean']} | {row['severity_ge_0.6']} | "
            f"{row['risk_off']} | {row['per_day_usd']} | {row['latency_p50_s']} | "
            f"{row.get('sev_vs_abs_move', '—')} |"
        )
    lines += ["", "| 对比 | 天数 | regime 一致率 | kappa | severity 平均差 | severity 秩相关 | 事件类型重合 | 只有左撤 | 只有右撤 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, row in report["pairs"].items():
        lines.append(
            f"| {name} | {row['days']} | {row['regime_agreement']} | {row['kappa']} | {row['severity_mae']} | "
            f"{row['severity_spearman']} | {row['event_type_jaccard']} | {row['veto_only_left']} | {row['veto_only_right']} |"
        )
    return "\n".join(lines)

def command(root: Path, *, per_year: int = 30, seed: int = 7, models: str = "deepseek-flash,qwen3.8-flash",
            workers: int = 4, sample_only: bool = False, label_only: bool = False) -> dict:
    """`forecast-log news-ab` 的入口。两家（或更多）读同一批天，然后只比这一件事。"""
    names = [name.strip() for name in models.split(",") if name.strip()]
    if len(names) < 2 and not label_only:
        raise SystemExit("至少给两个模型，否则没有可比的（只标一方请加 --label-only）")
    days = sample_days(root, per_year=per_year, seed=seed)
    print(f"抽样：{len(days)} 天，{days[0]} → {days[-1]}（每年最多 {per_year} 天，种子 {seed}）")
    if sample_only:
        for day in days:
            print(day.isoformat())
        return {"days": [day.isoformat() for day in days]}

    from forecast_log import providers

    for name in names:
        llm = providers.build(name, root)
        stats = label(root, days, name, llm, workers=workers)
        print(
            f"  {name}: 新标 {stats['days']} 天，失败 {stats['failed']}，"
            f"花费 ${stats['cost_usd']:.4f}"
        )
        for error in stats["errors"][:3]:
            print(f"    {error}")

    if label_only:
        return {"labels": {name: slug_of(name) for name in names}}

    report = compare(root, [slug_of(name) for name in names])
    print()
    print(render(report))
    print(f"\n完整结果写在 {ab_dir(root) / 'report.json'}")
    return report
