"""新闻关键字能不能当信号？先量，再决定要不要进模型。

做法和 CPI / NFP 那两个窗口一样：拿一个条件（当天新闻里出现某个词），看 5 个交易日
后的 NDX 上涨率，和同一段的基率比，再分两段看方向是否一致。这一页只打印表，
不写页面、不改 `cross.py`——过不了这张表的东西不该进信号。

用法：

    .venv/bin/python experiments/news_keywords.py [--holdout 2015-01-01]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_log.bars import load_series, Series  # noqa: E402
from forecast_log.paths import repo_root  # noqa: E402

HORIZON = 5

from forecast_log.keywords import COMPILED, LEXICON, RISK_OFF, RISK_ON  # noqa: E402


def keyword_days(root: Path, floor: date) -> dict[str, set[str]]:
    """每个关键字在哪些交易日出现过（按 `published_at` 归属那天）。"""
    hits: dict[str, set[str]] = defaultdict(set)
    for path in sorted((root / "data" / "news").glob("????-??-??.json")):
        day = path.stem
        if date.fromisoformat(day) < floor:
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        text = " ".join(document.get("title", "") for document in record.get("documents", []))
        if not text:
            continue
        for name, pattern in COMPILED.items():
            if pattern.search(text):
                hits[name].add(day)
    return hits


def forward(series: Series) -> dict[str, int]:
    """每个交易日 5 个交易日后是否更高，和 `cross.py` 的目标一致。"""
    bars = series.bars
    out: dict[str, int] = {}
    for index in range(len(bars) - HORIZON):
        out[bars[index].date.isoformat()] = int(bars[index + HORIZON].close > bars[index].close)
    return out


def rates(days: set[str], outcomes: dict[str, int], floor: str | None = None, ceiling: str | None = None):
    pool = [
        outcomes[day]
        for day in days
        if day in outcomes and (floor is None or day >= floor) and (ceiling is None or day < ceiling)
    ]
    if not pool:
        return 0, float("nan")
    return len(pool), sum(pool) / len(pool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", default="2015-01-01")
    parser.add_argument("--symbol", default="^NDX")
    args = parser.parse_args()

    root = repo_root()
    series = load_series(root, args.symbol)
    outcomes = forward(series)
    hits = keyword_days(root, date(2013, 1, 1))

    all_days = set(outcomes) & {path.stem for path in (root / "data" / "news").glob("????-??-??.json")}
    base_n, base_all = rates(all_days, outcomes)
    _, base_pre = rates(all_days, outcomes, ceiling=args.holdout)
    _, base_post = rates(all_days, outcomes, floor=args.holdout)
    print(f"{len(all_days)} 个交易日有新闻（{min(all_days)} → {max(all_days)}）")
    print(f"基率：全部 {base_all:.1%}（n={base_n}）｜{args.holdout} 前 {base_pre:.1%}｜之后 {base_post:.1%}\n")
    print("| 关键字 | 天数 | 全样本 | 前段 | 后段 | 对基率 |")
    print("|---|---|---|---|---|---|")

    rows = []
    for name in LEXICON:
        days = hits.get(name, set())
        n, hit = rates(days, outcomes)
        _, pre = rates(days, outcomes, ceiling=args.holdout)
        n_post, post = rates(days, outcomes, floor=args.holdout)
        if n < 30:
            continue
        edge = hit - base_all
        consistent = (pre - base_pre) > 0 and (post - base_post) > 0
        rows.append(
            {
                "keyword": name,
                "n": n,
                "hit": hit,
                "pre": pre,
                "post": post,
                "edge": edge,
                "consistent": consistent,
                "n_post": n_post,
            }
        )

    rows.sort(key=lambda row: row["edge"], reverse=True)
    for row in rows:
        flag = " ✅" if row["consistent"] else ""
        print(
            f"| {row['keyword']} | {row['n']} | {row['hit']:.1%} | {row['pre']:.1%} | "
            f"{row['post']:.1%} | {row['edge']:+.1%}{flag} |"
        )

    survivors = [row for row in rows if row["consistent"] and row["edge"] > 0.02]
    print(f"\n两段都高于基率、且超额 > 2 个点的：{[row['keyword'] for row in survivors] or '没有'}")

    out = root / "artifacts" / "news_keywords.json"
    out.write_text(
        json.dumps(
            {
                "symbol": args.symbol,
                "horizon": HORIZON,
                "holdout": args.holdout,
                "base": {"all": base_all, "pre": base_pre, "post": base_post},
                "rows": rows,
                "survivors": [row["keyword"] for row in survivors],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"写在 {out}")
    second_layer(root, outcomes, args.holdout)
    veto_test(root, args.holdout)




# ------------------------------------------------------------------ 第二层
#
# 单个词是二值的、太稀疏：一个词出现一次和出现二十次算同一件事。所以第二层做三件事：
# 1. 把词按方向收成两个桶（risk-on / risk-off），算当天的净调子（tone）；
# 2. 用当天的条目数把 tone 归一化，避免「大新闻日条目多」冒充「调子强」；
# 3. 按 tone 分五档看 5 日上涨率——连续量只有分档才看得出单调性。

def daily_features(root: Path, floor: date) -> dict[str, dict]:
    """每天的：条目数、正负桶计数、净调子。"""
    out: dict[str, dict] = {}
    for path in sorted((root / "data" / "news").glob("????-??-??.json")):
        day = path.stem
        if date.fromisoformat(day) < floor:
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        titles = [document.get("title", "") for document in record.get("documents", [])]
        if not titles:
            continue
        text = " ".join(titles)
        off = sum(1 for name in RISK_OFF if COMPILED[name].search(text))
        on = sum(1 for name in RISK_ON if COMPILED[name].search(text))
        out[day] = {
            "items": len(titles),
            "off": off,
            "on": on,
            "tone": (on - off) / max(len(titles), 1),
        }
    return out


def quintiles(features: dict[str, dict], outcomes: dict[str, int], key: str, holdout: str) -> list[dict]:
    values = sorted(
        (feature[key], day) for day, feature in features.items() if day in outcomes
    )
    if len(values) < 50:
        return []
    cuts = [values[int(len(values) * fraction)][0] for fraction in (0.2, 0.4, 0.6, 0.8)]
    buckets: list[list[str]] = [[] for _ in range(5)]
    for value, day in values:
        index = sum(1 for cut in cuts if value >= cut)
        buckets[min(index, 4)].append(day)
    rows = []
    for index, days in enumerate(buckets, start=1):
        n, hit = rates(set(days), outcomes)
        _, pre = rates(set(days), outcomes, ceiling=holdout)
        _, post = rates(set(days), outcomes, floor=holdout)
        rows.append({"bucket": index, "n": n, "hit": hit, "pre": pre, "post": post})
    return rows


def second_layer(root: Path, outcomes: dict[str, int], holdout: str) -> None:
    features = daily_features(root, date(2013, 1, 1))
    print("\n## 第二层：把词收成净调子，再分五档\n")
    for key, label in (("tone", "净调子 (risk-on − risk-off) / 条目数"), ("items", "当天条目数")):
        rows = quintiles(features, outcomes, key, holdout)
        if not rows:
            continue
        print(f"{label}：")
        print("| 档 | 天数 | 全样本 | 前段 | 后段 |")
        print("|---|---|---|---|---|")
        for row in rows:
            print(f"| {row['bucket']} | {row['n']} | {row['hit']:.1%} | {row['pre']:.1%} | {row['post']:.1%} |")
        print()




# ------------------------------------------------------------------ 第三层
#
# 单看新闻没方向，那就换个问法：新闻能不能当**信号的否决票**？
# 这正是 FOMC 的用法（只在决议当天否决，不加分）。这里拿 cross.json 已经打出的
# 信号，按当天新闻调子分两半，看精确率差多少。差得多才配进 cross.py。


def veto_test(root: Path, holdout: str) -> None:
    features = daily_features(root, date(2013, 1, 1))
    cross = json.loads((root / "artifacts" / "cross.json").read_text())
    rows = []
    for day, flag, outcome in zip(cross["time"], cross["flag"], cross["outcome"]):
        if not flag or outcome is None or day not in features:
            continue
        rows.append((day, outcome, features[day]["tone"], features[day]["off"]))
    if len(rows) < 100:
        print("\n信号日太少，跳过否决测试")
        return

    print(f"\n## 第三层：新闻能不能否决信号？（{len(rows)} 个信号日有新闻，{holdout} 前/后分开看）\n")
    for label, pick in (
        ("净调子为负（risk-off 词多于 risk-on 词）", lambda row: row[2] < 0),
        ("净调子为正", lambda row: row[2] > 0),
        ("净调子为 0", lambda row: row[2] == 0),
        ("risk-off 词命中 ≥ 2 个", lambda row: row[3] >= 2),
    ):
        group = [row for row in rows if pick(row)]
        if not group:
            continue
        pre = [row[1] for row in group if row[0] < holdout]
        post = [row[1] for row in group if row[0] >= holdout]
        print(
            f"- {label}: n={len(group)} 精确率 {sum(r[1] for r in group) / len(group):.1%}"
            f"（前 {len(pre)} 个 {sum(pre) / len(pre):.1%} / 后 {len(post)} 个 {sum(post) / len(post):.1%}）"
            if pre and post
            else f"- {label}: n={len(group)} 精确率 {sum(r[1] for r in group) / len(group):.1%}"
        )
    all_hit = sum(row[1] for row in rows) / len(rows)
    print(f"- 全部信号日: n={len(rows)} 精确率 {all_hit:.1%} ← 拿这个当对照")


if __name__ == "__main__":
    main()
