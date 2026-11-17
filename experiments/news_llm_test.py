"""LLM 读出来的事件标签，到底有没有预测价值？

三种问法，都按「两段样本 + bootstrap 区间」判：

1. **方向**：`regime=risk_off` 的那些天，5 日后 NDX 上涨率是否低于基率。
2. **强度**：severity 分档后上涨率是否单调（单点差异不算，要单调）。
3. **否决**：在 `cross.json` 已打出的信号日里，高 severity / risk_off 的那批精确率是否更低。

只有「两段同向 + 区间不跨 0」才写进 `cross.py`。过不了就照旧只记账。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_log.bars import load_series  # noqa: E402
from forecast_log.news_labels import flatten, load  # noqa: E402
from forecast_log.paths import repo_root  # noqa: E402

HORIZON = 5
HOLDOUT = "2015-01-01"
random.seed(11)


def forward(series, horizon: int = HORIZON) -> dict[str, int]:
    bars = series.bars
    return {
        bars[index].date.isoformat(): int(bars[index + horizon].close > bars[index].close)
        for index in range(len(bars) - horizon)
    }


def rate(rows, key=lambda row: row[1]) -> float:
    return sum(key(row) for row in rows) / len(rows) if rows else float("nan")


def main() -> None:
    root = repo_root()
    series = load_series(root, "^NDX")
    outcomes = forward(series)
    labels = {day: flatten(record["label"]) for day, record in load(root).items()}
    days = sorted(day for day in labels if day in outcomes)

    base_all = sum(outcomes[day] for day in days) / len(days)
    pre = [day for day in days if day < HOLDOUT]
    post = [day for day in days if day >= HOLDOUT]
    print(f"{len(days)} 天有标签且有行情（{days[0]} → {days[-1]}）")
    print(f"基率：全部 {base_all:.1%}｜{HOLDOUT} 前 {sum(outcomes[d] for d in pre)/len(pre):.1%}｜之后 {sum(outcomes[d] for d in post)/len(post):.1%}\n")

    def report(name: str, selected: list[str]) -> None:
        if len(selected) < 25:
            print(f"| {name} | {len(selected)} | 样本太少，跳过 | | |")
            return
        s_pre = [d for d in selected if d < HOLDOUT]
        s_post = [d for d in selected if d >= HOLDOUT]
        hit = sum(outcomes[d] for d in selected) / len(selected)
        h_pre = sum(outcomes[d] for d in s_pre) / len(s_pre) if s_pre else float("nan")
        h_post = sum(outcomes[d] for d in s_post) / len(s_post) if s_post else float("nan")
        b_pre = sum(outcomes[d] for d in pre) / len(pre)
        b_post = sum(outcomes[d] for d in post) / len(post)
        # bootstrap：在和 selected 同样大小的随机样本上算同样的差
        diffs = []
        for _ in range(3000):
            sample = random.sample(days, len(selected))
            diffs.append(sum(outcomes[d] for d in sample) / len(sample) - base_all)
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        consistent = (h_pre > b_pre) == (h_post > b_post)
        print(
            f"| {name} | {len(selected)} | {hit:.1%} | {h_pre:.1%} | {h_post:.1%} | "
            f"{100*(hit-base_all):+.1f} pt [{100*lo:+.1f}, {100*hi:+.1f}] | {'同向' if consistent else '反向'} |"
        )

    print("## 方向：LLM 的 regime 标签\n")
    print("| 条件 | n | 上涨率 | 前段 | 后段 | 对基率（95% 区间） | 两段 |")
    print("|---|---|---|---|---|---|---|")
    for regime in ("risk_off", "neutral", "risk_on"):
        report(f"regime = {regime}", [d for d in days if labels[d]["regime"] == regime])

    print("\n## 强度：severity 分档（单调才叫有信号）\n")
    print("| 分档 | n | 上涨率 | 前段 | 后段 | 对基率（95% 区间） | 两段 |")
    print("|---|---|---|---|---|---|---|")
    ordered = sorted(days, key=lambda day: labels[day]["severity"])
    size = len(ordered) // 4
    for index in range(4):
        chunk = ordered[index * size : (index + 1) * size] if index < 3 else ordered[3 * size :]
        low, high = labels[chunk[0]]["severity"], labels[chunk[-1]]["severity"]
        report(f"severity 第 {index+1}/4 档（{low:.2f}–{high:.2f}）", chunk)

    print("\n## 否决：在信号日里按标签分组\n")
    cross = json.loads((root / "artifacts" / "cross.json").read_text())
    signals = [
        (day, outcome)
        for day, flag, outcome in zip(cross["time"], cross["flag"], cross["outcome"])
        if flag and outcome is not None and day in labels
    ]
    total = rate(signals)
    print(f"全部信号日 {len(signals)} 个，精确率 {total:.1%}（下面每一行和它比）\n")
    print("| 分组 | n | 精确率 | 前段 | 后段 | 差 |")
    print("|---|---|---|---|---|---|")
    for name, pick in (
        ("regime = risk_off", lambda f: f["regime"] == "risk_off"),
        ("regime = neutral", lambda f: f["regime"] == "neutral"),
        ("severity ≥ 0.6", lambda f: f["severity"] >= 0.6),
        ("severity < 0.3", lambda f: f["severity"] < 0.3),
        ("adverse 事件 ≥ 1", lambda f: f["adverse"] >= 1),
        ("没有 adverse 事件", lambda f: f["adverse"] == 0),
    ):
        group = [(day, outcome) for day, outcome in signals if pick(labels[day])]
        if len(group) < 20:
            continue
        g_pre = [o for day, o in group if day < HOLDOUT]
        g_post = [o for day, o in group if day >= HOLDOUT]
        hit = rate(group)
        print(
            f"| {name} | {len(group)} | {hit:.1%} | "
            f"{(sum(g_pre)/len(g_pre) if g_pre else float('nan')):.1%} | "
            f"{(sum(g_post)/len(g_post) if g_post else float('nan')):.1%} | {100*(hit-total):+.1f} pt |"
        )


if __name__ == "__main__":
    main()
