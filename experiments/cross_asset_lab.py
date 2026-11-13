"""跨资产特征能不能预测指数 5 天的涨跌。

和 direction_lab 同一套判据，只是把线索换成别的市场：
美债收益率、美元、信用（高收益债 vs 长债）、广度（等权 vs 市值权）、黄金。

样本起点被迫推到 2008（HYG 从 2007 才有），所以真正独立的 5 天窗口
只有约 900 个 —— 这是这题最大的先天限制，先写在前面。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from forecast_log.bars import Series, load_series
from forecast_log.stats import expanding_z
from forecast_log.paths import series_path

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 5
BLOCK = 60
RESAMPLES = 4000

# (显示名, 符号, 变换)
CROSS = [
    ("10 年期收益率 20 日变化", "^TNX", "diff20"),
    ("美元 20 日涨跌", "DX-Y.NYB", "ret20"),
    ("信用 20 日（HYG/TLT）", "HYG/TLT", "ret20"),
    ("广度 20 日（RSP/SPY）", "RSP/SPY", "ret20"),
    ("黄金 20 日涨跌", "GC=F", "ret20"),
    ("VIX 相对一年中位", "^VIX", "offset"),
]


def aligned(dates: np.ndarray, series: Series) -> np.ndarray:
    """每个日期的「当天或最近一个收盘」，只用过去。"""
    out = np.full(len(dates), np.nan)
    bars = [(np.datetime64(bar.date), bar.close) for bar in series.bars]
    pointer, last = 0, np.nan
    for position, day in enumerate(dates):
        while pointer < len(bars) and bars[pointer][0] <= day:
            last = bars[pointer][1]
            pointer += 1
        out[position] = last
    return out


def transform(values: np.ndarray, how: str) -> np.ndarray:
    if how == "ret20":
        out = np.full(len(values), np.nan)
        out[20:] = np.log(values[20:]) - np.log(values[:-20])
        return out
    if how == "diff20":
        out = np.full(len(values), np.nan)
        out[20:] = values[20:] - values[:-20]
        return out
    if how == "offset":
        out = np.full(len(values), np.nan)
        for i in range(252, len(values)):
            window = values[i - 251 : i + 1]
            if np.all(np.isfinite(window)):
                out[i] = np.log(values[i] / np.median(window))
        return out
    raise ValueError(how)


def load_prices(symbol: str, dates: np.ndarray) -> np.ndarray:
    if "/" in symbol:
        left, right = symbol.split("/")
        return aligned(dates, load_series(series_path(ROOT, left))) / aligned(
            dates, load_series(series_path(ROOT, right))
        )
    return aligned(dates, load_series(series_path(ROOT, symbol)))


def blocks(values: np.ndarray) -> tuple[float, float, float]:
    """均值 + 分块 bootstrap 区间；重叠的 5 天样本按 60 天块处理。"""
    rng = np.random.default_rng(17)
    n = len(values)
    count = max(1, n // BLOCK)
    draws = np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        starts = rng.integers(0, max(1, n - BLOCK), size=count)
        index = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in starts])[:n]
        draws[k] = values[index].mean()
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    ndx = load_series(series_path(ROOT, "^NDX"))
    dates = np.array([np.datetime64(bar.date) for bar in ndx.bars])
    closes = np.array([bar.close for bar in ndx.bars], dtype=float)
    log_close = np.log(closes)
    n = len(closes)

    forward = np.full(n, np.nan)
    forward[:-HORIZON] = log_close[HORIZON:] - log_close[:-HORIZON]
    labels = np.where(np.isnan(forward), np.nan, (forward > 0).astype(float))

    columns, names = [], []
    for name, symbol, how in CROSS:
        column = transform(load_prices(symbol, dates), how)
        columns.append(column)
        names.append(name)

    print("=== 一、每个跨资产线索：高 / 低 10% 时，后 5 天上涨的比例 ===")
    print(f"{'线索':<26}{'桶':<8}{'n':>6}{'命中':>8}{'基率':>8}{'差':>8}   95% 区间")
    best = []
    for column, name in zip(columns, names):
        z = expanding_z(column)
        for side, mask in (("低 10%", z <= -1.2816), ("高 10%", z >= 1.2816)):
            keep = mask & ~np.isnan(labels)
            if keep.sum() < 150:
                continue
            hits = labels[keep]
            base = labels[~np.isnan(labels)].mean()
            mean, low, high = blocks(hits)
            star = " *" if not (low <= base <= high) else ""
            print(
                f"{name:<26}{side:<8}{keep.sum():>6}{mean:>8.1%}{base:>8.1%}{mean - base:>+8.1%}"
                f"   [{low:.1%}, {high:.1%}]{star}"
            )
            best.append((abs(mean - base), name, side, mean, base, low, high, keep.sum()))

    print()
    print(f"共做了 {len(best)} 次比较；带 * 的表示区间不含基率（未做多重比较修正）")
    print()
    print("=== 二、把全部线索一起塞进模型 ===")
    from direction_lab import expanding_baseline, fit  # 同一套滚动训练

    design = np.column_stack([expanding_z(c) for c in columns])
    usable = ~np.isnan(design).any(axis=1) & ~np.isnan(labels)
    weights, last = None, -10**9
    probability = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(design[i]).any():
            continue
        ready = np.flatnonzero(usable[: i + 1])
        if ready.size < 250:
            continue
        if weights is None or i - last >= 21:
            weights = fit(design[ready], labels[ready])
            last = i
        probability[i] = np.clip(
            1.0 / (1.0 + np.exp(-float(weights[0] + design[i] @ weights[1:]))), 0.02, 0.98
        )
    base = expanding_baseline(labels)
    keep = ~np.isnan(probability) & ~np.isnan(labels) & ~np.isnan(base)
    score = float(np.mean((probability[keep] - labels[keep]) ** 2))
    constant = float(np.mean((base[keep] - labels[keep]) ** 2))
    print(f"样本 {keep.sum()} 天（≈{keep.sum() / HORIZON:.0f} 个不重叠的 5 天窗口）")
    print(f"模型 Brier {score:.4f}   只按基率 {constant:.4f}   差 {score - constant:+.4f}")
    for shrink in (0.3, 0.5):
        mixed = base + shrink * (probability - base)
        mixed_score = float(np.mean((mixed[keep] - labels[keep]) ** 2))
        print(f"  收缩 {shrink}: {mixed_score:.4f}（{mixed_score - constant:+.4f}）")


if __name__ == "__main__":
    main()
