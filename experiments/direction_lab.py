"""方向题的实验台：先证明「没解」，或找出唯一撑得住的那条。

不打点，不画图。只回答两件事：

1. 把概率往基率收缩 / 重标定之后，Brier 能不能至少不输给常数。
2. 单因子分桶里，有没有哪一桶的命中率跟基率拉开了统计上站得住的差距。

用法：.venv/bin/python experiments/direction_lab.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from forecast_log.bars import load_series
from forecast_log.stats import expanding_z, _sigmoid
from forecast_log.features import load_matrix
from forecast_log.paths import series_path

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 5
MIN_TRAIN = 250
RETRAIN = 21
BLOCK = 60  # 分块 bootstrap 的块长：5 天重叠的样本不能按独立处理
RESAMPLES = 2000


def fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    a = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(a.shape[1])
    pen = ridge * np.eye(a.shape[1])
    pen[0, 0] = 0.0
    for _ in range(40):
        p = 1.0 / (1.0 + np.exp(-(a @ w)))
        weight = np.clip(p * (1 - p), 1e-6, None)
        step = np.linalg.solve(a.T @ (a * weight[:, None]) + pen, a.T @ (y - p))
        w = w + step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def walk(design: np.ndarray, labels: np.ndarray, ridge: float) -> np.ndarray:
    """逐日滚动，只用当天之前的数据训练。返回每天的原始概率。"""
    out = np.full(len(labels), np.nan)
    usable = ~np.isnan(design).any(axis=1) & ~np.isnan(labels)
    weights = None
    last = -10**9
    for i in range(len(labels)):
        if np.isnan(design[i]).any():
            continue
        ready = np.flatnonzero(usable[: i + 1])
        if ready.size < MIN_TRAIN:
            continue
        if weights is None or i - last >= RETRAIN:
            weights = fit(design[ready], labels[ready], ridge)
            last = i
        out[i] = np.clip(_sigmoid(float(weights[0] + design[i] @ weights[1:])), 0.02, 0.98)
    return out


def expanding_baseline(labels: np.ndarray) -> np.ndarray:
    """当天之前所有已解决样本的基率。"""
    out = np.full(len(labels), np.nan)
    seen = []
    for i in range(len(labels)):
        if seen:
            out[i] = float(np.mean(seen))
        if not np.isnan(labels[i]):
            seen.append(labels[i])
    return out


def brier(p: np.ndarray, y: np.ndarray, keep: np.ndarray) -> float:
    """同一批日子上比较，否则模型和常数用的不是同一批样本。"""
    return float(np.mean((p[keep] - y[keep]) ** 2))


def block_interval(hits: np.ndarray, samples: int = RESAMPLES) -> tuple[float, float]:
    """命中 0/1 序列的分块 bootstrap 区间。"""
    rng = np.random.default_rng(11)
    n = len(hits)
    blocks = max(1, n // BLOCK)
    means = np.empty(samples)
    for k in range(samples):
        starts = rng.integers(0, max(1, n - BLOCK), size=blocks)
        piece = np.concatenate([hits[s : s + BLOCK] for s in starts])[:n]
        means[k] = piece.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    ndx = load_series(series_path(ROOT, "^NDX"))
    dates = [bar.date for bar in ndx.bars]
    closes = np.array([bar.close for bar in ndx.bars], dtype=float)
    log_close = np.log(closes)
    n = len(closes)
    matrix = load_matrix(ROOT, dates)

    def ret(span: int) -> np.ndarray:
        out = np.full(n, np.nan)
        out[span:] = log_close[span:] - log_close[:-span]
        return out

    middle = np.full(n, np.nan)
    for i in range(199, n):
        middle[i] = closes[i - 199 : i + 1].mean()
    vol = np.full(n, np.nan)
    for i in range(20, n):
        vol[i] = float(np.std(np.diff(log_close[i - 20 : i + 1])))

    names = ["5 日涨跌", "20 日涨跌", "200 日涨跌", "离 200 日均线", "20 日波动"]
    raw = [ret(5), ret(20), ret(200), log_close - np.log(middle), vol]
    if matrix is not None:
        names.append("期限结构斜率")
        raw.append(matrix[:, 0])
    features = np.column_stack(raw)

    forward = np.full(n, np.nan)
    forward[:-HORIZON] = log_close[HORIZON:] - log_close[:-HORIZON]

    # 目标一：5 天后收盘更高（天然 60% 是「更高」，因为指数长期向上）
    labels_up = np.where(np.isnan(forward), np.nan, (forward > 0).astype(float))
    # 目标二：5 天涨跌比过去所有 5 天涨跌的中位数更高（去掉漂移，各半）
    drift = np.full(n, np.nan)
    for i in range(252, n):
        history = forward[:i]
        history = history[~np.isnan(history)]
        drift[i] = float(np.median(history)) if history.size else np.nan
    labels_excess = np.where(
        np.isnan(forward) | np.isnan(drift), np.nan, (forward > drift).astype(float)
    )

    design = np.column_stack([expanding_z(features[:, j]) for j in range(features.shape[1])])

    print("=== 一、把概率收缩 / 重标定，能不能至少追平常数 ===")
    print(f"{'目标':<10}{'做法':<16}{'Brier':>8}{'常数':>9}{'差':>9}")
    for label_name, labels in (("5 天后更高", labels_up), ("超过漂移", labels_excess)):
        base = expanding_baseline(labels)
        variants = {
            "原始": (walk(design, labels, 1e-3), None),
            "收缩 0.5": (walk(design, labels, 1e-3), "shrink"),
            "logit 对齐基率": (walk(design, labels, 1e-3), "calibrate"),
            "强正则 0.3": (walk(design, labels, 0.3), None),
            "只用基率": (base.copy(), None),
        }
        keep = ~np.isnan(labels) & ~np.isnan(base)
        for raw, _ in variants.values():
            keep &= ~np.isnan(raw)
        for name, (raw_probability, fix) in variants.items():
            probability = raw_probability.copy()
            if fix == "shrink":
                probability = base + 0.5 * (probability - base)
            elif fix == "calibrate":
                shift = logit(base) - np.nanmean(logit(raw_probability))
                probability = _sigmoid(np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1)) + shift)
            score = brier(probability, labels, keep)
            constant = brier(base, labels, keep)
            print(
                f"{label_name:<10}{name:<16}{score:>8.4f}{constant:>9.4f}{score - constant:>+9.4f}"
            )

    print()
    print("=== 二、单因子分桶，最高的那一桶跟基率比 ===")
    print(f"{'因子':<14}{'桶':<8}{'n':>6}{'命中':>8}{'基率':>8}{'差':>8}   95% 区间（分块）")
    for column, name in enumerate(names):
        values = features[:, column]
        z = expanding_z(values)
        for side in ("低 10%", "高 10%"):
            mask = z <= -1.2816 if side == "低 10%" else z >= 1.2816
            keep = mask & ~np.isnan(labels_up)
            if keep.sum() < 100:
                continue
            hits = labels_up[keep]
            base = expanding_baseline(labels_up)[keep]
            low, high = block_interval(hits)
            print(
                f"{name:<14}{side:<8}{keep.sum():>6}{hits.mean():>8.1%}{base.mean():>8.1%}"
                f"{(hits.mean() - base.mean()):>+8.1%}   [{low:.1%}, {high:.1%}]"
            )


if __name__ == "__main__":
    main()
