"""共用的小工具：扩张标准化。

分子分母都只用当天及以前的值，所以 i 这一行永远不会看到 i 之后的数据。
"""

from __future__ import annotations

import numpy as np

MIN_ROWS = 30


def expanding_z(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = values[: i + 1]
        window = window[~np.isnan(window)]
        if len(window) < MIN_ROWS or np.isnan(values[i]):
            continue
        spread = window.std()
        if spread <= 0:
            continue
        out[i] = (values[i] - window.mean()) / spread
    return out
