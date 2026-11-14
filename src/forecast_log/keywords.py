"""新闻关键字表：一个概念一条正则，桶分两边。

放在 `src/` 而不是 `experiments/`，因为 `daily.py` 每天要算一遍并写进记录，而
`experiments/news_keywords.py` 要拿同一张表做历史检验。同一张表，两处使用——否则量和
上线用的不是同一个东西。

表窄是故意的：宽正则（比如 `\\bbank\\b`）几乎每天都命中，测出来永远等于基率，等于没测。
"""

from __future__ import annotations

import re

LEXICON: dict[str, str] = {
    "fed": r"\bfed\b|federal reserve|fomc",
    "rate hike": r"raise rates|rate hike|hike rates|tighten",
    "rate cut": r"cut rates|rate cut|easing|stimulus",
    "inflation": r"inflation|consumer prices",
    "recession": r"recession|downturn|slowdown",
    "tariff": r"tariff|trade war|trade dispute",
    "jobs": r"unemployment|payrolls|jobless|jobs report|job cuts",
    "gdp": r"\bgdp\b|economic growth",
    "oil": r"oil price|crude|opec|barrel",
    "dollar": r"dollar|currency market",
    "bond yields": r"bond yield|treasury yield|yields",
    "stocks up": r"record high|shares (rise|surge|gain)|rally|stocks (rise|jump|climb)",
    "stocks down": r"sell-?off|plunge|slump|crash|tumble|stocks (fall|drop|slide)",
    "earnings": r"earnings|profit warning|quarterly results",
    "ipo": r"\bipo\b|initial public offering",
    "bankruptcy": r"bankrupt|insolven|default",
    "china": r"\bchina\b|beijing|pboc",
    "europe": r"\becb\b|eurozone|european central bank",
    "geopolitics": r"sanction|invasion|missile|air ?strike|military operation|\bwar\b",
    "trade deal": r"trade deal|trade agreement|exports|imports",
    "crypto": r"bitcoin|crypto|blockchain",
    "housing": r"housing|home sales|mortgage",
    "banks": r"\bbank\b|banking crisis|lender",
    "election": r"election|referendum|vote",
}

RISK_OFF = ("geopolitics", "recession", "bankruptcy", "tariff", "stocks down", "banks")
RISK_ON = ("rate cut", "stocks up", "trade deal", "housing")

COMPILED = {name: re.compile(pattern, re.I) for name, pattern in LEXICON.items()}


def scan(titles: list[str]) -> dict:
    """当天标题 → 命中的关键字、两边桶的计数、归一化后的净调子。

    这些数**不进模型**：三层检验（单词、分档、否决）都没有过线，唯一接近的是
    「risk-off 词命中 ≥2 个时信号精确率低 6 个点」，只有 1.4 个标准误。所以现在只记录，
    不改信号；攒够样本再谈。见 README「新闻关键字」一节。
    """
    text = " ".join(titles)
    hits = [name for name, pattern in COMPILED.items() if pattern.search(text)]
    off = [name for name in hits if name in RISK_OFF]
    on = [name for name in hits if name in RISK_ON]
    return {
        "items": len(titles),
        "hits": hits,
        "risk_off_hits": off,
        "risk_on_hits": on,
        "tone": round((len(on) - len(off)) / max(len(titles), 1), 4),
    }
