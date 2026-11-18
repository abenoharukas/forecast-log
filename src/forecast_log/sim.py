"""Multi-agent policy and market simulation with explicit market-clearing state equations.

Unlike `group_chat.py` (where models debate in prose), actors here never output prices.
Each heterogeneous institutional actor observes a strictly time-gated `MarketState`
(`AsOf.check` enforced on every bar and headline) plus an exogenous `PolicyShock`, and
submits bounded portfolio flow decisions (`equity_delta`, `bond_delta`, `credit_delta`,
`vol_hedge_delta`). A deterministic market-clearing engine aggregates flows under
endogenous dealer liquidity (Kyle's lambda) to compute the next state vector:

    S_t = (NDX_t, VXN_t, TNX_t, ln(HYG_t / TLT_t), DXY_t)

Why four distinct institutional mandates instead of N identical prompts:
1. `macro_fund` (Global Macro, w=0.30): trades rate duration vs. growth/inflation mix
   and USD liquidity, flipping its equity-yield sensitivity when trailing 63-session
   stock-yield correlation turns negative (inflation/discount-rate regime vs. ZLB growth).
2. `vol_target_cta` (Volatility-Targeting CTA / Risk Parity, w=0.25): mechanically scales
   equity exposure inversely to implied volatility (`target_vol / VXN`), generating
   multi-day deleveraging cascades after vol spikes and re-leveraging as fear subsides.
3. `corp_credit_hedger` (Credit & Corporate Hedger, w=0.25): prices tariff/margin shocks
   and credit spread widening (`HYG/TLT`), buying downside protection (`vol_hedge_delta`).
4. `options_dealer_mm` (Options & Equity Market Maker, w=0.20): supplies mean-reverting
   liquidity in calm contango (`^VIX9D / ^VIX3M < 0.98`) but pulls liquidity and hedges
   short gamma in backwardation, amplifying price impact non-linearly.

Supports three interchangeable actor backends on the exact same clearing equations:
- `rule`: Calibrated micro-founded behavioural reaction functions (fast, reproducible).
- `homogeneous`: Single undifferentiated actor ablation control (tests value of heterogeneity).
- `llm`: Live LLM actors (single model or multi-vendor e.g. DeepSeek + Qwen) returning
  validated JSON flow vectors, with full token/cost/latency accounting.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np

from forecast_log import events, providers
from forecast_log.agent.env import merged_env
from forecast_log.agent.eval import brier, ece, log_loss, reliability
from forecast_log.agent.gate import NY, AsOf, LookAheadError
from forecast_log.agent.llm import LLM, Usage
from forecast_log.bars import Series, load_series
from forecast_log.db import get_news_fetch, has_series_in_db
from forecast_log.estimate import AnalogModel

REQUIRED_SYMBOLS = ("^NDX", "^VXN", "^TNX", "HYG", "TLT", "DX-Y.NYB")
OPTIONAL_SYMBOLS = ("^VIX9D", "^VIX3M")

ActorMode = Literal["rule", "homogeneous", "llm"]


# --------------------------------------------------------------------- data types


@dataclass(frozen=True)
class MarketState:
    """Point-in-time cross-asset state anchored at or before `as_of`."""

    as_of: str
    step: int
    ndx: float
    vxn: float
    tnx: float                  # 10Y Treasury yield in percentage points (e.g. 4.45)
    credit_log_ratio: float     # ln(HYG / TLT)
    dxy: float                  # US Dollar Index
    vxn_median_252d: float
    vxn_std_252d: float
    realized_vol_21d: float     # Annualized 21-session realized NDX vol (%)
    vol_term_ratio: float       # ^VIX9D / ^VIX3M (1.0 if unavailable)
    stock_yield_corr_63d: float # Trailing 63-session corr(r_NDX, d_TNX): <0 = inflation regime
    ndx_ret_5d: float
    ndx_ret_20d: float
    tnx_chg_20d_bps: float
    credit_chg_20d: float
    dxy_ret_20d: float
    prev_implied_leverage: float = 1.0

    @property
    def vxn_z(self) -> float:
        return (self.vxn - self.vxn_median_252d) / max(self.vxn_std_252d, 1.0)

    @property
    def vol_risk_premium(self) -> float:
        return self.vxn - self.realized_vol_21d

    @property
    def regime_label(self) -> str:
        if self.stock_yield_corr_63d < -0.10 or self.tnx >= 3.25:
            base = "inflation_discount_rate"
        else:
            base = "growth_reflation"
        stress = "stressed_backwardation" if (self.vol_term_ratio > 1.03 or self.vxn_z > 1.2) else "normal_liquidity"
        return f"{base}:{stress}"

    def to_summary_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "step": self.step,
            "ndx": round(self.ndx, 2),
            "vxn": round(self.vxn, 2),
            "tnx_pct": round(self.tnx, 3),
            "credit_log_hyg_tlt": round(self.credit_log_ratio, 4),
            "dxy": round(self.dxy, 3),
            "vxn_median_252d": round(self.vxn_median_252d, 2),
            "vxn_z": round(self.vxn_z, 3),
            "realized_vol_21d_pct": round(self.realized_vol_21d, 2),
            "vol_term_ratio": round(self.vol_term_ratio, 3),
            "stock_yield_corr_63d": round(self.stock_yield_corr_63d, 3),
            "ndx_ret_5d_pct": round(self.ndx_ret_5d * 100.0, 2),
            "ndx_ret_20d_pct": round(self.ndx_ret_20d * 100.0, 2),
            "tnx_chg_20d_bps": round(self.tnx_chg_20d_bps, 1),
            "credit_chg_20d_pct": round(self.credit_chg_20d * 100.0, 2),
            "dxy_ret_20d_pct": round(self.dxy_ret_20d * 100.0, 2),
            "regime": self.regime_label,
        }


@dataclass(frozen=True)
class PolicyShock:
    """Exogenous policy or macro event impulse applied to the economy."""

    kind: str = "custom"          # fomc | cpi | nfp | tariff | fiscal | credit | custom
    title: str = ""
    rate_bps: float = 0.0         # Instantaneous 10Y / policy rate impulse in bps (+ = hawkish)
    fx_pct: float = 0.0           # Exogenous USD appreciation impulse in % (+ = stronger USD)
    inflation_z: float = 0.0      # Supply / inflation surprise in std devs (+ = hotter CPI/cost)
    growth_z: float = 0.0         # Real demand / payrolls surprise in std devs (+ = stronger growth)
    tariff_pct: float = 0.0       # Effective new tariff / trade barrier shock in percentage points
    uncertainty_z: float = 0.0    # Policy / geopolitical ambiguity impulse in std devs
    persistence: float = 0.35     # Decay factor rho in [0, 0.9] across post-shock steps s=2..H
    headlines: tuple[str, ...] = ()

    def at_step(self, step: int) -> PolicyShock:
        """Step 1 receives the primary shock; steps >= 2 receive decaying aftershocks."""
        if step <= 1:
            return self
        scale = float(self.persistence ** (step - 1))
        return PolicyShock(
            kind=self.kind,
            title=self.title,
            rate_bps=self.rate_bps * scale * 0.25,
            fx_pct=self.fx_pct * scale * 0.25,
            inflation_z=self.inflation_z * scale,
            growth_z=self.growth_z * scale,
            tariff_pct=self.tariff_pct * scale,
            uncertainty_z=self.uncertainty_z * scale,
            persistence=self.persistence,
            headlines=self.headlines if step == 1 else (),
        )

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "rate_bps": round(self.rate_bps, 2),
            "fx_pct": round(self.fx_pct, 3),
            "inflation_z": round(self.inflation_z, 3),
            "growth_z": round(self.growth_z, 3),
            "tariff_pct": round(self.tariff_pct, 2),
            "uncertainty_z": round(self.uncertainty_z, 3),
            "persistence": round(self.persistence, 3),
            "headlines": list(self.headlines),
        }


PRESET_SHOCKS: dict[str, PolicyShock] = {
    "tariff_25": PolicyShock(
        kind="tariff",
        title="US announces +25% tariff on semiconductor and industrial imports; retaliatory duties threatened",
        rate_bps=6.0,
        fx_pct=0.65,
        inflation_z=1.35,
        growth_z=-0.90,
        tariff_pct=25.0,
        uncertainty_z=1.60,
        persistence=0.55,
    ),
    "fomc_hawkish_25": PolicyShock(
        kind="fomc",
        title="Surprise +25 bps hawkish FOMC guidance shift; median dot plot removes two cuts",
        rate_bps=14.0,
        fx_pct=0.55,
        inflation_z=0.65,
        growth_z=0.10,
        tariff_pct=0.0,
        uncertainty_z=0.80,
        persistence=0.40,
    ),
    "fomc_dovish_cut": PolicyShock(
        kind="fomc",
        title="Dovish 50 bps Fed rate cut with liquidity backstop guidance and benign inflation outlook",
        rate_bps=-18.0,
        fx_pct=-0.70,
        inflation_z=-0.50,
        growth_z=0.45,
        tariff_pct=0.0,
        uncertainty_z=-0.60,
        persistence=0.40,
    ),
    "cpi_hot_surprise": PolicyShock(
        kind="cpi",
        title="Core CPI prints +0.5% MoM (2.1 sigma above consensus); shelter and services re-accelerate",
        rate_bps=16.0,
        fx_pct=0.60,
        inflation_z=2.10,
        growth_z=-0.20,
        tariff_pct=0.0,
        uncertainty_z=1.10,
        persistence=0.45,
    ),
    "nfp_goldilocks": PolicyShock(
        kind="nfp",
        title="Nonfarm payrolls beat expectations (+265k) while hourly wage growth cools to +0.2% MoM",
        rate_bps=3.0,
        fx_pct=0.15,
        inflation_z=-0.45,
        growth_z=1.25,
        tariff_pct=0.0,
        uncertainty_z=-0.55,
        persistence=0.35,
    ),
    "credit_crunch": PolicyShock(
        kind="credit",
        title="Regional banking & high-yield refinancing stress; credit spreads widen sharply",
        rate_bps=-12.0,
        fx_pct=0.40,
        inflation_z=-0.30,
        growth_z=-1.40,
        tariff_pct=0.0,
        uncertainty_z=1.90,
        persistence=0.60,
    ),
}


@dataclass(frozen=True)
class ActorSpec:
    actor_id: str
    label: str
    weight: float
    mandate: str
    constraints: str


ACTOR_SPECS: tuple[ActorSpec, ...] = (
    ActorSpec(
        actor_id="macro_fund",
        label="Global Macro Hedge Fund",
        weight=0.30,
        mandate=(
            "Discretionary macro fund trading equity index beta, 10Y Treasury duration, and FX "
            "liquidity. Evaluates whether yield moves reflect real growth (bullish when stock-yield "
            "correlation is positive and inflation is contained) or discount-rate/inflation tightening "
            "(bearish when stock-yield correlation is negative or inflation/tariffs spike)."
        ),
        constraints="Net flow per step bounded in [-1, +1]; penalizes fighting strong USD tightening.",
    ),
    ActorSpec(
        actor_id="vol_target_cta",
        label="Vol-Targeting / Risk-Parity CTA",
        weight=0.25,
        mandate=(
            "Systematic volatility-targeting and trend-following fund maintaining ~18% annualized "
            "portfolio volatility. Mechanically cuts equity exposure when implied/realized vol rises "
            "above target (especially on multi-day follow-through) and re-leverages when elevated "
            "short-dated vol mean-reverts."
        ),
        constraints="Rule-bound to volatility target ratio (18 / VXN) and 20-day price trend.",
    ),
    ActorSpec(
        actor_id="corp_credit_hedger",
        label="Corporate & Credit Hedger",
        weight=0.25,
        mandate=(
            "Institutional credit desk and multinational corporate treasury hedging supply-chain "
            "tariffs, refinancing costs, and HYG/TLT credit spread widening. Buys index put "
            "protection (vol_hedge_delta > 0) and trims credit/equity risk when tariffs or credit "
            "stress rise."
        ),
        constraints="Prioritizes drawdown protection; slow to unwind hedges until credit stabilizes.",
    ),
    ActorSpec(
        actor_id="options_dealer_mm",
        label="Options & Equity Market Maker",
        weight=0.20,
        mandate=(
            "Options gamma/vega dealer and equity liquidity provider. In calm contango "
            "(vol_term_ratio < 0.98), absorbs imbalances and sells overpriced vol; in stressed "
            "backwardation (vol_term_ratio > 1.03 or high VXN z-score), hedges short gamma by "
            "pulling liquidity and trading with momentum until panic hedging peaks."
        ),
        constraints="Inventory-constrained; widens price impact when net order flow is one-sided.",
    ),
)


@dataclass(frozen=True)
class ActorAction:
    actor_id: str
    equity_delta: float       # [-1, +1]: net equity index order flow (+buy / -sell)
    bond_delta: float         # [-1, +1]: net 10Y Treasury duration demand (+buy bonds -> lower yield)
    credit_delta: float       # [-1, +1]: net HYG vs TLT credit risk appetite (+tighter spreads)
    vol_hedge_delta: float    # [-1, +1]: net VXN option protection demand (+buy vol / -sell vol)
    confidence: float = 0.6
    rationale: str = ""
    model: str = "rule"
    usage: Usage = field(default_factory=Usage)

    def clipped(self) -> ActorAction:
        return ActorAction(
            actor_id=self.actor_id,
            equity_delta=_clip(self.equity_delta, -1.0, 1.0),
            bond_delta=_clip(self.bond_delta, -1.0, 1.0),
            credit_delta=_clip(self.credit_delta, -1.0, 1.0),
            vol_hedge_delta=_clip(self.vol_hedge_delta, -1.0, 1.0),
            confidence=_clip(self.confidence, 0.0, 1.0),
            rationale=(self.rationale or "")[:240],
            model=self.model,
            usage=self.usage,
        )

    def to_json(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "model": self.model,
            "equity_delta": round(self.equity_delta, 4),
            "bond_delta": round(self.bond_delta, 4),
            "credit_delta": round(self.credit_delta, 4),
            "vol_hedge_delta": round(self.vol_hedge_delta, 4),
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
        }


class SimActor(Protocol):
    spec: ActorSpec

    def act(
        self,
        state: MarketState,
        shock: PolicyShock,
        prior_steps: Sequence[SimulationStep],
    ) -> ActorAction: ...


@dataclass(frozen=True)
class ClearingDiagnostics:
    net_equity_flow: float
    net_bond_flow: float
    net_credit_flow: float
    net_vol_hedge_flow: float
    actor_disagreement: float
    kyle_lambda: float
    step_equity_log_return: float
    step_vxn_change: float
    step_tnx_change_bps: float

    def to_json(self) -> dict:
        return {k: round(v, 5) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class SimulationStep:
    step: int
    shock: PolicyShock
    actions: tuple[ActorAction, ...]
    clearing: ClearingDiagnostics
    state_after: MarketState

    def to_json(self) -> dict:
        return {
            "step": self.step,
            "shock": self.shock.to_json(),
            "actions": [a.to_json() for a in self.actions],
            "clearing": self.clearing.to_json(),
            "state_after": self.state_after.to_summary_dict(),
        }


@dataclass(frozen=True)
class ProbabilisticDistribution:
    """Calibrated distribution over terminal simulation outcomes from structural residuals."""

    p_ndx_up: float
    p_vxn_up: float
    p_ndx_drawdown_gt_2pct: float
    ndx_return_p10: float
    ndx_return_p50: float
    ndx_return_p90: float
    vxn_terminal_p10: float
    vxn_terminal_p50: float
    vxn_terminal_p90: float
    residual_vol_step: float
    residual_vxn_step: float = 1.25

    def to_json(self) -> dict:
        return {
            "p_ndx_up": round(self.p_ndx_up, 4),
            "p_vxn_up": round(self.p_vxn_up, 4),
            "p_ndx_drawdown_gt_2pct": round(self.p_ndx_drawdown_gt_2pct, 4),
            "ndx_return_pct_quantiles": {
                "p10": round(self.ndx_return_p10 * 100.0, 2),
                "p50": round(self.ndx_return_p50 * 100.0, 2),
                "p90": round(self.ndx_return_p90 * 100.0, 2),
            },
            "vxn_terminal_quantiles": {
                "p10": round(self.vxn_terminal_p10, 2),
                "p50": round(self.vxn_terminal_p50, 2),
                "p90": round(self.vxn_terminal_p90, 2),
            },
        }


@dataclass(frozen=True)
class SimulationResult:
    as_of: AsOf
    mode: str
    initial_state: MarketState
    shock: PolicyShock
    steps: tuple[SimulationStep, ...]
    distribution: ProbabilisticDistribution
    usage: Usage = field(default_factory=Usage)

    @property
    def final_state(self) -> MarketState:
        return self.steps[-1].state_after if self.steps else self.initial_state

    @property
    def cumulative_ndx_log_return(self) -> float:
        return math.log(self.final_state.ndx / self.initial_state.ndx)

    @property
    def cumulative_vxn_change(self) -> float:
        return self.final_state.vxn - self.initial_state.vxn

    def to_json(self) -> dict:
        return {
            "as_of": self.as_of.to_json(),
            "mode": self.mode,
            "initial_state": self.initial_state.to_summary_dict(),
            "shock": self.shock.to_json(),
            "steps": [s.to_json() for s in self.steps],
            "final_state": self.final_state.to_summary_dict(),
            "cumulative_ndx_return_pct": round((math.exp(self.cumulative_ndx_log_return) - 1.0) * 100.0, 3),
            "cumulative_vxn_change": round(self.cumulative_vxn_change, 3),
            "distribution": self.distribution.to_json(),
            "usage": self.usage.to_json(),
        }


# ------------------------------------------------------ point-in-time state extraction


def _clip(val: float, low: float, high: float) -> float:
    return float(min(max(float(val), low), high))


def _visible_closes(series: Series, as_of: AsOf) -> tuple[list[date], np.ndarray]:
    bars = [b for b in series.bars if b.date <= as_of.session]
    if len(bars) < 30:
        raise ValueError(f"{series.symbol} has fewer than 30 bars at or before {as_of.session}")
    as_of.check(bars[-1].date, f"sim:{series.symbol}")
    return [b.date for b in bars], np.array([b.close for b in bars], dtype=float)


def _aligned_at_or_before(series: Series, target_days: list[date], as_of: AsOf) -> np.ndarray:
    days, closes = _visible_closes(series, as_of)
    out = np.full(len(target_days), np.nan, dtype=float)
    ptr = 0
    last = np.nan
    for idx, day in enumerate(target_days):
        as_of.check(day, f"sim:align:{series.symbol}")
        while ptr < len(days) and days[ptr] <= day:
            last = closes[ptr]
            ptr += 1
        out[idx] = last
    return out


def gated_headlines_for_day(root: Path, day: date, as_of: AsOf, limit: int = 8) -> tuple[str, ...]:
    """Load event-day headlines from SQLite or JSON, enforcing `published_at <= as_of.at`."""
    record = get_news_fetch(root, day)
    docs: list[dict] = []
    if record and record.get("documents"):
        docs = list(record["documents"])
    else:
        json_path = root / "data" / "news" / f"{day.isoformat()}.json"
        if json_path.is_file():
            payload = json.loads(json_path.read_text())
            docs = payload.get("documents", []) if isinstance(payload, dict) else list(payload)

    titles: list[str] = []
    for doc in docs:
        raw_pub = str(doc.get("published_at") or "").strip()
        if raw_pub:
            if "T" in raw_pub:
                parsed = datetime.fromisoformat(raw_pub)
                as_of.check_published(parsed, f"sim:news:{doc.get('id', '?')}")
            else:
                as_of.check(date.fromisoformat(raw_pub[:10]), f"sim:news:{doc.get('id', '?')}")
        title = str(doc.get("title") or "").strip()
        if title:
            titles.append(title[:180])
        if len(titles) >= limit:
            break
    return tuple(titles)


@dataclass
class HistoricalPanel:
    """Pre-loaded cross-asset series for fast, leak-checked state construction across backtests."""

    root: Path
    series_map: dict[str, Series]

    @classmethod
    def load(cls, root: Path) -> HistoricalPanel:
        loaded: dict[str, Series] = {}
        for sym in REQUIRED_SYMBOLS:
            loaded[sym] = load_series(root, sym)
        for sym in OPTIONAL_SYMBOLS:
            if has_series_in_db(root, sym) or (root / "data" / f"{sym.replace('^', '')}.json").is_file():
                try:
                    loaded[sym] = load_series(root, sym)
                except Exception:
                    pass
        return cls(root=root, series_map=loaded)

    def build_state(self, as_of: AsOf) -> tuple[MarketState, float, float]:
        """Construct `MarketState` strictly from bars <= `as_of.session` + point-in-time residual scales."""
        ndx_days, ndx = _visible_closes(self.series_map["^NDX"], as_of)
        take = min(len(ndx_days), 300)
        days = ndx_days[-take:]
        ndx = ndx[-take:]

        vxn = _aligned_at_or_before(self.series_map["^VXN"], days, as_of)
        tnx = _aligned_at_or_before(self.series_map["^TNX"], days, as_of)
        hyg = _aligned_at_or_before(self.series_map["HYG"], days, as_of)
        tlt = _aligned_at_or_before(self.series_map["TLT"], days, as_of)
        dxy = _aligned_at_or_before(self.series_map["DX-Y.NYB"], days, as_of)

        valid = (
            np.isfinite(ndx)
            & np.isfinite(vxn)
            & np.isfinite(tnx)
            & np.isfinite(hyg)
            & np.isfinite(tlt)
            & np.isfinite(dxy)
        )
        if int(valid.sum()) < 65:
            raise ValueError(f"Insufficient aligned cross-asset history at {as_of.session}")

        days = [d for d, ok in zip(days, valid) if ok]
        ndx, vxn, tnx, hyg, tlt, dxy = (arr[valid] for arr in (ndx, vxn, tnx, hyg, tlt, dxy))
        as_of.check(days[-1], "sim:panel_last_day")

        credit = np.log(hyg / tlt)
        ndx_rets = np.diff(np.log(ndx))
        tnx_diffs = np.diff(tnx)
        vxn_diffs = np.diff(vxn)

        win_v = vxn[-252:] if len(vxn) >= 252 else vxn
        vxn_med = float(np.median(win_v))
        vxn_std = float(np.std(win_v, ddof=1)) if len(win_v) > 2 else 4.0
        real_vol_21 = float(np.std(ndx_rets[-21:], ddof=1) * math.sqrt(252.0) * 100.0)

        corr_len = min(63, len(ndx_rets))
        r_sub = ndx_rets[-corr_len:]
        y_sub = tnx_diffs[-corr_len:]
        if np.std(r_sub) > 1e-9 and np.std(y_sub) > 1e-9:
            sy_corr = float(np.corrcoef(r_sub, y_sub)[0, 1])
        else:
            sy_corr = 0.0

        term_ratio = 1.0
        if "^VIX9D" in self.series_map and "^VIX3M" in self.series_map:
            v9 = _aligned_at_or_before(self.series_map["^VIX9D"], days[-2:], as_of)[-1]
            v3m = _aligned_at_or_before(self.series_map["^VIX3M"], days[-2:], as_of)[-1]
            if np.isfinite(v9) and np.isfinite(v3m) and v3m > 0:
                term_ratio = float(v9 / v3m)
        else:
            # Fallback proxy from VXN vs 252d median when term-structure indices are absent
            term_ratio = _clip(0.94 + 0.06 * ((vxn[-1] - vxn_med) / max(vxn_std, 1.0)), 0.75, 1.35)

        prev_vxn = float(vxn[-2]) if len(vxn) >= 2 else float(vxn[-1])
        prev_lev = _clip(18.0 / max(prev_vxn, 9.0), 0.35, 1.65)

        state = MarketState(
            as_of=days[-1].isoformat(),
            step=0,
            ndx=float(ndx[-1]),
            vxn=float(vxn[-1]),
            tnx=float(tnx[-1]),
            credit_log_ratio=float(credit[-1]),
            dxy=float(dxy[-1]),
            vxn_median_252d=vxn_med,
            vxn_std_252d=max(vxn_std, 1.0),
            realized_vol_21d=real_vol_21,
            vol_term_ratio=term_ratio,
            stock_yield_corr_63d=sy_corr,
            ndx_ret_5d=float(math.log(ndx[-1] / ndx[-6])) if len(ndx) >= 6 else 0.0,
            ndx_ret_20d=float(math.log(ndx[-1] / ndx[-21])) if len(ndx) >= 21 else 0.0,
            tnx_chg_20d_bps=float((tnx[-1] - tnx[-21]) * 100.0) if len(tnx) >= 21 else 0.0,
            credit_chg_20d=float(credit[-1] - credit[-21]) if len(credit) >= 21 else 0.0,
            dxy_ret_20d=float(math.log(dxy[-1] / dxy[-21])) if len(dxy) >= 21 else 0.0,
            prev_implied_leverage=prev_lev,
        )
        res_ret_std = float(np.std(ndx_rets[-126:], ddof=1)) if len(ndx_rets) >= 30 else 0.012
        res_vxn_std = float(np.std(vxn_diffs[-126:], ddof=1)) if len(vxn_diffs) >= 30 else 1.35
        return state, res_ret_std, res_vxn_std


def build_market_state(root: Path, as_of: AsOf) -> tuple[MarketState, float, float]:
    return HistoricalPanel.load(root).build_state(as_of)


# --------------------------------------------------------- heterogeneous actors


@dataclass
class RuleActor:
    """Micro-founded institutional reaction function parameterized by mandate."""

    spec: ActorSpec

    def act(
        self,
        state: MarketState,
        shock: PolicyShock,
        prior_steps: Sequence[SimulationStep],
    ) -> ActorAction:
        aid = self.spec.actor_id
        rate_z = shock.rate_bps / 10.0
        fx_z = shock.fx_pct / 0.50
        tariff_z = shock.tariff_pct / 15.0

        if aid == "macro_fund":
            # Regime-switching duration beta: in inflation/high-rate regimes (corr < 0 or
            # hot inflation surprise), rate hikes hurt equities; in ZLB/growth regimes,
            # moderate yield rises accompany growth optimism unless USD spikes sharply.
            inflation_regime = (
                state.stock_yield_corr_63d < -0.05
                or state.tnx >= 3.25
                or shock.inflation_z > 0.40
                or tariff_z > 0.20
            )
            rate_coeff = -0.34 if inflation_regime else +0.16
            eq = (
                rate_coeff * rate_z
                - 0.32 * fx_z
                + 0.28 * shock.growth_z
                - 0.26 * shock.inflation_z
                - 0.30 * tariff_z
                - 0.08 * (state.dxy_ret_20d / 0.02)
            )
            bond = -0.35 * rate_z - 0.30 * shock.inflation_z + 0.22 * max(0.0, -shock.growth_z)
            cred = 0.22 * shock.growth_z - 0.25 * tariff_z - 0.18 * max(0.0, rate_z)
            vol_h = 0.25 * shock.uncertainty_z + 0.20 * max(0.0, -eq)
            why = (
                f"Macro({'inflation-beta<0' if inflation_regime else 'growth-beta>0'}): "
                f"rate={shock.rate_bps:+.1f}bp, fx={shock.fx_pct:+.2f}%, eq={eq:+.2f}"
            )
            return ActorAction(aid, eq, bond, cred, vol_h, 0.70, why, model="rule").clipped()

        if aid == "vol_target_cta":
            # Target 18% annualized volatility; responds to realized/implied vol shifts
            # and trend momentum, creating mechanical multi-step deleveraging/re-leveraging.
            effective_vol = max(9.0, 0.65 * state.vxn + 0.35 * state.realized_vol_21d + 1.1 * shock.uncertainty_z)
            target_lev = _clip(18.0 / effective_vol, 0.35, 1.65)
            lev_delta = target_lev - state.prev_implied_leverage
            trend = _clip(state.ndx_ret_20d / 0.06, -1.0, 1.0)
            # When short-term panic is extreme but already peaked (term_ratio > 1.06 & no fresh shock),
            # anticipated vol mean-reversion moderates forced selling.
            reversion_relief = (
                0.14 * (state.vol_term_ratio - 1.0)
                if (state.vol_term_ratio > 1.04 and abs(shock.rate_bps) < 4.0 and shock.uncertainty_z <= 0.2)
                else 0.0
            )
            eq = 1.35 * lev_delta + 0.14 * trend + reversion_relief - 0.15 * max(0.0, shock.uncertainty_z)
            bond = 0.18 * lev_delta if state.stock_yield_corr_63d > -0.10 else -0.10 * max(0.0, rate_z)
            cred = 0.40 * lev_delta
            vol_h = -0.30 * lev_delta + 0.18 * max(0.0, shock.uncertainty_z)
            why = f"VolTarget(lev {state.prev_implied_leverage:.2f}->{target_lev:.2f}, vxn={state.vxn:.1f})"
            return ActorAction(aid, eq, bond, cred, vol_h, 0.75, why, model="rule").clipped()

        if aid == "corp_credit_hedger":
            credit_stress = -state.credit_chg_20d / 0.02
            eq = (
                -0.38 * tariff_z
                - 0.22 * max(0.0, rate_z)
                - 0.24 * fx_z
                + 0.22 * shock.growth_z
                - 0.20 * credit_stress
            )
            bond = 0.15 * max(0.0, tariff_z) - 0.18 * shock.inflation_z
            cred = -0.42 * tariff_z - 0.25 * max(0.0, rate_z) + 0.26 * shock.growth_z - 0.20 * credit_stress
            vol_h = (
                0.42 * tariff_z
                + 0.30 * max(0.0, credit_stress)
                + 0.28 * shock.uncertainty_z
                - 0.18 * max(0.0, state.vxn_z - 0.8)  # monetizes/takes profit on expensive puts when VXN very high
            )
            why = f"CorpCredit(tariff={shock.tariff_pct:.1f}%, cred20d={state.credit_chg_20d*100:+.2f}%)"
            return ActorAction(aid, eq, bond, cred, vol_h, 0.68, why, model="rule").clipped()

        # options_dealer_mm
        in_backwardation = state.vol_term_ratio > 1.03 or state.vxn_z > 1.25
        prior_flow = prior_steps[-1].clearing.net_equity_flow if prior_steps else 0.0
        if in_backwardation:
            # Short-gamma regime: hedges in the direction of shock/flow on impact, but as
            # front-end event vol crushes post-event (uncertainty <= 0.3), vega/charm unwinds VXN.
            gamma_procyclical = -0.22 * shock.uncertainty_z - 0.20 * fx_z + 0.25 * prior_flow
            vega_supply = -0.38 * (state.vol_term_ratio - 1.0) - 0.22 * max(0.0, state.vxn_z) + 0.35 * shock.uncertainty_z
            eq = gamma_procyclical + (0.14 if shock.uncertainty_z <= 0.2 else 0.0)
            vol_h = vega_supply
            why = f"DealerMM(short-gamma, term={state.vol_term_ratio:.2f}, vxn_z={state.vxn_z:+.2f})"
        else:
            # Long-gamma / calm contango: absorbs order-flow imbalances and supplies vol slowly
            eq = -0.25 * prior_flow - 0.12 * (state.ndx_ret_5d / 0.03) + 0.08 * (1.0 - state.vol_term_ratio)
            vol_h = -0.15 * state.vxn_z + 0.24 * shock.uncertainty_z + 0.10 * max(0.0, 0.92 - state.vol_term_ratio)
            why = f"DealerMM(long-gamma liquidity, term={state.vol_term_ratio:.2f})"
        bond = 0.10 * (-rate_z)
        cred = 0.08 * eq
        return ActorAction(aid, eq, bond, cred, vol_h, 0.65, why, model="rule").clipped()


@dataclass
class HomogeneousRuleActor:
    """Ablation baseline: ignores institutional mandates, regime flip, and vol term structure."""

    spec: ActorSpec

    def act(
        self,
        state: MarketState,
        shock: PolicyShock,
        prior_steps: Sequence[SimulationStep],
    ) -> ActorAction:
        # Static textbook reaction: assumes higher yields are mildly bad (-0.08), ignores
        # 63d stock-yield correlation sign flip, vol-targeting leverage, and dealer gamma.
        rate_z = shock.rate_bps / 10.0
        eq = (
            -0.08 * rate_z
            - 0.12 * (shock.fx_pct / 0.50)
            + 0.20 * shock.growth_z
            - 0.20 * shock.inflation_z
            - 0.20 * (shock.tariff_pct / 15.0)
            + 0.05 * (state.ndx_ret_20d / 0.05)
        )
        bond = -0.20 * rate_z - 0.15 * shock.inflation_z
        cred = 0.15 * eq
        vol_h = 0.20 * shock.uncertainty_z - 0.10 * eq
        return ActorAction(
            self.spec.actor_id,
            eq,
            bond,
            cred,
            vol_h,
            0.55,
            "Homogeneous average trader reaction",
            model="homogeneous",
        ).clipped()


LLM_ACTOR_SYSTEM = """You are an institutional market participant inside a structural quantitative simulation.
You NEVER predict asset prices directly. Instead, given your strict institutional mandate, the
time-frozen market state S_{t-1}, and the current policy/macro shock impulse, output ONLY a JSON
object specifying your net portfolio flow adjustments in [-1.0, +1.0]:

{
  "equity_delta": float,      // [-1.0, +1.0] net Nasdaq-100 equity exposure change (+buy / -sell)
  "bond_delta": float,        // [-1.0, +1.0] net 10Y Treasury duration demand (+buy bonds / -sell)
  "credit_delta": float,      // [-1.0, +1.0] net HYG vs TLT credit risk appetite (+risk-on / -wider spreads)
  "vol_hedge_delta": float,   // [-1.0, +1.0] net VXN volatility protection demand (+buy puts/vol / -sell vol)
  "confidence": float,        // [0.0, 1.0] conviction in this flow vector
  "rationale": "concise 1-sentence institutional mechanism (<180 chars)"
}

Calibration scale for flows:
- +/-0.05 to +/-0.20: routine rebalancing or mild surprise
- +/-0.25 to +/-0.55: significant macro/policy surprise (1-2 sigma)
- +/-0.60 to +/-1.00: extreme tail shock or forced mandate deleveraging
Stay strictly faithful to your specific mandate and constraints, even if another desk would act differently.
Return raw JSON only."""


def _parse_actor_json(actor_id: str, text: str, model_name: str, usage: Usage) -> ActorAction:
    raw = (text or "").strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"{actor_id}: LLM returned no JSON object: {raw[:200]}")
    body = json.loads(match.group(0))
    return ActorAction(
        actor_id=actor_id,
        equity_delta=float(body.get("equity_delta", 0.0)),
        bond_delta=float(body.get("bond_delta", 0.0)),
        credit_delta=float(body.get("credit_delta", 0.0)),
        vol_hedge_delta=float(body.get("vol_hedge_delta", 0.0)),
        confidence=float(body.get("confidence", 0.6)),
        rationale=str(body.get("rationale", "")),
        model=model_name,
        usage=usage,
    ).clipped()


@dataclass
class LLMActor:
    """LLM-driven institutional actor outputting structured order-flow vectors."""

    spec: ActorSpec
    llm: LLM
    temperature: float = 0.2

    def act(
        self,
        state: MarketState,
        shock: PolicyShock,
        prior_steps: Sequence[SimulationStep],
    ) -> ActorAction:
        history_summary = [
            {
                "step": s.step,
                "net_equity_flow": round(s.clearing.net_equity_flow, 3),
                "step_ndx_ret_pct": round(s.clearing.step_equity_log_return * 100.0, 2),
                "step_vxn_chg": round(s.clearing.step_vxn_change, 2),
                "my_prev_equity_delta": next(
                    (round(a.equity_delta, 3) for a in s.actions if a.actor_id == self.spec.actor_id),
                    0.0,
                ),
            }
            for s in prior_steps[-3:]
        ]
        user_payload = {
            "role_id": self.spec.actor_id,
            "role_label": self.spec.label,
            "capital_weight": self.spec.weight,
            "mandate": self.spec.mandate,
            "constraints": self.spec.constraints,
            "current_step": state.step + 1,
            "market_state": state.to_summary_dict(),
            "step_shock": shock.to_json(),
            "prior_simulation_steps": history_summary,
        }
        messages = [
            {"role": "system", "content": LLM_ACTOR_SYSTEM},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]
        reply = self.llm.chat(messages, temperature=self.temperature, max_tokens=500)
        model_label = getattr(self.llm, "model", None) or getattr(self.llm, "name", "llm")
        return _parse_actor_json(self.spec.actor_id, reply.content or "", str(model_label), reply.usage)


def build_actors(
    mode: ActorMode = "rule",
    *,
    root: Path | None = None,
    llms: Sequence[LLM] | dict[str, LLM] | None = None,
    models: Sequence[str] | None = None,
    temperature: float = 0.2,
) -> list[SimActor]:
    if mode == "rule":
        return [RuleActor(spec) for spec in ACTOR_SPECS]
    if mode == "homogeneous":
        return [HomogeneousRuleActor(spec) for spec in ACTOR_SPECS]
    if mode != "llm":
        raise ValueError(f"Unknown simulation actor mode: {mode!r}")

    resolved_clients: dict[str, LLM] = {}
    if isinstance(llms, dict):
        resolved_clients.update(llms)
    elif isinstance(llms, Sequence) and llms:
        for idx, spec in enumerate(ACTOR_SPECS):
            resolved_clients[spec.actor_id] = llms[idx % len(llms)]
    else:
        if root is None:
            raise ValueError("root Path or explicit `llms` is required when mode='llm'")
        env = merged_env(root)
        model_list = [m.strip() for m in (models or ()) if m.strip()]
        if not model_list:
            default_m = env.get("FORECAST_LOG_MODEL") or "deepseek-chat"
            model_list = [default_m]
        built = [providers.build(m, root, env=env) for m in model_list]
        for idx, spec in enumerate(ACTOR_SPECS):
            resolved_clients[spec.actor_id] = built[idx % len(built)]

    out: list[SimActor] = []
    for spec in ACTOR_SPECS:
        client = resolved_clients.get(spec.actor_id) or next(iter(resolved_clients.values()))
        out.append(LLMActor(spec=spec, llm=client, temperature=temperature))
    return out


# ------------------------------------------- explicit market-clearing state engine


def clear_market_step(
    state: MarketState,
    shock: PolicyShock,
    actions: Sequence[ActorAction],
    weights: dict[str, float] | None = None,
) -> tuple[MarketState, ClearingDiagnostics]:
    """Compute next cross-asset equilibrium S_{t+1} from actor flows and shock impulses."""
    w_map = weights or {spec.actor_id: spec.weight for spec in ACTOR_SPECS}
    total_w = sum(w_map.get(a.actor_id, 0.25) for a in actions) or 1.0

    f_eq = sum(w_map.get(a.actor_id, 0.25) * a.equity_delta for a in actions) / total_w
    f_bond = sum(w_map.get(a.actor_id, 0.25) * a.bond_delta for a in actions) / total_w
    f_cred = sum(w_map.get(a.actor_id, 0.25) * a.credit_delta for a in actions) / total_w
    f_vol = sum(w_map.get(a.actor_id, 0.25) * a.vol_hedge_delta for a in actions) / total_w

    disagreement = math.sqrt(
        sum(w_map.get(a.actor_id, 0.25) * ((a.equity_delta - f_eq) ** 2) for a in actions) / total_w
    )
    dealer_eq = next((a.equity_delta for a in actions if a.actor_id == "options_dealer_mm"), 0.0)

    # 1. Endogenous Kyle's lambda (price impact per unit institutional flow):
    #    Expands when VXN exceeds its 252d median and when the options dealer pulls liquidity.
    vol_excess = max(0.0, (state.vxn - state.vxn_median_252d) / max(state.vxn_median_252d, 12.0))
    dealer_withdrawal = max(0.0, -dealer_eq)
    kyle_lambda = 0.022 * (1.0 + 0.55 * vol_excess + 0.45 * dealer_withdrawal)

    # 2. 10Y Treasury Yield transition (^TNX, in percentage points):
    dy_pct = (shock.rate_bps / 100.0) - 0.055 * f_bond + 0.018 * shock.inflation_z
    next_tnx = max(0.05, state.tnx + dy_pct)

    # 3. US Dollar Index transition (DX-Y.NYB):
    d_log_dxy = (
        (shock.fx_pct / 100.0)
        + 0.012 * dy_pct
        + 0.00018 * shock.tariff_pct
        - 0.0035 * f_eq
    )
    next_dxy = state.dxy * math.exp(_clip(d_log_dxy, -0.06, 0.06))

    # 4. Credit Appetite Log-Ratio transition (ln(HYG / TLT)):
    d_credit = (
        0.0065 * f_cred
        - 0.014 * max(0.0, dy_pct)
        - 0.00025 * shock.tariff_pct
    )
    next_credit = state.credit_log_ratio + _clip(d_credit, -0.08, 0.08)

    # 5. Equity Log-Return & Index transition (^NDX):
    #    Combines cleared institutional order flow (lambda * F_eq), credit transmission,
    #    FX headwind, and regime-dependent direct duration valuation impact.
    direct_duration_beta = (
        -0.028
        if (state.stock_yield_corr_63d < -0.05 or state.tnx >= 3.25 or shock.inflation_z > 0.4)
        else +0.012
    )
    eq_log_ret = _clip(
        kyle_lambda * f_eq
        + 0.35 * d_credit
        - 0.24 * d_log_dxy
        + direct_duration_beta * dy_pct,
        -0.12,
        0.12,
    )
    next_ndx = state.ndx * math.exp(eq_log_ret)

    # 6. Implied Volatility transition (^VXN):
    #    Ornstein-Uhlenbeck pull toward anchor + asymmetric equity leverage effect
    #    + net protection demand + cross-actor disagreement + term-structure relaxation.
    anchor_v = 0.75 * state.vxn_median_252d + 0.25 * state.realized_vol_21d
    ou_pull = 0.075 * (anchor_v - state.vxn)
    # Normal CBOE VIX9D/VIX3M contango sits near 0.93; backwardation (>0.93) unwinds once event passes
    term_relax = -1.65 * (state.vol_term_ratio - 0.93) if shock.uncertainty_z <= 0.25 else 0.0
    lev_effect = -56.0 * eq_log_ret
    excess_disagreement = max(0.0, disagreement - 0.18)
    dvxn = _clip(
        ou_pull
        + term_relax
        + lev_effect
        + 1.60 * f_vol
        + 0.75 * excess_disagreement
        + 0.95 * shock.uncertainty_z,
        -12.0,
        18.0,
    )
    next_vxn = max(9.0, state.vxn + dvxn)

    # Update rolling features consistently along the simulated path
    next_term = _clip(
        1.0 + 0.68 * (state.vol_term_ratio - 1.0) + 0.028 * dvxn,
        0.75,
        1.35,
    )
    next_real_vol = max(
        6.0,
        math.sqrt((20.0 / 21.0) * ((state.realized_vol_21d / 100.0) ** 2) + (252.0 / 21.0) * (eq_log_ret ** 2))
        * 100.0,
    )
    next_lev = _clip(18.0 / max(next_vxn, 9.0), 0.35, 1.65)

    next_state = MarketState(
        as_of=state.as_of,
        step=state.step + 1,
        ndx=next_ndx,
        vxn=next_vxn,
        tnx=next_tnx,
        credit_log_ratio=next_credit,
        dxy=next_dxy,
        vxn_median_252d=state.vxn_median_252d,
        vxn_std_252d=state.vxn_std_252d,
        realized_vol_21d=next_real_vol,
        vol_term_ratio=next_term,
        stock_yield_corr_63d=state.stock_yield_corr_63d,
        ndx_ret_5d=0.80 * state.ndx_ret_5d + eq_log_ret,
        ndx_ret_20d=(19.0 / 20.0) * state.ndx_ret_20d + eq_log_ret,
        tnx_chg_20d_bps=(19.0 / 20.0) * state.tnx_chg_20d_bps + dy_pct * 100.0,
        credit_chg_20d=(19.0 / 20.0) * state.credit_chg_20d + d_credit,
        dxy_ret_20d=(19.0 / 20.0) * state.dxy_ret_20d + d_log_dxy,
        prev_implied_leverage=next_lev,
    )
    diag = ClearingDiagnostics(
        net_equity_flow=f_eq,
        net_bond_flow=f_bond,
        net_credit_flow=f_cred,
        net_vol_hedge_flow=f_vol,
        actor_disagreement=disagreement,
        kyle_lambda=kyle_lambda,
        step_equity_log_return=eq_log_ret,
        step_vxn_change=dvxn,
        step_tnx_change_bps=dy_pct * 100.0,
    )
    return next_state, diag


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _compute_distribution(
    initial: MarketState,
    steps: Sequence[SimulationStep],
    res_ret_step: float,
    res_vxn_step: float,
) -> ProbabilisticDistribution:
    """Closed-form Gaussian state-space uncertainty calibrated to trailing residual scale."""
    h = max(1, len(steps))
    final = steps[-1].state_after if steps else initial
    mu_ret = math.log(final.ndx / initial.ndx)
    mu_dvxn = final.vxn - initial.vxn

    mean_disagreement = (
        statistics.fmean(s.clearing.actor_disagreement for s in steps) if steps else 0.0
    )
    # Total predictive variance combines market noise + structural specification uncertainty;
    # point displacement is shrinkage-damped when mapping to calibrated probabilities.
    inflation_factor = 1.35 + 0.40 * mean_disagreement
    sigma_ret = max(0.006, res_ret_step * math.sqrt(h) * inflation_factor)
    sigma_vxn = max(0.85, res_vxn_step * math.sqrt(h) * inflation_factor)
    damping = 0.48 if h == 1 else 0.32

    p_ndx_up = _clip(_normal_cdf((damping * mu_ret) / sigma_ret), 0.02, 0.98)
    p_vxn_up = _clip(_normal_cdf((damping * mu_dvxn) / sigma_vxn), 0.02, 0.98)
    p_dd_2pct = _clip(_normal_cdf((math.log(0.98) - damping * mu_ret) / sigma_ret), 0.01, 0.99)

    z10, z90 = -1.28155, 1.28155
    return ProbabilisticDistribution(
        p_ndx_up=p_ndx_up,
        p_vxn_up=p_vxn_up,
        p_ndx_drawdown_gt_2pct=p_dd_2pct,
        ndx_return_p10=math.exp(mu_ret + z10 * sigma_ret) - 1.0,
        ndx_return_p50=math.exp(mu_ret) - 1.0,
        ndx_return_p90=math.exp(mu_ret + z90 * sigma_ret) - 1.0,
        vxn_terminal_p10=max(9.0, initial.vxn + mu_dvxn + z10 * sigma_vxn),
        vxn_terminal_p50=max(9.0, initial.vxn + mu_dvxn),
        vxn_terminal_p90=max(9.0, initial.vxn + mu_dvxn + z90 * sigma_vxn),
        residual_vol_step=res_ret_step,
        residual_vxn_step=res_vxn_step,
    )


def simulate(
    state: MarketState,
    shock: PolicyShock,
    *,
    as_of: AsOf | None = None,
    actors: Sequence[SimActor] | None = None,
    mode: ActorMode = "rule",
    steps: int = 5,
    res_ret_step: float = 0.012,
    res_vxn_step: float = 1.35,
    parallel_actors: bool = True,
) -> SimulationResult:
    """Run an H-step closed-loop policy simulation starting from `state`."""
    if steps < 1 or steps > 30:
        raise ValueError("steps must be in [1, 30]")
    gate = as_of or AsOf.at_session_close(date.fromisoformat(state.as_of))
    gate.check(date.fromisoformat(state.as_of), "sim:initial_state")

    roster = list(actors) if actors is not None else build_actors(mode)
    weights = {a.spec.actor_id: a.spec.weight for a in roster}

    history: list[SimulationStep] = []
    cur = state
    total_usage = Usage()

    for step_idx in range(1, steps + 1):
        step_shock = shock.at_step(step_idx)
        if parallel_actors and mode == "llm" and len(roster) > 1:
            with ThreadPoolExecutor(max_workers=len(roster)) as pool:
                step_actions = tuple(
                    pool.map(lambda actor: actor.act(cur, step_shock, history), roster)
                )
        else:
            step_actions = tuple(actor.act(cur, step_shock, history) for actor in roster)

        for action in step_actions:
            total_usage = total_usage + action.usage

        next_state, diag = clear_market_step(cur, step_shock, step_actions, weights=weights)
        rec = SimulationStep(
            step=step_idx,
            shock=step_shock,
            actions=step_actions,
            clearing=diag,
            state_after=next_state,
        )
        history.append(rec)
        cur = next_state

    dist = _compute_distribution(state, history, res_ret_step, res_vxn_step)
    return SimulationResult(
        as_of=gate,
        mode=mode,
        initial_state=state,
        shock=shock,
        steps=tuple(history),
        distribution=dist,
        usage=total_usage,
    )


# ------------------------------------------- historical macro shock backtesting


def _event_dates_aligned(
    root: Path,
    panel: HistoricalPanel,
    kinds: Sequence[str] = ("fomc", "cpi", "nfp"),
    start: date | None = None,
    end: date | None = None,
) -> tuple[list[date], dict[date, int], list[tuple[date, str]]]:
    """Return common cross-asset trading calendar and valid historical macro event dates."""
    cal = events.load(root)
    if not cal:
        raise RuntimeError("data/events.json is missing; run `forecast-log events` or provide calendar.")

    common_sets = [{b.date for b in panel.series_map[sym].bars} for sym in REQUIRED_SYMBOLS]
    common_dates = sorted(set.intersection(*common_sets))
    idx_map = {d: i for i, d in enumerate(common_dates)}

    seen: dict[date, str] = {}
    # Optional series can start much later than the required ones (^VIX9D starts 2011-01-03)
    # and `build_state` needs 30 bars of each; skip those warmup episodes instead of aborting.
    warmup: list[date] = []
    for sym in OPTIONAL_SYMBOLS:
        series = panel.series_map.get(sym)
        if series is None or len(series.bars) < 30:
            continue
        warmup.append(series.bars[29].date)
    warm_start = max(warmup) if warmup else None
    # Priority order when two macro releases share a date: fomc > cpi > nfp
    for kind in ("nfp", "cpi", "fomc"):
        if kind not in kinds:
            continue
        for ev_day in cal.get(kind, []):
            if ev_day not in idx_map:
                continue
            idx = idx_map[ev_day]
            if idx < 292 or idx >= len(common_dates) - 5:
                continue
            if warm_start is not None and ev_day < warm_start:
                continue
            if start and ev_day < start:
                continue
            if end and ev_day > end:
                continue
            seen[ev_day] = kind

    episodes = sorted(seen.items(), key=lambda pair: pair[0])
    return common_dates, idx_map, episodes


def _score_prob_table(pairs: list[tuple[float, int]]) -> dict:
    if not pairs:
        return {"n": 0, "brier": None, "log_loss": None, "ece": None, "hit_direction": None}
    hit = statistics.fmean(1.0 if (p >= 0.5) == (y == 1) else 0.0 for p, y in pairs)
    return {
        "n": len(pairs),
        "brier": round(brier(pairs), 4),
        "log_loss": round(log_loss(pairs), 4),
        "ece": round(ece(pairs), 4),
        "hit_direction": round(hit, 4),
        "mean_probability": round(statistics.fmean(p for p, _ in pairs), 4),
        "observed_rate": round(statistics.fmean(y for _, y in pairs), 4),
        "reliability": reliability(pairs),
    }


def backtest_macro_shocks(
    root: Path,
    *,
    kinds: Sequence[str] = ("fomc", "cpi", "nfp"),
    start: date | None = None,
    end: date | None = None,
    holdout: date = date(2021, 1, 1),
    horizon: int = 5,
    limit: int | None = None,
    include_llm: bool = False,
    llm_models: Sequence[str] | None = None,
    llms: Sequence[LLM] | dict[str, LLM] | None = None,
) -> dict:
    """Backtest the simulation engine across historical FOMC / CPI / NFP macro events.

    Evaluates two leak-checked empirical tracks:
    1. `shock_transmission` (t-1 -> t0): State frozen strictly at pre-event close `t-1`
       (`AsOf.at_session_close(t-1)`). Given the observed event-day rate/FX impulse
       (`d_TNX`, `d_DXY`) without touching `t0` equity/credit/vol prices, computes the
       contemporaneous equilibrium equity return and VXN shift. Tests whether regime-aware
       heterogeneous actors outperform homogeneous actors and expanding-window OLS when
       macro regimes flip (ZLB growth pre-2021 vs. inflation discount-rate 2021+).
    2. `ex_ante_5d_propagation` (t0 -> t0+5): State frozen strictly at event-day close `t0`
       (`AsOf.at_session_close(t0)`), zero post-`t0` inputs. Simulates 5-session endogenous
       relaxation / deleveraging and scores `P(VXN_{t0+5} > VXN_{t0})` and
       `P(NDX_{t0+5} > NDX_{t0})` against `homogeneous_ablation`, `analogue_model`,
       `expanding_base_rate`, and `constant_0.5`.
    """
    panel = HistoricalPanel.load(root)
    common_dates, idx_map, episodes = _event_dates_aligned(
        root, panel, kinds=kinds, start=start, end=end
    )
    if limit is not None and len(episodes) > limit:
        step = len(episodes) / limit
        episodes = [episodes[int(i * step)] for i in range(limit)]

    ndx_map = {b.date: b.close for b in panel.series_map["^NDX"].bars}
    vxn_map = {b.date: b.close for b in panel.series_map["^VXN"].bars}
    tnx_map = {b.date: b.close for b in panel.series_map["^TNX"].bars}
    dxy_map = {b.date: b.close for b in panel.series_map["DX-Y.NYB"].bars}

    rule_actors = build_actors("rule")
    homo_actors = build_actors("homogeneous")
    llm_actors = (
        build_actors("llm", root=root, llms=llms, models=llm_models) if include_llm else None
    )

    # Track 1 accumulators: Shock transmission (t-1 -> t0)
    t1_models = ["multi_agent_rule", "homogeneous_ablation", "expanding_ols", "constant_0.5"]
    if llm_actors is not None:
        t1_models.insert(0, "multi_agent_llm")
    t1_probs: dict[str, list[tuple[float, int]]] = {m: [] for m in t1_models}
    t1_holdout_probs: dict[str, list[tuple[float, int]]] = {m: [] for m in t1_models}
    t1_pred_ret: dict[str, list[tuple[float, float, bool]]] = {m: [] for m in t1_models if m != "constant_0.5"}

    # Expanding-window OLS state for Track 1 (strictly trained on prior events < idx)
    prior_X: list[list[float]] = []
    prior_y: list[float] = []

    # Track 2 accumulators: Pure Ex-Ante 5-Day Post-Shock Propagation (t0 -> t0+h)
    t2_models = [
        "multi_agent_rule",
        "homogeneous_ablation",
        "analogue_model",
        "expanding_base_rate",
        "constant_0.5",
    ]
    if llm_actors is not None:
        t2_models.insert(0, "multi_agent_llm")
    t2_vxn_probs: dict[str, list[tuple[float, int]]] = {m: [] for m in t2_models}
    t2_ndx_probs: dict[str, list[tuple[float, int]]] = {m: [] for m in t2_models}
    prior_vxn_outcomes: list[int] = []
    prior_ndx_outcomes: list[int] = []

    llm_usage = Usage()
    sample_records: list[dict] = []

    for ev_day, kind in episodes:
        idx = idx_map[ev_day]
        prev_day = common_dates[idx - 1]
        target_day = common_dates[idx + horizon]
        in_holdout = ev_day >= holdout

        # ---------------- Track 1: Conditioned Shock Transmission (t-1 -> t0)
        gate_pre = AsOf.at_session_close(prev_day)
        state_pre, res_ret_pre, res_vxn_pre = panel.build_state(gate_pre)

        dy_bps = (tnx_map[ev_day] - tnx_map[prev_day]) * 100.0
        dd_pct = math.log(dxy_map[ev_day] / dxy_map[prev_day]) * 100.0
        actual_r0 = math.log(ndx_map[ev_day] / ndx_map[prev_day])
        actual_up0 = 1 if ndx_map[ev_day] > ndx_map[prev_day] else 0

        infl_z = _clip(dy_bps / 10.0, -2.5, 2.5) if kind == "cpi" else 0.0
        grow_z = _clip(dy_bps / 10.0, -2.5, 2.5) if kind == "nfp" else 0.0
        shock_t0 = PolicyShock(
            kind=kind,
            title=f"Historical {kind.upper()} macro impulse on {ev_day.isoformat()}",
            rate_bps=dy_bps,
            fx_pct=dd_pct,
            inflation_z=infl_z,
            growth_z=grow_z,
            uncertainty_z=abs(dy_bps) / 14.0,
        )

        sim_rule_t1 = simulate(
            state_pre,
            shock_t0,
            as_of=gate_pre,
            actors=rule_actors,
            mode="rule",
            steps=1,
            res_ret_step=res_ret_pre,
            res_vxn_step=res_vxn_pre,
        )
        sim_homo_t1 = simulate(
            state_pre,
            shock_t0,
            as_of=gate_pre,
            actors=homo_actors,
            mode="homogeneous",
            steps=1,
            res_ret_step=res_ret_pre,
            res_vxn_step=res_vxn_pre,
        )

        # Expanding OLS baseline (uses only prior events!)
        if len(prior_X) >= 20:
            X_mat = np.array(prior_X, dtype=float)
            y_vec = np.array(prior_y, dtype=float)
            beta, *_ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
            ols_r0 = float(np.dot([1.0, dy_bps, dd_pct], beta))
        else:
            ols_r0 = 0.0
        ols_p0 = _clip(_normal_cdf(ols_r0 / max(res_ret_pre, 0.005)), 0.02, 0.98)

        prior_X.append([1.0, dy_bps, dd_pct])
        prior_y.append(actual_r0)

        t1_entries = {
            "multi_agent_rule": (sim_rule_t1.distribution.p_ndx_up, sim_rule_t1.cumulative_ndx_log_return),
            "homogeneous_ablation": (sim_homo_t1.distribution.p_ndx_up, sim_homo_t1.cumulative_ndx_log_return),
            "expanding_ols": (ols_p0, ols_r0),
        }
        if llm_actors is not None:
            sim_llm_t1 = simulate(
                state_pre,
                shock_t0,
                as_of=gate_pre,
                actors=llm_actors,
                mode="llm",
                steps=1,
                res_ret_step=res_ret_pre,
                res_vxn_step=res_vxn_pre,
            )
            llm_usage = llm_usage + sim_llm_t1.usage
            t1_entries["multi_agent_llm"] = (
                sim_llm_t1.distribution.p_ndx_up,
                sim_llm_t1.cumulative_ndx_log_return,
            )

        for name, (prob, pred_r) in t1_entries.items():
            t1_probs[name].append((prob, actual_up0))
            if in_holdout:
                t1_holdout_probs[name].append((prob, actual_up0))
            t1_pred_ret[name].append((pred_r, actual_r0, in_holdout))
        t1_probs["constant_0.5"].append((0.5, actual_up0))
        if in_holdout:
            t1_holdout_probs["constant_0.5"].append((0.5, actual_up0))

        # ---------------- Track 2: Pure Ex-Ante Post-Event 5d Propagation (t0 -> t0+h)
        gate_t0 = AsOf.at_session_close(ev_day)
        state_t0, res_ret_t0, res_vxn_t0 = panel.build_state(gate_t0)
        headlines_t0 = gated_headlines_for_day(root, ev_day, gate_t0, limit=5)

        # Event-day observed shock (already printed at t0 close!) initializes decaying aftershocks
        dv0 = vxn_map[ev_day] - vxn_map[prev_day]
        # Once a scheduled release prints at t0, pre-event uncertainty resolves (vol crush when dv0 > 0)
        aftershock = PolicyShock(
            kind=kind,
            title=f"Post-{kind.upper()} propagation from {ev_day.isoformat()}",
            rate_bps=dy_bps * 0.08,
            fx_pct=dd_pct * 0.08,
            inflation_z=(_clip(dy_bps / 12.0, -2.0, 2.0) if kind == "cpi" else 0.0) * 0.15,
            growth_z=(_clip(actual_r0 / 0.012, -2.0, 2.0) if kind == "nfp" else 0.0) * 0.15,
            uncertainty_z=-0.22 * _clip(dv0 / 2.5, -1.0, 1.5),
            persistence=0.35,
            headlines=headlines_t0,
        )

        sim_rule_t2 = simulate(
            state_t0,
            aftershock,
            as_of=gate_t0,
            actors=rule_actors,
            mode="rule",
            steps=horizon,
            res_ret_step=res_ret_t0,
            res_vxn_step=res_vxn_t0,
        )
        sim_homo_t2 = simulate(
            state_t0,
            aftershock,
            as_of=gate_t0,
            actors=homo_actors,
            mode="homogeneous",
            steps=horizon,
            res_ret_step=res_ret_t0,
            res_vxn_step=res_vxn_t0,
        )

        # Point-in-time AnalogueModel baselines at t0
        _, vxn_vis = _visible_closes(panel.series_map["^VXN"], gate_t0)
        _, ndx_vis = _visible_closes(panel.series_map["^NDX"], gate_t0)
        ana_vxn_p = AnalogModel(vxn_vis).forecast(len(vxn_vis) - 1, horizon, float(vxn_vis[-1])).probability
        ana_ndx_p = AnalogModel(ndx_vis).forecast(len(ndx_vis) - 1, horizon, float(ndx_vis[-1])).probability

        base_vxn_p = statistics.fmean(prior_vxn_outcomes) if len(prior_vxn_outcomes) >= 15 else 0.5
        base_ndx_p = statistics.fmean(prior_ndx_outcomes) if len(prior_ndx_outcomes) >= 15 else 0.5

        actual_vxn_up5 = 1 if vxn_map[target_day] > vxn_map[ev_day] else 0
        actual_ndx_up5 = 1 if ndx_map[target_day] > ndx_map[ev_day] else 0
        prior_vxn_outcomes.append(actual_vxn_up5)
        prior_ndx_outcomes.append(actual_ndx_up5)

        t2_vxn_probs["multi_agent_rule"].append((sim_rule_t2.distribution.p_vxn_up, actual_vxn_up5))
        t2_vxn_probs["homogeneous_ablation"].append((sim_homo_t2.distribution.p_vxn_up, actual_vxn_up5))
        t2_vxn_probs["analogue_model"].append((ana_vxn_p, actual_vxn_up5))
        t2_vxn_probs["expanding_base_rate"].append((base_vxn_p, actual_vxn_up5))
        t2_vxn_probs["constant_0.5"].append((0.5, actual_vxn_up5))

        t2_ndx_probs["multi_agent_rule"].append((sim_rule_t2.distribution.p_ndx_up, actual_ndx_up5))
        t2_ndx_probs["homogeneous_ablation"].append((sim_homo_t2.distribution.p_ndx_up, actual_ndx_up5))
        t2_ndx_probs["analogue_model"].append((ana_ndx_p, actual_ndx_up5))
        t2_ndx_probs["expanding_base_rate"].append((base_ndx_p, actual_ndx_up5))
        t2_ndx_probs["constant_0.5"].append((0.5, actual_ndx_up5))

        if llm_actors is not None:
            sim_llm_t2 = simulate(
                state_t0,
                aftershock,
                as_of=gate_t0,
                actors=llm_actors,
                mode="llm",
                steps=horizon,
                res_ret_step=res_ret_t0,
                res_vxn_step=res_vxn_t0,
            )
            llm_usage = llm_usage + sim_llm_t2.usage
            t2_vxn_probs["multi_agent_llm"].append((sim_llm_t2.distribution.p_vxn_up, actual_vxn_up5))
            t2_ndx_probs["multi_agent_llm"].append((sim_llm_t2.distribution.p_ndx_up, actual_ndx_up5))

        if len(sample_records) < 12 or llm_actors is not None:
            sample_records.append(
                {
                    "event_date": ev_day.isoformat(),
                    "kind": kind,
                    "regime": state_pre.regime_label,
                    "rate_bps": round(dy_bps, 1),
                    "fx_pct": round(dd_pct, 2),
                    "pred_r0_pct": round(sim_rule_t1.cumulative_ndx_log_return * 100.0, 2),
                    "actual_r0_pct": round(actual_r0 * 100.0, 2),
                    "p_vxn_up_5d": round(sim_rule_t2.distribution.p_vxn_up, 3),
                    "actual_vxn_up_5d": actual_vxn_up5,
                }
            )

    def _ret_diagnostics(rows: list[tuple[float, float, bool]], holdout_only: bool = False) -> dict:
        pts = [(p, a) for p, a, h in rows if (h or not holdout_only)]
        if len(pts) < 3:
            return {"mae_pct": None, "corr": None}
        preds = np.array([p for p, _ in pts], dtype=float)
        acts = np.array([a for _, a in pts], dtype=float)
        mae = float(np.mean(np.abs(preds - acts))) * 100.0
        corr = float(np.corrcoef(preds, acts)[0, 1]) if np.std(preds) > 1e-9 and np.std(acts) > 1e-9 else 0.0
        return {"mae_pct": round(mae, 3), "corr": round(corr, 4)}

    track1_summary = {}
    for name in t1_models:
        base_metrics = _score_prob_table(t1_probs[name])
        hold_metrics = _score_prob_table(t1_holdout_probs[name])
        if name in t1_pred_ret:
            base_metrics.update(_ret_diagnostics(t1_pred_ret[name], holdout_only=False))
            hold_metrics.update(_ret_diagnostics(t1_pred_ret[name], holdout_only=True))
        track1_summary[name] = {"full_sample": base_metrics, "holdout_2021_plus": hold_metrics}

    return {
        "n_episodes": len(episodes),
        "first_event": episodes[0][0].isoformat() if episodes else None,
        "last_event": episodes[-1][0].isoformat() if episodes else None,
        "holdout_from": holdout.isoformat(),
        "horizon_sessions": horizon,
        "kinds": list(kinds),
        "llm_usage": llm_usage.to_json() if include_llm else None,
        "track1_shock_transmission": track1_summary,
        "track2_ex_ante_vxn_5d": {m: _score_prob_table(t2_vxn_probs[m]) for m in t2_models},
        "track2_ex_ante_ndx_5d": {m: _score_prob_table(t2_ndx_probs[m]) for m in t2_models},
        "samples": sample_records[:20],
    }


# --------------------------------------------------------- formatting & CLI


def render_simulation(result: SimulationResult) -> str:
    init = result.initial_state
    fin = result.final_state
    dist = result.distribution
    lines = [
        f"Policy Simulation ({result.mode}) | as-of {init.as_of} | regime={init.regime_label}",
        f"Shock [{result.shock.kind}]: {result.shock.title or '(numeric impulse)'}",
        (
            f"Impulse: rate={result.shock.rate_bps:+.1f}bp, fx={result.shock.fx_pct:+.2f}%, "
            f"infl_z={result.shock.inflation_z:+.2f}, growth_z={result.shock.growth_z:+.2f}, "
            f"tariff={result.shock.tariff_pct:.1f}%, unc_z={result.shock.uncertainty_z:+.2f}"
        ),
        "",
        f"{'Step':<5}{'NetEq':>8}{'NetVolH':>9}{'Lambda':>9}{'NDX':>11}{'dNDX%':>8}{'VXN':>8}{'TNX%':>8}{'HYG/TLT':>9}",
        "-" * 75,
        (
            f"{0:<5}{'—':>8}{'—':>9}{'—':>9}{init.ndx:>11.2f}{'0.00%':>8}"
            f"{init.vxn:>8.2f}{init.tnx:>8.3f}{init.credit_log_ratio:>9.4f}"
        ),
    ]
    for step in result.steps:
        st = step.state_after
        cl = step.clearing
        ret_pct = (math.exp(cl.step_equity_log_return) - 1.0) * 100.0
        lines.append(
            f"{step.step:<5}{cl.net_equity_flow:>+8.3f}{cl.net_vol_hedge_flow:>+9.3f}"
            f"{cl.kyle_lambda:>9.4f}{st.ndx:>11.2f}{ret_pct:>+7.2f}%"
            f"{st.vxn:>8.2f}{st.tnx:>8.3f}{st.credit_log_ratio:>9.4f}"
        )

    lines.append("")
    lines.append("Step 1 Heterogeneous Actor Decisions:")
    if result.steps:
        for action in result.steps[0].actions:
            lines.append(
                f"  - {action.actor_id:<20} [{action.model}] "
                f"eq={action.equity_delta:+.3f} bond={action.bond_delta:+.3f} "
                f"cred={action.credit_delta:+.3f} volH={action.vol_hedge_delta:+.3f} | {action.rationale}"
            )

    cum_pct = (math.exp(result.cumulative_ndx_log_return) - 1.0) * 100.0
    lines.extend(
        [
            "",
            (
                f"Terminal ({len(result.steps)}d): NDX {init.ndx:.2f} -> {fin.ndx:.2f} ({cum_pct:+.2f}%), "
                f"VXN {init.vxn:.2f} -> {fin.vxn:.2f} ({result.cumulative_vxn_change:+.2f})"
            ),
            (
                f"Calibrated Probabilities: P(NDX up)={dist.p_ndx_up:.1%}, "
                f"P(VXN up)={dist.p_vxn_up:.1%}, P(NDX <-2%)={dist.p_ndx_drawdown_gt_2pct:.1%}"
            ),
            (
                f"80% Band: NDX return [{dist.ndx_return_p10*100:+.2f}%, {dist.ndx_return_p90*100:+.2f}%], "
                f"VXN [{dist.vxn_terminal_p10:.2f}, {dist.vxn_terminal_p90:.2f}]"
            ),
        ]
    )
    if result.usage.prompt_tokens > 0:
        lines.append(
            f"LLM Usage: {result.usage.prompt_tokens} in / {result.usage.completion_tokens} out tokens, "
            f"${result.usage.cost_usd:.5f}, {result.usage.latency_s:.2f}s"
        )
    return "\n".join(lines)


def render_backtest(report: dict) -> str:
    lines = [
        (
            f"Macro Shock Simulation Backtest: {report['n_episodes']} events "
            f"({report['first_event']} -> {report['last_event']}, kinds={','.join(report['kinds'])})"
        ),
        "",
        "1. Shock Transmission (t-1 -> t0, given rate/FX impulse; regime-flip holdout >= 2021-01-01):",
        f"{'Predictor':<24}{'Full Brier':>11}{'Full Hit':>10}{'Full Corr':>10}{'2021+ Brier':>13}{'2021+ Hit':>11}{'2021+ Corr':>11}",
        "-" * 90,
    ]
    for name, entry in report["track1_shock_transmission"].items():
        full = entry["full_sample"]
        hold = entry["holdout_2021_plus"]
        fc = f"{full['corr']:+.3f}" if full.get("corr") is not None else "—"
        hc = f"{hold['corr']:+.3f}" if hold.get("corr") is not None else "—"
        lines.append(
            f"{name:<24}{full['brier']:>11.4f}{full['hit_direction']:>10.1%}{fc:>10}"
            f"{hold['brier']:>13.4f}{hold['hit_direction']:>11.1%}{hc:>11}"
        )

    lines.extend(
        [
            "",
            f"2. Pure Ex-Ante {report['horizon_sessions']}d Volatility Propagation P(VXN_{{t+{report['horizon_sessions']}}} > VXN_t) (strictly gated at t0):",
            f"{'Predictor':<24}{'Brier':>10}{'LogLoss':>10}{'ECE':>9}{'HitRate':>10}{'MeanProb':>10}{'ObsRate':>10}",
            "-" * 83,
        ]
    )
    for name, row in report["track2_ex_ante_vxn_5d"].items():
        lines.append(
            f"{name:<24}{row['brier']:>10.4f}{row['log_loss']:>10.4f}{row['ece']:>9.3f}"
            f"{row['hit_direction']:>10.1%}{row['mean_probability']:>10.3f}{row['observed_rate']:>10.3f}"
        )

    lines.extend(
        [
            "",
            f"3. Pure Ex-Ante {report['horizon_sessions']}d Equity Drift P(NDX_{{t+{report['horizon_sessions']}}} > NDX_t) (strictly gated at t0):",
            f"{'Predictor':<24}{'Brier':>10}{'LogLoss':>10}{'ECE':>9}{'HitRate':>10}{'MeanProb':>10}{'ObsRate':>10}",
            "-" * 83,
        ]
    )
    for name, row in report["track2_ex_ante_ndx_5d"].items():
        lines.append(
            f"{name:<24}{row['brier']:>10.4f}{row['log_loss']:>10.4f}{row['ece']:>9.3f}"
            f"{row['hit_direction']:>10.1%}{row['mean_probability']:>10.3f}{row['observed_rate']:>10.3f}"
        )

    if report.get("llm_usage"):
        u = report["llm_usage"]
        lines.append(
            f"\nLLM Backtest Cost: ${u['cost_usd']:.4f} ({u['prompt_tokens']} prompt / {u['completion_tokens']} completion tokens)"
        )
    return "\n".join(lines)


def sim_command(
    root: Path,
    *,
    scenario: str = "tariff_25",
    day: str | None = None,
    steps: int = 5,
    mode: ActorMode = "rule",
    models: str | None = None,
    rate_bps: float | None = None,
    fx_pct: float | None = None,
    inflation_z: float | None = None,
    growth_z: float | None = None,
    tariff_pct: float | None = None,
    uncertainty_z: float | None = None,
    title: str | None = None,
    out: str | None = "artifacts/sim/latest_scenario.json",
) -> SimulationResult:
    panel = HistoricalPanel.load(root)
    last_day = date.fromisoformat(day) if day else panel.series_map["^NDX"].bars[-1].date
    as_of = AsOf.at_session_close(last_day)
    state, res_ret, res_vxn = panel.build_state(as_of)

    base = PRESET_SHOCKS.get(scenario, PolicyShock(kind="custom", title=scenario))
    shock = PolicyShock(
        kind=base.kind,
        title=title if title is not None else base.title,
        rate_bps=base.rate_bps if rate_bps is None else rate_bps,
        fx_pct=base.fx_pct if fx_pct is None else fx_pct,
        inflation_z=base.inflation_z if inflation_z is None else inflation_z,
        growth_z=base.growth_z if growth_z is None else growth_z,
        tariff_pct=base.tariff_pct if tariff_pct is None else tariff_pct,
        uncertainty_z=base.uncertainty_z if uncertainty_z is None else uncertainty_z,
        persistence=base.persistence,
        headlines=gated_headlines_for_day(root, last_day, as_of, limit=5),
    )

    model_list = [m.strip() for m in (models or "").split(",") if m.strip()]
    actors = build_actors(mode, root=root, models=model_list or None)
    result = simulate(
        state,
        shock,
        as_of=as_of,
        actors=actors,
        mode=mode,
        steps=steps,
        res_ret_step=res_ret,
        res_vxn_step=res_vxn,
    )
    print(render_simulation(result))
    if out:
        target = root / out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n")
        print(f"\nSaved simulation trace -> {target}")
    return result


def sim_backtest_command(
    root: Path,
    *,
    kinds: str = "fomc,cpi,nfp",
    horizon: int = 5,
    limit: int | None = None,
    with_llm: bool = False,
    models: str | None = None,
    out: str = "artifacts/sim/backtest_report.json",
) -> dict:
    kind_tuple = tuple(k.strip().lower() for k in kinds.split(",") if k.strip())
    model_list = [m.strip() for m in (models or "").split(",") if m.strip()]
    report = backtest_macro_shocks(
        root,
        kinds=kind_tuple,
        horizon=horizon,
        limit=limit,
        include_llm=with_llm,
        llm_models=model_list or None,
    )
    print(render_backtest(report))
    if out:
        target = root / out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nSaved backtest report -> {target}")
    return report


# --------------------------------------------------------- Web UI API helpers

import threading as _threading
from typing import Iterator

_PANEL_LOCK = _threading.Lock()
_PANEL_MEM_CACHE: dict[str, tuple[float, HistoricalPanel]] = {}

PRESET_UI_META: tuple[dict, ...] = (
    {"id": "tariff_25", "short_label": "+25% 半导体关税", "badge": "TARIFF"},
    {"id": "fomc_hawkish_25", "short_label": "FOMC 鹰派 +25bp", "badge": "FOMC"},
    {"id": "fomc_dovish_cut", "short_label": "FOMC 鸽派降息 -50bp", "badge": "FOMC"},
    {"id": "cpi_hot_surprise", "short_label": "核心 CPI 超预期", "badge": "CPI"},
    {"id": "nfp_goldilocks", "short_label": "非农金发姑娘", "badge": "NFP"},
    {"id": "credit_crunch", "short_label": "信用利差危机", "badge": "CREDIT"},
)


def cached_panel(root: Path) -> HistoricalPanel:
    """Memory-cached HistoricalPanel invalidated automatically when `forecast.db` updates."""
    from forecast_log.db import db_path

    resolved = str(root.resolve())
    dbp = db_path(root)
    mtime = dbp.stat().st_mtime if dbp.is_file() else 0.0
    with _PANEL_LOCK:
        hit = _PANEL_MEM_CACHE.get(resolved)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        panel = HistoricalPanel.load(root)
        _PANEL_MEM_CACHE[resolved] = (mtime, panel)
        return panel


def _realized_overlay(panel: HistoricalPanel, anchor_day: date, steps: int) -> dict | None:
    """Return actual realised post-anchor ^NDX and ^VXN closes ONLY for UI visual comparison."""
    ndx_bars = panel.series_map["^NDX"].bars
    vxn_map = {b.date: b.close for b in panel.series_map["^VXN"].bars}
    tnx_map = {b.date: b.close for b in panel.series_map["^TNX"].bars}

    anchor_idx = None
    for i, bar in enumerate(ndx_bars):
        if bar.date <= anchor_day:
            anchor_idx = i
        else:
            break
    if anchor_idx is None or anchor_idx + 1 >= len(ndx_bars):
        return None

    base_bar = ndx_bars[anchor_idx]
    base_vxn = vxn_map.get(base_bar.date)
    if base_vxn is None:
        return None

    points: list[dict] = [
        {
            "step": 0,
            "date": base_bar.date.isoformat(),
            "ndx": round(base_bar.close, 2),
            "ndx_ret_pct": 0.0,
            "vxn": round(base_vxn, 2),
            "vxn_chg": 0.0,
            "tnx_pct": round(tnx_map.get(base_bar.date, 0.0), 3),
        }
    ]
    for s in range(1, steps + 1):
        target_idx = anchor_idx + s
        if target_idx >= len(ndx_bars):
            break
        bar = ndx_bars[target_idx]
        v_val = vxn_map.get(bar.date)
        if v_val is None:
            break
        points.append(
            {
                "step": s,
                "date": bar.date.isoformat(),
                "ndx": round(bar.close, 2),
                "ndx_ret_pct": round((bar.close / base_bar.close - 1.0) * 100.0, 2),
                "vxn": round(v_val, 2),
                "vxn_chg": round(v_val - base_vxn, 2),
                "tnx_pct": round(tnx_map.get(bar.date, 0.0), 3),
            }
        )
    if len(points) <= 1:
        return None
    last_pt = points[-1]
    return {
        "anchor_date": base_bar.date.isoformat(),
        "resolved_steps": len(points) - 1,
        "requested_steps": steps,
        "terminal_ndx": last_pt["ndx"],
        "terminal_ndx_ret_pct": last_pt["ndx_ret_pct"],
        "terminal_vxn": last_pt["vxn"],
        "terminal_vxn_chg": last_pt["vxn_chg"],
        "points": points,
    }


def historical_replay_catalog(root: Path, panel: HistoricalPanel, limit: int = 36) -> list[dict]:
    """Curated list of recent and largest-impulse historical FOMC/CPI/NFP shocks for UI replay."""
    try:
        common_dates, idx_map, episodes = _event_dates_aligned(root, panel)
    except Exception:
        return []
    ndx_map = {b.date: b.close for b in panel.series_map["^NDX"].bars}
    vxn_map = {b.date: b.close for b in panel.series_map["^VXN"].bars}
    tnx_map = {b.date: b.close for b in panel.series_map["^TNX"].bars}
    dxy_map = {b.date: b.close for b in panel.series_map["DX-Y.NYB"].bars}

    enriched: list[dict] = []
    for ev_day, kind in episodes:
        idx = idx_map[ev_day]
        prev_day = common_dates[idx - 1]
        dy_bps = (tnx_map[ev_day] - tnx_map[prev_day]) * 100.0
        dd_pct = math.log(dxy_map[ev_day] / dxy_map[prev_day]) * 100.0
        r0_pct = (ndx_map[ev_day] / ndx_map[prev_day] - 1.0) * 100.0
        dv0 = vxn_map[ev_day] - vxn_map[prev_day]
        enriched.append(
            {
                "event_date": ev_day.isoformat(),
                "pre_event_date": prev_day.isoformat(),
                "kind": kind,
                "rate_bps": round(dy_bps, 1),
                "fx_pct": round(dd_pct, 2),
                "actual_r0_pct": round(r0_pct, 2),
                "actual_dv0": round(dv0, 2),
                "abs_move": abs(r0_pct) + abs(dy_bps) / 10.0,
            }
        )
    recent = enriched[-16:]
    recent_dates = {e["event_date"] for e in recent}
    major = sorted(
        (e for e in enriched if e["event_date"] not in recent_dates),
        key=lambda x: x["abs_move"],
        reverse=True,
    )[: max(0, limit - len(recent))]
    merged = sorted(recent + major, key=lambda x: x["event_date"], reverse=True)
    for item in merged:
        item.pop("abs_move", None)
    return merged


def _clip_int(val: object, low: int, high: int, default: int = 5) -> int:
    try:
        parsed = int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(low, min(parsed, high))


def _build_shock_from_request(root: Path, body: dict, as_of: AsOf) -> PolicyShock:
    scenario_key = str(body.get("scenario") or "tariff_25").strip()
    base = PRESET_SHOCKS.get(scenario_key, PolicyShock(kind=str(body.get("kind") or "custom"), title=scenario_key))

    def _num(key: str, default: float) -> float:
        val = body.get(key)
        return float(default) if val is None or val == "" else float(val)

    headlines = gated_headlines_for_day(root, as_of.session, as_of, limit=5)
    return PolicyShock(
        kind=str(body.get("kind") or base.kind),
        title=str(body.get("title") if body.get("title") is not None else base.title),
        rate_bps=_num("rate_bps", base.rate_bps),
        fx_pct=_num("fx_pct", base.fx_pct),
        inflation_z=_num("inflation_z", base.inflation_z),
        growth_z=_num("growth_z", base.growth_z),
        tariff_pct=_num("tariff_pct", base.tariff_pct),
        uncertainty_z=_num("uncertainty_z", base.uncertainty_z),
        persistence=_clip(_num("persistence", base.persistence), 0.0, 0.90),
        headlines=headlines,
    )


def run_sim_payload(root: Path, body: dict, llms: Sequence[LLM] | dict[str, LLM] | None = None) -> dict:
    """Execute one simulation from a web request dict and attach optional realised overlay."""
    panel = cached_panel(root)
    ndx_bars = panel.series_map["^NDX"].bars
    raw_day = str(body.get("day") or "").strip()
    anchor_day = date.fromisoformat(raw_day) if raw_day else ndx_bars[-1].date
    if anchor_day > ndx_bars[-1].date:
        anchor_day = ndx_bars[-1].date

    as_of = AsOf.at_session_close(anchor_day)
    state, res_ret, res_vxn = panel.build_state(as_of)
    steps = _clip_int(body.get("steps"), 1, 15, default=5)
    mode_raw = str(body.get("mode") or "rule").strip().lower()
    mode: ActorMode = mode_raw if mode_raw in ("rule", "homogeneous", "llm") else "rule"  # type: ignore[assignment]
    models_raw = str(body.get("models") or "").strip()
    model_list = [m.strip() for m in models_raw.split(",") if m.strip()] or None

    shock = _build_shock_from_request(root, body, as_of)
    actors = build_actors(mode, root=root, llms=llms, models=model_list)
    result = simulate(
        state,
        shock,
        as_of=as_of,
        actors=actors,
        mode=mode,
        steps=steps,
        res_ret_step=res_ret,
        res_vxn_step=res_vxn,
    )
    doc = result.to_json()
    doc["ok"] = True
    doc["realized_overlay"] = _realized_overlay(panel, date.fromisoformat(state.as_of), steps)
    return doc


def run_sim_stream(root: Path, body: dict, llms: Sequence[LLM] | dict[str, LLM] | None = None) -> Iterator[dict]:
    """NDJSON event stream for live multi-step LLM simulation in the web UI."""
    panel = cached_panel(root)
    ndx_bars = panel.series_map["^NDX"].bars
    raw_day = str(body.get("day") or "").strip()
    anchor_day = date.fromisoformat(raw_day) if raw_day else ndx_bars[-1].date
    if anchor_day > ndx_bars[-1].date:
        anchor_day = ndx_bars[-1].date

    as_of = AsOf.at_session_close(anchor_day)
    state, res_ret, res_vxn = panel.build_state(as_of)
    steps = _clip_int(body.get("steps"), 1, 15, default=5)
    mode_raw = str(body.get("mode") or "llm").strip().lower()
    mode: ActorMode = mode_raw if mode_raw in ("rule", "homogeneous", "llm") else "llm"  # type: ignore[assignment]
    models_raw = str(body.get("models") or "deepseek-chat,qwen3.8-flash").strip()
    model_list = [m.strip() for m in models_raw.split(",") if m.strip()] or None

    shock = _build_shock_from_request(root, body, as_of)
    roster = build_actors(mode, root=root, llms=llms, models=model_list)
    weights = {a.spec.actor_id: a.spec.weight for a in roster}

    yield {
        "type": "start",
        "as_of": state.as_of,
        "steps": steps,
        "mode": mode,
        "initial_state": state.to_summary_dict(),
    }

    history: list[SimulationStep] = []
    cur = state
    total_usage = Usage()

    for step_idx in range(1, steps + 1):
        step_shock = shock.at_step(step_idx)
        yield {
            "type": "thinking",
            "step": step_idx,
            "steps": steps,
            "models": model_list or [mode],
        }
        if mode == "llm" and len(roster) > 1:
            with ThreadPoolExecutor(max_workers=len(roster)) as pool:
                step_actions = tuple(
                    pool.map(lambda actor: actor.act(cur, step_shock, history), roster)
                )
        else:
            step_actions = tuple(actor.act(cur, step_shock, history) for actor in roster)

        for action in step_actions:
            total_usage = total_usage + action.usage

        next_state, diag = clear_market_step(cur, step_shock, step_actions, weights=weights)
        rec = SimulationStep(
            step=step_idx,
            shock=step_shock,
            actions=step_actions,
            clearing=diag,
            state_after=next_state,
        )
        history.append(rec)
        cur = next_state
        yield {
            "type": "step",
            "step": step_idx,
            "step_data": rec.to_json(),
        }

    dist = _compute_distribution(state, history, res_ret, res_vxn)
    final_res = SimulationResult(
        as_of=as_of,
        mode=mode,
        initial_state=state,
        shock=shock,
        steps=tuple(history),
        distribution=dist,
        usage=total_usage,
    )
    doc = final_res.to_json()
    doc["ok"] = True
    doc["realized_overlay"] = _realized_overlay(panel, date.fromisoformat(state.as_of), steps)
    yield {"type": "done", "result": doc}


def sim_init_payload(root: Path) -> dict:
    """Initial bundle for `web/sim.html`: presets, actors, replay events, LLM status, and default run."""
    panel = cached_panel(root)
    ndx_bars = panel.series_map["^NDX"].bars
    latest_date = ndx_bars[-1].date.isoformat()
    min_date = ndx_bars[min(295, len(ndx_bars) - 1)].date.isoformat()
    env = merged_env(root)

    presets_out = []
    for meta in PRESET_UI_META:
        pid = meta["id"]
        spec = PRESET_SHOCKS[pid]
        presets_out.append(
            {
                "id": pid,
                "short_label": meta["short_label"],
                "badge": meta["badge"],
                **spec.to_json(),
            }
        )

    actors_out = [
        {
            "actor_id": a.actor_id,
            "label": a.label,
            "weight": a.weight,
            "mandate": a.mandate,
            "constraints": a.constraints,
        }
        for a in ACTOR_SPECS
    ]
    llm_status = {
        "deepseek": providers.describe("deepseek-chat", env),
        "qwen": providers.describe("qwen3.8-flash", env),
    }
    initial_run = run_sim_payload(root, {"scenario": "tariff_25", "steps": 5, "mode": "rule"})
    return {
        "ok": True,
        "latest_date": latest_date,
        "min_date": min_date,
        "presets": presets_out,
        "actors": actors_out,
        "historical_events": historical_replay_catalog(root, panel, limit=36),
        "llm_status": llm_status,
        "initial": initial_run,
    }


def load_or_run_backtest(
    root: Path,
    *,
    force: bool = False,
    kinds: str = "fomc,cpi,nfp",
    horizon: int = 5,
) -> dict:
    cache_path = root / "artifacts" / "sim" / "backtest_report.json"
    kind_tuple = tuple(k.strip().lower() for k in kinds.split(",") if k.strip())
    if not force and cache_path.is_file() and kind_tuple == ("fomc", "cpi", "nfp") and horizon == 5:
        try:
            doc = json.loads(cache_path.read_text())
            doc["ok"] = True
            doc["cached"] = True
            return doc
        except Exception:
            pass
    report = backtest_macro_shocks(root, kinds=kind_tuple, horizon=horizon)
    if kind_tuple == ("fomc", "cpi", "nfp") and horizon == 5:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    report["ok"] = True
    report["cached"] = False
    return report
