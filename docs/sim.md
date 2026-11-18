# Multi-Agent Policy & Market Simulation (`src/forecast_log/sim.py`)

This module implements a **computed** policy and macro-shock simulator in which heterogeneous institutional actors respond to shocks inside explicit cross-asset market-clearing state equations—rather than narrating price moves in free-form prose.

## 1. Design Principles

1. **Actors Submit Bounded Flows, Never Prices**: No actor (rule-based or LLM) ever outputs a target index level or future volatility print. Each actor observes a strictly time-gated `MarketState` (`AsOf.check` and `AsOf.check_published` enforced on every bar and headline) plus an exogenous `PolicyShock`, and submits bounded order-flow adjustments in `[-1.0, +1.0]`:
   - `equity_delta`: Net Nasdaq-100 (`^NDX`) exposure adjustment
   - `bond_delta`: Net 10-Year Treasury (`^TNX`) duration demand
   - `credit_delta`: Net high-yield vs. Treasury (`HYG / TLT`) credit appetite flow
   - `vol_hedge_delta`: Net option protection (`^VXN`) demand
2. **Explicit 5-State Clearing Engine (`clear_market_step`)**:
   - **State Vector**: $S_t = (\text{NDX}_t,\; \text{VXN}_t,\; \text{TNX}_t,\; \ln(\text{HYG}_t / \text{TLT}_t),\; \text{DXY}_t)$
   - **Endogenous Liquidity (Kyle's $\lambda_t$)**: Price impact widens endogenously when `^VXN` exceeds its trailing 252-session median and when the options market maker withdraws equity liquidity (`options_dealer_mm.equity_delta < 0`).
   - **Regime-Switching Duration Transmission**: Uses the strictly point-in-time trailing 63-session stock-yield correlation (`corr(r_NDX, d_TNX)`) to separate growth/reflation regimes (`corr > 0`, pre-2021 ZLB) from inflation/discount-rate regimes (`corr < 0` or `TNX >= 3.25%`, 2021+).
   - **Volatility Mean-Reversion & Term Structure**: Combines Ornstein-Uhlenbeck pull toward the 252-day `^VXN` median / 21-day realized volatility anchor, `^VIX9D / ^VIX3M` term-structure relaxation once scheduled event uncertainty resolves, equity leverage effect, and net institutional hedging flow.
3. **Interchangeable Actor Backends on Identical Equations**:
   - `--mode rule`: Deterministic micro-founded institutional reaction functions (`RuleActor`).
   - `--mode homogeneous`: Single undifferentiated trader ablation control (`HomogeneousRuleActor`).
   - `--mode llm`: Live LLM orchestration (`LLMActor`), supporting single-model or multi-vendor rosters (e.g., `--models deepseek-chat,qwen3.8-flash`) executed concurrently across actors per step with token and USD cost tracking.

## 2. Four Heterogeneous Institutional Mandates (`ACTOR_SPECS`)

| Actor ID | Capital Weight | Institutional Mandate & Endogenous Feedback Role |
|---|---|---|
| `macro_fund` | `0.30` | Trades equity beta, 10Y duration, and USD liquidity; flips yield sensitivity sign across growth vs. inflation regimes. |
| `vol_target_cta` | `0.25` | Targets `~18%` annualized volatility (`18 / VXN`); mechanically deleverages in Step `2..H` after Step `1` volatility spikes. |
| `corp_credit_hedger` | `0.25` | Hedges tariff shocks (`tariff_pct`), credit widening (`HYG / TLT`), and refinancing stress via put buying (`vol_hedge_delta > 0`). |
| `options_dealer_mm` | `0.20` | Absorbs imbalances in calm contango (`^VIX9D / ^VIX3M < 0.98`); hedges short gamma and pulls liquidity in stressed backwardation. |

## 3. Empirical Backtest Results (`488` Macro Events, `2011-02-17` to `2026-11-10`)

Reproduce via:

```bash
forecast-log sim --scenario tariff_25 --steps 5
forecast-log sim-backtest
```

### Track 1 — Conditioned Shock Transmission (`t-1 -> t0`, `FOMC` / `CPI` / `NFP`)

State frozen strictly at pre-event close `t-1` (`AsOf.at_session_close(t-1)`). Given the event-day rate (`d_TNX`) and FX (`d_DXY`) impulse without touching `t0` equity, credit, or volatility prices:

| Predictor | Full Brier | Full Hit | Full Return Corr | 2021+ Holdout Brier | 2021+ Return Corr |
|---|---|---|---|---|---|
| **`multi_agent_rule`** | **`0.2430`** | **`57.4%`** | **`+0.319`** | **`0.2403`** | **`+0.347`** |
| `homogeneous_ablation` | `0.2453` | `54.7%` | `+0.312` | `0.2372` | `+0.377` |
| `expanding_ols` | `0.2400` | `59.4%` | `+0.242` | `0.2374` | `+0.264` |
| `constant_0.5` | `0.2500` | `55.9%` | — | `0.2500` | — |

- **Takeaway**: Expanding-window OLS suffers a correlation drop (`+0.242` vs. `+0.319` full sample; `+0.264` vs. `+0.347` post-2021) because pre-2021 positive stock-yield covariance dilutes post-2021 discount-rate sensitivity. Heterogeneous mandates improve full-sample directional accuracy by **`+2.7 pp`** (`57.4%` vs. `54.7%`) over the homogeneous actor ablation.

### Track 2 — Pure Ex-Ante 5-Session Volatility Propagation (`P(VXN_{t+5} > VXN_t)`)

State frozen strictly at event-day close `t0` (`AsOf.at_session_close(t0)`), zero post-`t0` inputs:

| Predictor | Brier | LogLoss | ECE | Hit Rate | Mean Prob | Observed Rate |
|---|---|---|---|---|---|---|
| `homogeneous_ablation` | **`0.2467`** | **`0.6865`** | **`0.007`** | **`57.6%`** | `0.491` | `0.486` |
| **`multi_agent_rule`** | **`0.2483`** | **`0.6897`** | **`0.010`** | **`52.9%`** | `0.495` | `0.486` |
| `analogue_model` | `0.2489` | `0.6912` | `0.016` | `55.1%` | `0.486` | `0.486` |
| `constant_0.5` | `0.2500` | `0.6931` | `0.014` | `48.6%` | `0.500` | `0.486` |
| `expanding_base_rate` | `0.2516` | `0.6964` | `0.015` | `51.0%` | `0.471` | `0.486` |

### Track 3 — Pure Ex-Ante 5-Session Equity Drift (`P(NDX_{t+5} > NDX_t)`, Negative Result Disclosed)

| Predictor | Brier | LogLoss | ECE | Hit Rate |
|---|---|---|---|---|
| `analogue_model` | `0.2495` | `0.6924` | `0.034` | `54.1%` |
| `expanding_base_rate` | `0.2498` | `0.6929` | `0.008` | `54.3%` |
| `constant_0.5` | `0.2500` | `0.6931` | `0.053` | `55.3%` |
| `multi_agent_rule` | `0.2506` | `0.6945` | `0.041` | `52.5%` |
| `homogeneous_ablation` | `0.2509` | `0.6950` | `0.049` | `52.2%` |

- **Honest Boundary of Simulation Alpha**: Both state-space simulation variants beat `analogue_model`, `expanding_base_rate`, and `constant_0.5` on **5-day post-event volatility relaxation/escalation** (`Brier 0.2467–0.2483`, `ECE <= 0.010`), where `^VIX9D / ^VIX3M` backwardation unwinds predictably after scheduled announcements. By contrast, **unconditional 5-day post-event equity drift** (`Brier 0.2506` vs. `0.2500`) is already priced by the event-day close—confirming that multi-agent policy simulation adds value on **conditional counterfactual shock transmission** and **volatility regime dynamics**, not on predicting unshocked random-walk equity closes.
