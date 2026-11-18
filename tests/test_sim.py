from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from forecast_log.agent.gate import NY, AsOf, LookAheadError
from forecast_log.agent.llm import Reply, ScriptedLLM, Usage
from forecast_log.bars import Bar, Series
from forecast_log.db import save_news_fetch, upsert_series
from forecast_log.sim import (
    ACTOR_SPECS,
    PRESET_SHOCKS,
    HistoricalPanel,
    MarketState,
    PolicyShock,
    backtest_macro_shocks,
    build_actors,
    gated_headlines_for_day,
    render_backtest,
    render_simulation,
    simulate,
)


def _seed_cross_asset_panel(root: Path, n_days: int = 340) -> list[date]:
    rng = np.random.default_rng(42)
    start = date(2020, 1, 2)
    days: list[date] = []
    cur = start
    while len(days) < n_days:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)

    # First half: low rate (1.5%), positive stock-yield correlation;
    # Second half: high rate (4.2%), negative stock-yield correlation (inflation regime).
    ndx = [10000.0]
    vxn = [22.0]
    tnx = [1.50]
    hyg = [82.0]
    tlt = [135.0]
    dxy = [97.0]
    vix9d = [20.0]
    vix3m = [23.0]

    for i in range(1, n_days):
        high_rate = i >= n_days // 2
        dy = float(rng.normal(0.015 if i == n_days // 2 else 0.0, 0.06))
        tnx_next = max(0.3, (4.10 if (i == n_days // 2) else tnx[-1]) + dy)
        beta_ry = -0.14 if high_rate else +0.12
        r_eq = beta_ry * dy + float(rng.normal(0.0005, 0.011))
        ndx_next = ndx[-1] * float(np.exp(r_eq))
        dv = -55.0 * r_eq + 0.12 * (22.0 - vxn[-1]) + float(rng.normal(0.0, 0.9))
        vxn_next = max(11.0, vxn[-1] + dv)
        dxy_next = dxy[-1] * float(np.exp(0.01 * dy + float(rng.normal(0.0, 0.003))))
        hyg_next = hyg[-1] * float(np.exp(0.35 * r_eq))
        tlt_next = tlt[-1] * float(np.exp(-0.06 * dy))

        ndx.append(ndx_next)
        vxn.append(vxn_next)
        tnx.append(tnx_next)
        hyg.append(hyg_next)
        tlt.append(tlt_next)
        dxy.append(dxy_next)
        vix9d.append(max(10.0, vxn_next * (1.08 if dv > 1.2 else 0.92)))
        vix3m.append(max(12.0, 0.7 * vix3m[-1] + 0.3 * vxn_next))

    fetched = "2026-11-17T22:00:00Z"
    series_data = {
        "^NDX": ndx,
        "^VXN": vxn,
        "^TNX": tnx,
        "HYG": hyg,
        "TLT": tlt,
        "DX-Y.NYB": dxy,
        "^VIX9D": vix9d,
        "^VIX3M": vix3m,
    }
    for sym, vals in series_data.items():
        bars = tuple(Bar(d, round(float(v), 4), is_closed=True) for d, v in zip(days, vals))
        upsert_series(root, Series(sym, "America/New_York", fetched, bars))

    # Write synthetic macro event calendar inside the valid window
    ev_dates = days[295 : n_days - 8 : 6] if n_days >= 360 else days[260 : n_days - 8 : 6]
    calendar = {
        "fomc": [d.isoformat() for d in ev_dates[::3]],
        "cpi": [d.isoformat() for d in ev_dates[1::3]],
        "nfp": [d.isoformat() for d in ev_dates[2::3]],
    }
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "events.json").write_text(json.dumps({"calendar": calendar}))
    return days


def test_state_builder_enforces_as_of_gate_and_rejects_future_news(tmp_path: Path) -> None:
    days = _seed_cross_asset_panel(tmp_path, n_days=310)
    cutoff = days[280]
    gate = AsOf.at_session_close(cutoff)
    panel = HistoricalPanel.load(tmp_path)

    state, res_ret, res_vxn = panel.build_state(gate)
    assert state.as_of == cutoff.isoformat()
    assert res_ret > 0 and res_vxn > 0

    # Inject a post-close headline on cutoff day (18:30 ET > 16:15 ET session close)
    save_news_fetch(
        tmp_path,
        {
            "as_of": cutoff.isoformat(),
            "source": "test",
            "documents": [
                {
                    "id": "after-hours-leak",
                    "published_at": f"{cutoff.isoformat()}T18:30:00-05:00",
                    "title": "Emergency tariff leaked after market close",
                }
            ],
        },
    )
    with pytest.raises(LookAheadError, match="after the decision instant"):
        gated_headlines_for_day(tmp_path, cutoff, gate)

    # Simulating from a state newer than `as_of` raises LookAheadError immediately
    older_gate = AsOf.at_session_close(days[279])
    with pytest.raises(LookAheadError):
        simulate(state, PRESET_SHOCKS["tariff_25"], as_of=older_gate)


def test_regime_flip_changes_macro_fund_reaction_and_computed_equilibrium() -> None:
    hawkish = PolicyShock(kind="fomc", rate_bps=18.0, fx_pct=0.4, inflation_z=0.3, growth_z=0.1)

    low_rate_growth_state = MarketState(
        as_of="2019-06-19",
        step=0,
        ndx=7700.0,
        vxn=18.5,
        tnx=2.05,
        credit_log_ratio=-0.45,
        dxy=97.2,
        vxn_median_252d=19.0,
        vxn_std_252d=3.5,
        realized_vol_21d=16.5,
        vol_term_ratio=0.92,
        stock_yield_corr_63d=+0.28,
        ndx_ret_5d=0.01,
        ndx_ret_20d=0.02,
        tnx_chg_20d_bps=-8.0,
        credit_chg_20d=0.005,
        dxy_ret_20d=-0.004,
        prev_implied_leverage=0.97,
    )
    high_rate_inflation_state = MarketState(
        as_of="2023-09-20",
        step=0,
        ndx=15000.0,
        vxn=18.5,
        tnx=4.40,
        credit_log_ratio=-0.15,
        dxy=105.2,
        vxn_median_252d=19.0,
        vxn_std_252d=3.5,
        realized_vol_21d=16.5,
        vol_term_ratio=0.92,
        stock_yield_corr_63d=-0.35,
        ndx_ret_5d=0.01,
        ndx_ret_20d=0.02,
        tnx_chg_20d_bps=15.0,
        credit_chg_20d=-0.005,
        dxy_ret_20d=0.01,
        prev_implied_leverage=0.97,
    )

    sim_growth = simulate(low_rate_growth_state, hawkish, steps=3, mode="rule")
    sim_infl = simulate(high_rate_inflation_state, hawkish, steps=3, mode="rule")

    macro_growth = next(a for a in sim_growth.steps[0].actions if a.actor_id == "macro_fund")
    macro_infl = next(a for a in sim_infl.steps[0].actions if a.actor_id == "macro_fund")
    assert macro_infl.equity_delta < macro_growth.equity_delta
    assert sim_infl.cumulative_ndx_log_return < sim_growth.cumulative_ndx_log_return


def test_tariff_shock_triggers_multi_step_vol_targeting_deleveraging(tmp_path: Path) -> None:
    days = _seed_cross_asset_panel(tmp_path, n_days=310)
    panel = HistoricalPanel.load(tmp_path)
    state, res_r, res_v = panel.build_state(AsOf.at_session_close(days[-1]))

    res = simulate(
        state,
        PRESET_SHOCKS["tariff_25"],
        steps=5,
        mode="rule",
        res_ret_step=res_r,
        res_vxn_step=res_v,
    )
    assert len(res.steps) == 5
    assert res.cumulative_ndx_log_return < 0.0
    assert res.cumulative_vxn_change > 0.0
    assert res.distribution.p_ndx_up < 0.50
    assert res.distribution.p_vxn_up > 0.50

    # Verify Vol-Target CTA reduces equity exposure after Step 1 VXN spike
    cta_step2 = next(a for a in res.steps[1].actions if a.actor_id == "vol_target_cta")
    assert cta_step2.equity_delta < 0.0

    rendered = render_simulation(res)
    assert "tariff_25" in rendered or "tariff" in rendered.lower()
    assert "macro_fund" in rendered and "options_dealer_mm" in rendered


def test_multi_vendor_llm_actors_execute_inside_same_clearing_equations(tmp_path: Path) -> None:
    days = _seed_cross_asset_panel(tmp_path, n_days=310)
    panel = HistoricalPanel.load(tmp_path)
    state, res_r, res_v = panel.build_state(AsOf.at_session_close(days[-1]))

    def make_reply(eq: float, vh: float, note: str, cost: float) -> Reply:
        payload = {
            "equity_delta": eq,
            "bond_delta": -0.2,
            "credit_delta": -0.3,
            "vol_hedge_delta": vh,
            "confidence": 0.8,
            "rationale": note,
        }
        return Reply(
            content=f"```json\n{json.dumps(payload)}\n```",
            usage=Usage(prompt_tokens=200, completion_tokens=60, cost_usd=cost, latency_s=0.4),
        )

    ds_llm = ScriptedLLM([make_reply(-0.55, 0.45, "Bearish discount-rate compression", 0.0002)], name="deepseek-chat")
    qw_llm = ScriptedLLM([make_reply(-0.35, 0.60, "Hedging supply-chain tariff shock", 0.0001)], name="qwen3.8-flash")

    actors = build_actors("llm", llms=[ds_llm, qw_llm])
    assert len(actors) == len(ACTOR_SPECS)

    res = simulate(
        state,
        PRESET_SHOCKS["tariff_25"],
        actors=actors,
        mode="llm",
        steps=2,
        res_ret_step=res_r,
        res_vxn_step=res_v,
        parallel_actors=False,
    )
    assert len(res.steps) == 2
    assert res.usage.prompt_tokens == 2 * 4 * 200
    assert res.usage.cost_usd == pytest.approx(4 * 0.0002 + 4 * 0.0001)
    assert res.final_state.ndx < res.initial_state.ndx
    assert res.final_state.vxn > res.initial_state.vxn


def test_macro_shock_backtest_runs_end_to_end_and_preserves_five_db_tables(tmp_path: Path) -> None:
    from forecast_log.db import connect, db_path

    _seed_cross_asset_panel(tmp_path, n_days=380)
    report = backtest_macro_shocks(tmp_path, horizon=5)
    assert report["n_episodes"] >= 10
    assert "multi_agent_rule" in report["track1_shock_transmission"]
    assert "homogeneous_ablation" in report["track1_shock_transmission"]
    assert "analogue_model" in report["track2_ex_ante_vxn_5d"]

    text = render_backtest(report)
    assert "Shock Transmission" in text
    assert "Volatility Propagation" in text

    conn = connect(db_path(tmp_path))
    try:
        tables = sorted(
            r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        )
    finally:
        conn.close()
    assert tables in (
        ["agent_eval", "bars", "chat_messages", "news", "signals"],
    )


def test_web_sim_api_init_historical_overlay_and_stream(tmp_path: Path) -> None:
    from forecast_log.server import PAGES
    from forecast_log.sim import run_sim_payload, run_sim_stream, sim_init_payload

    assert PAGES.get("/sim") == "web/sim.html"
    days = _seed_cross_asset_panel(tmp_path, n_days=380)

    bundle = sim_init_payload(tmp_path)
    assert bundle["ok"] is True
    assert len(bundle["presets"]) == 6
    assert len(bundle["actors"]) == 4
    assert bundle["initial"]["realized_overlay"] is None  # latest date has no future bars

    # Historical replay 15 sessions before the end has full 5-step realized overlay
    hist_day = days[-15].isoformat()
    replay_doc = run_sim_payload(
        tmp_path,
        {"scenario": "fomc_hawkish_25", "day": hist_day, "steps": 5, "mode": "rule"},
    )
    assert replay_doc["ok"] is True
    overlay = replay_doc["realized_overlay"]
    assert overlay is not None
    assert overlay["resolved_steps"] == 5
    assert len(overlay["points"]) == 6

    # Streaming LLM mode emits start -> thinking/step -> done
    fake_llm = ScriptedLLM(
        [
            Reply(
                content='{"equity_delta":-0.4,"bond_delta":-0.1,"credit_delta":-0.2,"vol_hedge_delta":0.5,"confidence":0.7,"rationale":"Tariff hedge"}',
                usage=Usage(prompt_tokens=100, completion_tokens=40, cost_usd=0.0001, latency_s=0.2),
            )
        ],
        name="deepseek-chat",
    )
    events = list(
        run_sim_stream(
            tmp_path,
            {"scenario": "tariff_25", "day": hist_day, "steps": 2, "mode": "llm"},
            llms=[fake_llm],
        )
    )
    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "done"
    assert len(events[-1]["result"]["steps"]) == 2

