"""The A/B harness: sampling, the "two vendors really are two" guard, and the maths.

No network here. `compare()` reads whatever is on disk, so the test writes fake label
files and checks the numbers against values computed by hand.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from forecast_log import providers
from forecast_log.agent.llm import LLMError
from forecast_log.news_ab import compare, sample_days, slug_of


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def _news_day(root: Path, day: str, titles: list[str]) -> None:
    path = root / "data" / "news" / f"{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"as_of": day, "documents": [{"title": title} for title in titles]}))


def _labels(root: Path, slug: str, rows: dict[str, dict]) -> None:
    target = root / "artifacts" / "news_ab" / slug
    target.mkdir(parents=True, exist_ok=True)
    for day, label in rows.items():
        (target / f"{day}.json").write_text(
            json.dumps(
                {"as_of": day, "model": slug, "label": label, "usage": {"cost_usd": 0.001, "latency_s": 10.0}},
                ensure_ascii=False,
            )
        )


def test_sampling_is_reproducible_and_stratified(root: Path) -> None:
    for year in (2020, 2021, 2022):
        for month in range(1, 13):
            _news_day(root, f"{year}-{month:02d}-03", [f"{year}-{month} headline"])
    first = sample_days(root, per_year=2, seed=7)
    again = sample_days(root, per_year=2, seed=7)
    assert first == again
    assert len(first) == 6  # two per year, not two overall
    assert all(day.year in (2020, 2021, 2022) for day in first)
    other = sample_days(root, per_year=2, seed=8)
    assert other != first


def test_a_missing_key_for_one_vendor_does_not_fall_back_to_the_other(root: Path) -> None:
    env = {"DEEPSEEK_API_KEY": "sk-deepseek", "QWEN_BASE_URL": "https://example.invalid/v1"}
    assert providers.build("deepseek-flash", root, env).model == "deepseek-flash"
    with pytest.raises(LLMError):
        providers.build("qwen3.8-flash", root, env)


def test_slug_keeps_models_apart() -> None:
    assert slug_of("deepseek-flash") != slug_of("qwen3.8-flash")
    assert slug_of("qwen3.8-flash") == "qwen3.8-flash"


def test_compare_reports_agreement_veto_overlap_and_a_dead_heat(root: Path) -> None:
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    for day in days:
        _news_day(root, day, ["headline"])

    # A: two vetoes, one risk_off.
    _labels(
        root,
        "a",
        {
            "2020-01-02": {"regime": "risk_off", "severity": 0.9, "novelty": 0.5, "events": [{"type": "geopolitics"}]},
            "2020-01-03": {"regime": "neutral", "severity": 0.2, "novelty": 0.5, "events": [{"type": "markets"}]},
            "2020-01-06": {"regime": "risk_off", "severity": 0.7, "novelty": 0.5, "events": []},
            "2020-01-07": {"regime": "neutral", "severity": 0.1, "novelty": 0.5, "events": []},
        },
    )
    # B: vetoes on 01-02 and 01-07, so one day only A would veto and one only B.
    _labels(
        root,
        "b",
        {
            "2020-01-02": {"regime": "risk_off", "severity": 0.8, "novelty": 0.5, "events": [{"type": "geopolitics"}]},
            "2020-01-03": {"regime": "neutral", "severity": 0.3, "novelty": 0.5, "events": [{"type": "markets"}]},
            "2020-01-06": {"regime": "neutral", "severity": 0.1, "novelty": 0.5, "events": []},
            "2020-01-07": {"regime": "risk_off", "severity": 0.6, "novelty": 0.5, "events": []},
        },
    )

    report = compare(root, ["a", "b"])
    pair = report["pairs"]["a vs b"]

    assert report["shared_days"] == 4
    assert pair["regime_agreement"] == 0.5  # 01-06 and 01-07 differ
    assert pair["severity_mae"] == pytest.approx(0.325, abs=1e-6)  # (0.1+0.1+0.6+0.5)/4
    assert pair["veto_same_days"] == 1  # only 01-02
    assert pair["veto_only_left"] == 1  # 01-06
    assert pair["veto_only_right"] == 1  # 01-07
    assert pair["veto_union"] == 3
    # A ranks 4,2,3,1 against B's 4,2,1,3: sum of squared rank gaps 8 -> rho = 1 - 48/60.
    assert pair["severity_spearman"] == pytest.approx(0.2, abs=1e-6)
    assert report["per_labeler"]["a"]["severity_ge_0.6"] == 2
