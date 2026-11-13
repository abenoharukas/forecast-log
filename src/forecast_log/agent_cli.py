"""CLI side of the agent: a shadow forecast for the real questions, and the eval bench.

The shadow record never touches `forecasts/`. Those files are the locked contract and the
only thing that gets a Brier score; the agent writes to `artifacts/agent/` so the two
numbers can be compared without either one being able to overwrite the other.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from forecast_log.agent.ensemble import pool, spread
from forecast_log.agent.eval import (
    AgentPredictor,
    AnaloguePredictor,
    BaseRatePredictor,
    ConstantPredictor,
    build_questions,
    evaluate,
    render,
    settlement_questions,
    write_report,
)
from forecast_log.agent.gate import AsOf
from forecast_log.agent.llm import LLM, OpenAICompatLLM
from forecast_log.agent.loop import Agent
from forecast_log.agent.tools import Toolbox
from forecast_log.bars import load_series
from forecast_log.paths import series_path

NY = ZoneInfo("America/New_York")


def build_llm(model: str | None = None) -> LLM:
    llm = OpenAICompatLLM.from_env()
    if model:
        llm.model = model
        llm.name = f"{model}@{llm.base_url.split('//')[-1].split('/')[0]}"
    return llm


def agent_forecast_command(
    root: Path,
    *,
    runs: int = 1,
    model: str | None = None,
    temperature: float = 0.7,
    force: bool = False,
    now: datetime | None = None,
) -> dict:
    llm = build_llm(model)
    out_dir = root / "artifacts" / "agent"
    trace_dir = out_dir / "traces"
    questions = settlement_questions(root)
    if not questions:
        raise SystemExit("questions.json 里没有题目")

    records = {}
    for question in questions:
        series = load_series(series_path(root, question.symbol))
        as_of_session = series.bars[-1].date
        as_of = AsOf.at_session_close(as_of_session)
        path = out_dir / f"{question.id}.json"
        if path.exists() and not force:
            print(f"{question.id} 已经有影子记录，跳过（--force 才覆盖）")
            records[question.id] = json.loads(path.read_text())
            continue

        agent = Agent(llm, Toolbox(root, as_of), temperature=temperature)
        answers, traces = [], []
        for _ in range(runs):
            trace = agent.run(question)
            traces.append(trace)
            if trace.answer is not None:
                answers.append(trace.answer)
            else:
                print(f"  {question.id} 这一轮没有给出答案：{trace.error}")
        if not answers:
            raise SystemExit(f"{question.id} 没有任何一轮给出答案，不写记录")

        probability = pool([answer.probability for answer in answers])
        locked_path = root / "forecasts" / f"{question.id}.json"
        locked = json.loads(locked_path.read_text())["probability"] if locked_path.exists() else None
        record = {
            "question_id": question.id,
            "kind": "shadow forecast; the locked record is forecasts/<id>.json",
            "recorded_at": (now or datetime.now(NY)).astimezone(NY).isoformat(timespec="seconds"),
            "as_of": as_of.to_json(),
            "probability": round(probability, 6),
            "method": {
                "name": "agent_ensemble" if runs > 1 else "agent",
                "model": getattr(llm, "name", "?"),
                "runs": runs,
                "run_probabilities": [round(answer.probability, 6) for answer in answers],
                "pooling": "mean log-odds",
                "spread_logit": round(spread([answer.probability for answer in answers]), 4),
                "max_steps": agent.max_steps,
                "temperature": temperature,
            },
            "answer": answers[0].to_json(),
            "locked_probability": locked,
            "usage": {
                key: round(sum(getattr(trace.usage, key) for trace in traces), 6)
                for key in ("prompt_tokens", "completion_tokens", "cost_usd", "latency_s")
            },
            "tool_calls": sum(trace.tool_calls for trace in traces),
            "gate_hits": sum(trace.gate_hits for trace in traces),
            "trace_ref": str((trace_dir / f"{question.id}.json").relative_to(root)),
        }
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{question.id}.json").write_text(
            json.dumps([trace.to_json() for trace in traces], indent=2, ensure_ascii=False) + "\n"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        records[question.id] = record
        comparison = f"，锁定记录 {locked:.4f}（类比模型）" if locked is not None else ""
        print(
            f"{question.id} 影子概率 {record['probability']:.4f}{comparison}，"
            f"${record['usage']['cost_usd']:.4f}，{record['tool_calls']} 次工具调用"
        )
        if record["answer"]["evidence"]:
            print(f"  证据锚点：{record['answer']['anchored_on']}；{record['answer']['rationale'][:160]}")
    return records


def agent_eval_command(
    root: Path,
    *,
    symbol: str = "^VXN",
    stride: int = 20,
    limit: int = 12,
    runs: int = 1,
    model: str | None = None,
    blind: bool = False,
    with_agent: bool = True,
    also_blind: bool = False,
    modes: tuple[str, ...] = ("level", "quantile"),
    out: str = "artifacts/agent_eval.json",
    max_steps: int = 8,
    workers: int = 1,
) -> dict:
    series = load_series(series_path(root, symbol))
    questions = build_questions(series, stride=stride, limit=limit, modes=modes)
    print(f"{len(questions)} 道题，{symbol}，{questions[0].as_of} → {questions[-1].question.resolve_session}")

    predictors = [ConstantPredictor(), BaseRatePredictor(root), AnaloguePredictor(root)]
    agents: list[AgentPredictor] = []
    if with_agent:
        llm = build_llm(model)
        agents.append(AgentPredictor(root, llm, runs=runs, blind=blind, max_steps=max_steps))
        if also_blind and not blind:
            # Same model, same questions, no tools: the ablation that says whether the
            # tools and the gate are doing the work, or the LLM's own price intuition.
            agents.append(AgentPredictor(root, llm, runs=runs, blind=True, max_steps=max_steps))
        predictors.extend(agents)

    report = evaluate(predictors, questions, workers=workers)
    report["generated_at"] = datetime.now(NY).isoformat(timespec="seconds")
    report["config"] = {
        "symbol": symbol,
        "stride": stride,
        "limit": limit,
        "runs": runs,
        "blind": blind,
        "modes": list(modes),
        "model": getattr(llm, "name", None) if with_agent else None,
        "workers": workers,
    }
    report["traces"] = {
        predictor.name: [trace.to_json() for trace in predictor.traces]
        for predictor in agents
        if predictor.traces
    }
    path = write_report(root / out, report)
    # The page should not download 1.2 MB of traces to draw six rows.
    write_report(
        Path(str(path).replace(".json", "_summary.json")),
        {key: value for key, value in report.items() if key != "traces"},
    )
    print(render(report))
    print(f"\n完整结果写在 {path}")
    return report


def markdown_table(report: dict) -> str:
    """The tables in README.md and SUBMISSION.md are generated from the JSON, not typed."""
    rows = report.get("predictors") or {}
    if not rows:
        return "没有可比的结果"
    lines = [
        f"| 预测者 | Brier | LogLoss | ECE | 方向 | $/条 | p50 秒 | 工具 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in sorted(rows.items(), key=lambda pair: pair[1]["brier"]):
        lines.append(
            f"| {name} | {row['brier']:.4f} | {row['log_loss']:.4f} | {row['ece']:.3f} | "
            f"{row['hit_direction']:.1%} | {row['cost_usd_per_forecast']:.5f} | "
            f"{row['latency_p50_s']:.2f} | {row['tool_calls_mean']:.2f} |"
        )
    reference = report.get("references") or {}
    if reference:
        lines.append(
            f"| 参照：始终 0.5 | {reference['constant_0.5']['brier']:.4f} | "
            f"{reference['constant_0.5']['log_loss']:.4f} | — | — | 0 | 0 | 0 |"
        )
        lines.append("")
        lines.append(f"{report['n']} 道题，实际发生率 {reference['observed_rate']:.1%}。")

    families = sorted({family for row in rows.values() for family in row.get("by_family", {})})
    if families:
        names = list(rows)
        lines.append("")
        lines.append("| 题型 | n | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 2) + "|")
        for family in families:
            counts = [rows[name]["by_family"][family]["n"] for name in names if family in rows[name].get("by_family", {})]
            cells = [
                f"{rows[name]['by_family'][family]['brier']:.4f}"
                if family in rows[name].get("by_family", {})
                else "—"
                for name in names
            ]
            lines.append(f"| {family} | {max(counts, default=0)} | " + " | ".join(cells) + " |")

    extras = [
        f"{name} 失败 {row['failures']} 次"
        for name, row in rows.items()
        if row.get("failures")
    ] + [
        f"{name} 被闸门拦住 {row['gate_hits']} 次"
        for name, row in rows.items()
        if row.get("gate_hits")
    ]
    if extras:
        lines.append("")
        lines.append("；".join(extras) + "。")
    return "\n".join(lines)


def agent_report_command(root: Path, path: str = "artifacts/agent_eval.json") -> str:
    report = json.loads((root / path).read_text())
    table = markdown_table(report)
    print(table)
    return table
