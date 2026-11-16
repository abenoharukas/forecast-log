# forecast-log

A forecasting agent whose answer is locked before the answer is knowable, and scored after.

Three questions about `^VXN` and `^NDX` were frozen on 2026-11-11, with the rules, the
data source and the settlement session written down first. They resolve on 11/20 and 11/30.
You can come back and check the Brier score, or check whether the numbers moved.

That is the easy half. The half worth reading is the machinery that makes "we locked it and
did not peek" checkable instead of a promise.

## The claim, in one file

```python
# src/forecast_log/agent/gate.py
def check(self, day: date, source: str) -> date:
    if day > self.session:
        raise LookAheadError(...)
    return day
```

Every tool result, every document timestamp, and every date in the submitted evidence goes
through this. It **raises** rather than clips, because a clipped row looks exactly like a
correct answer and survives every test you would think to write. An exception shows up in
the first trace you read, which is the whole difference between an auditable forecast and a
backtest that quietly cheats.

A model that cites a date after the as-of session gets its submission rejected, is told
why, and has to answer again. The rejected attempt stays in the trace.

## Architecture

```
questions.json ──► Agent (8-step tool loop) ──► locked probability ──► Brier
                        │                            │
                        ├── 7 gated tools            └── forecasts/<id>.json (append-only)
                        ├── validator (rejects bad submissions, keeps them in the trace)
                        └── trace (every turn, every tool, every token, every dollar)
```

- **Tools**: price history, state statistics, point-in-time base rate, the repo's own
  analogue model (demoted from "the answer" to "one opinion"), news search, document read,
  prediction-market quote.
- **Loop**: no framework. 8-step budget, 2.4 KB per tool result, structured submission as
  a tool call, rejections fed back into the transcript.
- **Models**: ~120 lines of `urllib` against `/chat/completions`. DeepSeek, OpenAI,
  OpenRouter, vLLM and Ollama are the same code with a different base URL. Cost comes from
  the tokens the provider reports, not from an estimate.
- **Ensemble**: N independent runs pooled in log-odds, with the logit spread returned so
  disagreement is visible instead of averaged away.

## Measured

96 questions, `^VXN`, 2003-01-28 → 2025-10-07, 48 of each shape, one DeepSeek run each:

| predictor | Brier | LogLoss | ECE | direction | $/forecast | p50 |
|---|---|---|---|---|---|---|
| **agent + tools** | **0.1631** | 0.4815 | **0.036** | **72.9%** | 0.0011 | 7.6 s |
| analogue model (the repo's own) | 0.1636 | 0.4719 | 0.053 | 71.9% | 0 | 0.04 s |
| agent, tools removed | 0.2293 | 0.6486 | 0.115 | 55.2% | 0.0002 | 2.3 s |
| constant 0.5 | 0.2500 | 0.6931 | 0.073 | 42.7% | 0 | 0 |
| point-in-time base rate | 0.2647 | 0.7953 | 0.162 | 55.2% | 0 | 0 |

Three things this table says, none of which I picked in advance:

1. **The tools are the forecast.** Same model, same questions, tools removed: Brier 0.1631 →
   0.2293, direction 72.9% → 55.2%. The LLM has no prior about `^VXN`; what it does is decide
   which tool to ask, how many times, and which answer to believe.
2. **It ties the analogue model and is better calibrated** (ECE 0.036 vs 0.053; mean 0.442
   against an observed 0.427). It inherits the model's judgement rather than adding any of
   its own. Which is why the table has a `$` column: on a purely numeric question the agent
   is the explanation, not the estimator.
3. **The base rate is worse than a coin flip** (0.2647 vs 0.25). A historical hit rate on an
   *absolute* price level is carried away by drift: `VXN > 46` meant something different in
   2003. The agent caught this on the real NDX question and said so in its rationale — the
   historical rate there was 1.15% and the threshold was today's close.

The gate never fired in 96 runs, which is the right amount of drama: it is cheap enough to
keep for the case where it does, and the tests prove it would.

Locked versus shadow on the three live questions:

| question | locked (analogue) | shadow (3 runs pooled) | spread |
|---|---|---|---|
| `vxn-2026-11-20-gt-30` | 0.3150 | 0.2995 | 0.10 |
| `vxn-2026-11-30-gt-30` | 0.3250 | 0.2557 | 0.14 |
| `ndx-2026-11-30-gt-baseline` | 0.5800 | 0.5134 | 0.13 |

## What I did not build

Said plainly, because it is the difference between a demo and a delivery:

- **No licensed intraday financial newswire.** `search_news` is backed by 127k+ time-gated headlines in `data/forecast.db` (Wikipedia `Portal:Current events` 2002–2026 plus runtime GDELT/RSS feeds via `forecast-log daily`), rather than an empty corpus. What is still missing is a licensed institutional feed with minute-level timestamps rather than daily world-news summaries.
- **No policy simulation.** `ensemble.py` is the multi-agent machinery; there is no economic model for the actors to act inside.
- **No queue, API service, or DB-enforced forecast locks.** Market bars and news live in `data/forecast.db` (SQLite WAL), while `forecasts/` remains append-only JSON files by convention (`lock` refuses to overwrite) with no background worker queue or production service.

## The part I would actually point at

`experiments/` preserves nine approaches that did not survive
their own backtest — logistic regression at Brier 0.2472 against a constant 0.2410,
probability shrinkage towards the base rate, twelve cross-asset factors each individually
inside the base rate's confidence interval, and a sell signal that cannot be built from
direction at all. Numbers I could have quietly dropped.

The same discipline is why the table above keeps a predictor that is free, 200x faster, and
better on one of the two question shapes. That is not a failure of the agent; it is the
reason to trust anything else the bench says.

## Run it

```bash
git lfs pull                          # pulls data/forecast.db (15 series, 127k+ headlines)
pip install -e ".[dev]"
cp .env.example .env && $EDITOR .env    # DEEPSEEK_API_KEY=... ; git-ignored
forecast-log fetch
forecast-log backtest          # 3,700+ forecasts, scored against what happened
forecast-log agent-eval --limit 96 --stride 60 --also-blind
pytest                          # 72 tests, no network, no API key
python3 serve.py 8731           # web/agent.html reads artifacts/agent_eval_summary.json
```
