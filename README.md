# Value Investing Agent

A small Python starter project using LangChain, LangGraph, and LangMem to screen a public company for conservative value-investing eligibility.

It takes a ticker, fetches SEC XBRL company facts, builds a compact 10-year annual fundamentals table, calculates deterministic value-investing metrics, uses an LLM only for the final explanation, persists thread state, stores long-term memory, and emits LangSmith traces.

> This is a rules-based screening tool for research and education. It is not personalized investment advice. Always review original filings before making investment decisions.

## What it does

- Resolves ticker to SEC CIK.
- Fetches SEC `companyfacts` XBRL data.
- Extracts 10 years of annual balance-sheet, income, cash-flow, EPS, and share-count facts where available.
- Computes:
  - revenue CAGR,
  - book-equity CAGR,
  - positive net-income years,
  - positive operating-cash-flow years,
  - current ratio,
  - debt/equity,
  - P/E,
  - P/B,
  - Graham number,
  - margin of safety.
- Decides `Eligible` vs `Not eligible / watchlist` using conservative thresholds.
- Computes a minimum starter allocation using configured portfolio rules.
- Persists:
  - LangGraph checkpoints by `thread_id`,
  - LangMem long-term memory in Postgres/pgvector.
- Stores system prompt versions in Postgres (`public.system_prompts`) with soft deletes and audit history (`public.system_prompts_audit`).
- After each run, asks for feedback and can auto-optimize + persist a new prompt version when quality issues are detected.
- Supports LangSmith tracing and metadata.

## Architecture

```text
CLI ticker
  -> LangGraph thread_id
  -> load_memory           LangMem search by thread
  -> fetch_sec             SEC ticker mapping + companyfacts
  -> fetch_price           yfinance latest available price
  -> analyze               deterministic metrics and eligibility
  -> build_context         compact context pack, not raw SEC dump
  -> synthesize            LangChain chat model explanation
  -> remember              LangMem background memory update
```

## Quick start

```bash
cd value_investing_agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=...
SEC_USER_AGENT="ValueInvestingAgent/0.1 your.real.email@example.com"
LANGSMITH_API_KEY=...
```

Start Postgres with pgvector:

```bash
docker compose up -d
```

Run the agent:

```bash
value-agent AAPL --thread-id demo-aapl --portfolio-value 10000
```

Reuse the same `--thread-id` to preserve threadwise state and LangMem memories:

```bash
value-agent MSFT --thread-id demo-aapl --portfolio-value 10000
```

## Unit sizing rule

The agent cannot know your real risk tolerance from a ticker alone. The default rule is therefore deliberately explicit and configurable:

```text
minimum_units = floor(portfolio_value * max_position_pct * first_tranche_pct / latest_price)
```

Defaults:

```text
portfolio_value      = 10000
max_position_pct     = 0.05
first_tranche_pct    = 0.25
```

For example, with a 10,000 portfolio, a 5% maximum position, and a 25% first tranche, the first tranche budget is 125. If the stock is 50, the minimum units are 2. If the screen fails, the minimum units are 0.

## Eligibility rules

The deterministic screen requires a strong score plus hard gates:

- at least 8 annual SEC data points,
- profitable in at least 8 years or all available years if fewer than 8,
- acceptable leverage,
- Graham-number margin of safety of at least 25%,
- total score at least 70/100.

Scoring components include profitability, operating cash flow, revenue growth, book-equity growth, current ratio, leverage, and valuation.

## Observability

Set these in `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=value-investing-agent
LANGSMITH_API_KEY=...
```

Prompt lifecycle controls:

```bash
PROMPT_NAME=value_investing_screening_prompt
AUTO_PROMPT_OPTIMIZATION=true
PROMPT_OPTIMIZER_KIND=prompt_memory
```

### Automatic Report Evaluation and Prompt Tuning

The agent includes an LLM-as-a-judge module (`src/value_agent/llm_judge.py`) that automatically:

1. **Evaluates each screening report** against:
   - Correctness: Does the report match deterministic metrics?
   - Hallucinations: Are there fabricated numbers or unsupported claims?
   - Format: Do all required sections exist and are they well-formed?
   - Clarity: Is it understandable without excessive jargon?

2. **Triggers prompt updates** if the judge finds:
   - Hallucinations detected
   - Format violations
   - Score below 0.65 (65% quality threshold)
   - Marked as "incorrect" by the judge

3. **Persists new prompt versions** with full audit trail when updates occur.

**No human-in-the-loop feedback needed.** Each run automatically judges and improves the prompt.

Notes:

- The agent loads the active prompt from Postgres at startup.
- If no prompt exists yet, it bootstraps version 1 from `src/value_agent/prompts.py` default text.
- On update, the old row is soft deleted (`is_active=false`, `deleted_at` set) and a new version row is inserted.
- Judge assessments and full report context are stored in `public.system_prompts.feedback_snapshot` for traceability.

The CLI attaches tags and metadata: ticker, thread id, and portfolio value.

## Constant self-improvement

This starter implements two improvement loops:

1. Hot-path memory: each run searches and updates LangMem memories under `("value-agent", thread_id)`.
2. Prompt optimization: `scripts/optimize_prompt.py` can read feedback JSONL and ask LangMem to propose a better analyst prompt.

### Exporting feedback from LangSmith

`scripts/export_langsmith_feedback.py` fetches root runs from your LangSmith project and writes them as `feedback.jsonl` in the exact schema the optimizer expects.

```bash
# Export the 100 most recent runs
python scripts/export_langsmith_feedback.py \
  --project value-investing-agent \
  --limit 100 \
  --output feedback.jsonl
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--project` | `$LANGSMITH_PROJECT` | LangSmith project name |
| `--limit` | `100` | Maximum root runs to fetch |
| `--output` | `feedback.jsonl` | Output file path |
| `--skip-without-feedback` | off | Only export runs that already have LangSmith feedback attached |
| `--feedback-key` | all keys | Restrict to a specific feedback key, e.g. `user_score` |

Examples:

```bash
# Only include runs that have been scored in LangSmith
python scripts/export_langsmith_feedback.py \
  --project value-investing-agent \
  --limit 200 \
  --skip-without-feedback \
  --output feedback.jsonl

# Filter to a specific feedback key
python scripts/export_langsmith_feedback.py \
  --project value-investing-agent \
  --feedback-key user_score \
  --output feedback.jsonl
```

Each output line is a JSON object:

```json
{
  "messages": [
    {"role": "user", "content": "Screen AAPL for value-investing eligibility."},
    {"role": "assistant", "content": "...report..."}
  ],
  "feedback": {"score": 1, "note": "concise and accurate"},
  "run_id": "...",
  "project": "value-investing-agent"
}
```

### Running prompt optimization

Once you have `feedback.jsonl`, run the optimizer:

```bash
python scripts/optimize_prompt.py feedback.jsonl --kind prompt_memory
```

Review the suggested prompt before replacing `src/value_agent/prompts.py`.

## Important limitations

- SEC facts are as-filed XBRL and can contain restatements, taxonomy changes, missing tags, and company-specific reporting choices.
- Some public companies, foreign issuers, banks, insurers, REITs, ADRs, and recent IPOs need sector-specific screens.
- Current market price is fetched via yfinance for convenience; replace `market_data.py` with a licensed market-data provider for production.
- The LLM writes the report, but it does not perform the calculations or change the deterministic decision.

## Research basis

The automatic evaluation and prompt-optimization pipeline draws on the following
published work.

### LLM-as-a-Judge evaluation

| Paper | Venue | What we use |
|---|---|---|
| **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**<br>Zheng et al. (2023) | NeurIPS 2023 | Foundational reference for using an LLM to score another LLM's output |
| **LLM-as-Judge on a Budget**<br>Saha, Wagde & Kveton (2026) · [arXiv:2602.15481](https://arxiv.org/abs/2602.15481) | – | Variance-adaptive sampling: run the judge multiple times; weight by estimated score variance to reduce worst-case estimation error |
| **Bi-Level Prompt Optimization for Multimodal LLM-as-a-Judge**<br>Pan et al. (2026) · [arXiv:2602.11340](https://arxiv.org/abs/2602.11340) | – | Joint optimisation of the generator prompt and the judge prompt; motivates storing the judge prompt as a versioned `system_prompts` row |
| **JAF: Judge Agent Forest**<br>Garg et al. (2026) · [arXiv:2601.22269](https://arxiv.org/abs/2601.22269) | – | Cross-batch consistency checking: evaluate a cohort of related responses together so the judge can flag cross-run inconsistencies |

### Automatic prompt optimisation

| Paper | Venue | What we use |
|---|---|---|
| **Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection (ERM)**<br>Yan et al. (2025) · [arXiv:2411.07446](https://arxiv.org/abs/2411.07446) | ACL 2025 | **Implemented (High Priority).** Historical (trajectory, feedback) pairs stored in `public.system_prompts` are loaded by `load_recent_feedback_examples` and prepended to the current trajectory before each optimizer call. Reduces optimization steps by ~50% and improves output quality. |
| **PrefPO: Pairwise Preference Prompt Optimization**<br>Singhal, Tambwekar & Maamari (2026) · [arXiv:2603.19311](https://arxiv.org/abs/2603.19311) | – | Pairwise comparison of old vs. new prompt outputs before persisting an update; reduces prompt regressions and prompt-hacking by ~50% vs. absolute-score methods |
| **Build, Judge, Optimize**<br>Herrera et al. (2026) · [arXiv:2603.03565](https://arxiv.org/abs/2603.03565) | – | Multi-rubric evaluation decomposed into structured dimensions; motivates the six-section report format and the grounding rule in the system prompt |
| **CentaurTA Studio**<br>Wang, Huang & Dragut (2026) · [arXiv:2604.18589](https://arxiv.org/abs/2604.18589) | – | **Implemented (High Priority).** Persistent alignment principles distilled from feedback; motivates the Grounding Rule and Self-Check block added to the system prompt. Shows removing the feedback loop drops accuracy from 90% → 81%. |
