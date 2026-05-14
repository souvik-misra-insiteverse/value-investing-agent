"""LLM-as-a-Judge for automated report evaluation.

Evaluates screening reports for:
- Correctness: Does the report align with deterministic metrics?
- Hallucinations: Are there fabricated numbers or unsupported claims?
- Format: Does it follow the required output structure?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass(frozen=True)
class JudgmentResult:
    score: float  # 0.0 to 1.0, where 1.0 is perfect
    confidence: float  # 0.0 to 1.0
    correctness: str  # "correct", "questionable", "incorrect"
    hallucinations_detected: bool
    format_valid: bool
    reasoning: str


JUDGE_PROMPT = """You are an expert financial analyst and fact-checker for value-investing screening reports.

Your task is to evaluate a screening report against deterministic metrics and identify issues.

## Evaluation Criteria

1. **Correctness**: Does the report accurately reflect the deterministic metrics provided?
   - Do verdict and score match the analysis?
   - Are P/E, P/B ratios cited correctly?
   - Does the minimum_units calculation align with the formula?
   - Is the holding rule consistent with the eligibility decision?

2. **Hallucinations**: Are there any fabricated or unsupported claims?
   - Numbers that do not appear in the provided metrics?
   - Historical claims not in the 10-year data?
   - Made-up ratios or calculations?
   - False attributions to SEC filings?

3. **Format**: Does the report follow the required structure?
   - Verdict section present?
   - 10-year business performance section?
   - Valuation section with key metrics?
   - Minimum units section?
   - Holding rule section?
   - Caveats section?

4. **Clarity**: Is the report understandable to a layperson without excessive jargon?

## JSON Output

Return ONLY valid JSON with this structure (no markdown, no extra text):

```json
{
  "score": 0.85,
  "confidence": 0.92,
  "correctness": "correct",
  "hallucinations_detected": false,
  "format_valid": true,
  "major_issues": ["issue1", "issue2"],
  "minor_issues": ["issue1"],
  "hallucination_examples": [],
  "reasoning": "Report accurately reflects metrics. All sections present. No fabrications detected."
}
```

**score**: 0.0-1.0. Deduct for: each hallucination (-0.15), each major format issue (-0.10), each minor issue (-0.05).
**confidence**: Your confidence in this judgment (0.0-1.0).
**correctness**: "correct", "questionable", or "incorrect".
**hallucinations_detected**: true if any fabrications found.
**format_valid**: true if all required sections present and well-formed.
**major_issues**: Critical problems (hallucinations, missing required sections, math errors).
**minor_issues**: Improvement areas (clarity, jargon, repetition).
**hallucination_examples**: List specific examples if detected.
**reasoning**: Brief summary of your judgment.
""".strip()


async def judge_report(
    *,
    model_name: str,
    ticker: str,
    company_name: str,
    analysis: dict[str, Any],
    annuals: list[dict[str, Any]],
    report: str,
) -> JudgmentResult:
    """Run LLM judge on a screening report.
    
    Args:
        model_name: LangChain model name (e.g., "openai:gpt-4-mini")
        ticker: Stock ticker
        company_name: Company name
        analysis: Deterministic analysis result dict
        annuals: List of annual data dicts
        report: Generated screening report text
    
    Returns:
        JudgmentResult with score, correctness, hallucination flag, and reasoning
    """
    llm = init_chat_model(model_name)

    context = f"""
## Company: {ticker} ({company_name})

## Deterministic Analysis
{json.dumps(analysis, indent=2, default=str)}

## 10-Year Annual Data (Sample)
{json.dumps(annuals[-3:], indent=2, default=str)}

## Generated Report
{report}
""".strip()

    messages = [
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=context),
    ]

    response = await llm.ainvoke(messages)
    response_text = str(response.content).strip()

    try:
        judgment_raw = json.loads(response_text)
    except json.JSONDecodeError:
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                judgment_raw = json.loads(response_text[start:end])
            else:
                raise
        except (json.JSONDecodeError, ValueError):
            judgment_raw = {
                "score": 0.5,
                "confidence": 0.3,
                "correctness": "questionable",
                "hallucinations_detected": False,
                "format_valid": False,
                "reasoning": f"Failed to parse judge response: {response_text[:200]}",
            }

    return JudgmentResult(
        score=float(judgment_raw.get("score", 0.5)),
        confidence=float(judgment_raw.get("confidence", 0.3)),
        correctness=str(judgment_raw.get("correctness", "questionable")),
        hallucinations_detected=bool(judgment_raw.get("hallucinations_detected", False)),
        format_valid=bool(judgment_raw.get("format_valid", True)),
        reasoning=str(judgment_raw.get("reasoning", "")),
    )
