# Prompt Optimization Experiment

This document details an intentional edge-case experiment designed to test and trigger the autonomous prompt optimization mechanism (`LangMem`) within the `value_investing_agent`.

## Background

The agent relies on an **LLM Judge** (`src/value_agent/prompt_optimizer.py`) to auto-evaluate the generated investment reports against the deterministic JSON analysis computed by the underlying Python logic.

If the LLM Judge detects a severe issue—such as hallucinations, incorrect math, or missing required markdown sections—it triggers the `LangMem` metaprompt optimizer. The optimizer then takes the failing prompt, the input variables, and the Judge's negative feedback, and automatically writes and persists a newly optimized prompt to the Postgres database.

## The Challenge: Azure OpenAI Robustness

In our testing, we attempted to intentionally crash the prompt by passing edge-case tickers:
1. **`DJT` (New IPO/SPAC)**: Minimal data (0/2 years of profitability).
2. **`MULN` (Penny Stock)**: Extreme integer calculations for portfolio units.

**Observation**: The Azure OpenAI `gpt-5.4-mini` (or equivalent high-tier model) is incredibly robust. It flawlessly handled missing data, correctly interpolated N/A fields, and strictly adhered to the formatting guidelines despite the mathematical chaos in the underlying deterministic JSON. The LLM Judge awarded these runs scores of `0.97` or higher, meaning **no prompt optimization was triggered**.

## The Solution: The "Soft-Poisoning" Strategy

To force the `LangMem` optimizer to trigger without triggering Azure OpenAI's strict "Jailbreak" content safety filters, we utilized a "Soft Poisoning" strategy directly against the active prompt in the database.

### Step 1: Poisoning the Prompt
We ran a script to overwrite the active `system_prompt` in the Postgres database with the following instruction:
> *"You are a financial analyst. Write a very brief 1-paragraph summary of the company. Do NOT include any sections for Valuation, Verdict, or Minimum Units. Keep it under 50 words."*

This completely contradicted the strict structural rules the LLM Judge was looking for, but was harmless enough to bypass Azure's content filters.

### Step 2: Triggering the Failure
We executed the agent for Microsoft (`MSFT`):
```bash
python -m value_agent.cli MSFT --thread-id msft-soft-poison-test
```

### Step 3: The Result
As instructed by the poisoned prompt, the LLM generated a single paragraph of text, omitting all required sections (Valuation, Verdict, Margin of Safety, etc.). 

The LLM Judge caught this immediately and evaluated the report:
```json
{
    "score": 0.86,
    "confidence": 0.98,
    "correctness": "correct",
    "hallucinations_detected": false,
    "format_valid": false,
    "reasoning": "The narrative matches the deterministic conclusion... However, the output fails the required report structure by omitting the mandated sections and explicit metric discussion, so format_valid is false."
}
```

Because `format_valid` was `false`, `prompt_optimizer.py` initiated the metaprompt update process:
```json
{
    "updated": true,
    "previous_version": 1,
    "current_version": 2,
    "reason": "auto_prompt_update:judge_format_invalid"
}
```

## Conclusion

This experiment successfully demonstrated the end-to-end self-healing capability of the agent. By intentionally breaking the prompt, we proved that the LLM Judge accurately monitors formatting and correctness, and successfully invokes `LangMem` to patch and increment the prompt version in the Postgres database without any human intervention.
