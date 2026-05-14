from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional
from typing import Literal, cast

from langmem import create_prompt_optimizer

from .config import Settings
from .prompt_store import FeedbackExample, PromptRecord, create_new_prompt_version, load_recent_feedback_examples


@dataclass(frozen=True)
class PromptFeedback:
    score: Optional[float]
    note: str


@dataclass(frozen=True)
class PromptUpdateDecision:
    should_update: bool
    reason: str


@dataclass(frozen=True)
class PromptUpdateResult:
    updated: bool
    reason: str
    previous_version: int
    current_version: int


def _serialize_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, default=str)


def _score_indicates_problem(score: Optional[float]) -> bool:
    """Judge scores are 0.0-1.0; human feedback is 0 or 1.
    Accept both formats for backward compatibility.
    """
    if score is None:
        return False
    # Judge score: 0.0-1.0, problem if < 0.65 (65% quality threshold)
    if 0.0 <= score <= 1.0:
        return score < 0.65
    # Legacy human score: 0 or 1+
    if score <= 0:
        return True
    if score > 1:
        return score <= 2
    return score < 0.5


def evaluate_prompt_update_need(
    *,
    report: str,
    feedback: PromptFeedback,
    judge_result: Optional[Any] = None,
) -> PromptUpdateDecision:
    """Decide if prompt should be updated based on feedback and optional judge assessment.
    
    Judge results take priority. If no judge result, fall back to feedback score/note.
    """
    # Judge-based evaluation (primary)
    if judge_result is not None:
        if hasattr(judge_result, "hallucinations_detected") and judge_result.hallucinations_detected:
            return PromptUpdateDecision(True, "judge_hallucinations_detected")
        if hasattr(judge_result, "format_valid") and not judge_result.format_valid:
            return PromptUpdateDecision(True, "judge_format_invalid")
        if hasattr(judge_result, "score") and judge_result.score < 0.65:
            return PromptUpdateDecision(True, "judge_low_quality_score")
        if hasattr(judge_result, "correctness") and judge_result.correctness == "incorrect":
            return PromptUpdateDecision(True, "judge_correctness_incorrect")
        return PromptUpdateDecision(False, "judge_quality_acceptable")

    # Legacy human-feedback fallback
    note = (feedback.note or "").strip().lower()
    if _score_indicates_problem(feedback.score):
        return PromptUpdateDecision(True, "low_feedback_score")

    trigger_terms = (
        "wrong",
        "incorrect",
        "halluc",
        "verbose",
        "confusing",
        "unclear",
        "jargon",
        "format",
        "missing",
        "incomplete",
    )
    if any(term in note for term in trigger_terms):
        return PromptUpdateDecision(True, "feedback_note_quality_issue")

    # Guardrails for report format drift.
    required_markers = ("Verdict", "Valuation", "Minimum", "Caveats")
    missing = [marker for marker in required_markers if marker.lower() not in report.lower()]
    if missing:
        return PromptUpdateDecision(True, "output_missing_required_sections")

    return PromptUpdateDecision(False, "no_prompt_change_needed")


def _extract_prompt_text(result: Any) -> Optional[str]:
    if isinstance(result, str):
        text = result.strip()
        return text or None

    if isinstance(result, dict):
        for key in ("prompt", "improved_prompt", "text", "content", "value"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _examples_to_trajectories(
    examples: list[FeedbackExample],
) -> list[tuple[list[dict[str, str]], dict[str, Any]]]:
    """Convert historical FeedbackExamples into the trajectory format expected by
    the LangMem prompt optimizer.

    Historical examples are prepended (oldest-relevant first) so the optimizer
    sees prior failure patterns as background context before the current run,
    following the ERM memory mechanism from:
    Yan et al. "Efficient and Accurate Prompt Optimization: the Benefit of Memory
    in Exemplar-Guided Reflection." ACL 2025. arXiv:2411.07446
    """
    result = []
    for ex in reversed(examples):  # oldest first
        fb = ex.feedback_snapshot
        result.append(
            (
                [
                    {"role": "user", "content": _serialize_json(ex.input_snapshot)},
                    {"role": "assistant", "content": _serialize_json(ex.output_snapshot)},
                ],
                {
                    "score": fb.get("score"),
                    "note": fb.get("note", ex.optimizer_reason),
                },
            )
        )
    return result


async def maybe_optimize_and_persist_prompt(
    *,
    settings: Settings,
    current_prompt: PromptRecord,
    user_input: dict[str, Any],
    run_output: dict[str, Any],
    feedback: PromptFeedback,
    run_id: Optional[str] = None,
    judge_result: Optional[Any] = None,
) -> PromptUpdateResult:
    decision = evaluate_prompt_update_need(
        report=str(run_output.get("report", "")),
        feedback=feedback,
        judge_result=judge_result,
    )
    if not settings.auto_prompt_optimization:
        return PromptUpdateResult(False, "auto_prompt_optimization_disabled", current_prompt.version, current_prompt.version)

    if not decision.should_update:
        return PromptUpdateResult(False, decision.reason, current_prompt.version, current_prompt.version)

    # Load historical problem runs as memory exemplars (ERM, arXiv:2411.07446).
    # Prepend up to 5 past negative trajectories so the optimizer sees recurring
    # failure patterns in addition to the current run.
    historical_examples = await load_recent_feedback_examples(
        settings.database_url,
        prompt_name=settings.prompt_name,
        limit=5,
        only_problems=True,
    )
    historical_trajectories = _examples_to_trajectories(historical_examples)

    current_trajectory = (
        [
            {"role": "user", "content": _serialize_json(user_input)},
            {"role": "assistant", "content": _serialize_json(run_output)},
        ],
        {"score": feedback.score, "note": feedback.note},
    )
    # Historical memory first, current run last so it is the most salient signal.
    trajectories = historical_trajectories + [current_trajectory]

    kind = settings.prompt_optimizer_kind.strip().lower()
    allowed_kinds = {"prompt_memory", "metaprompt", "gradient"}
    if kind not in allowed_kinds:
        kind = "prompt_memory"

    optimizer = create_prompt_optimizer(
        settings.model_name,
        kind=cast(Literal["prompt_memory", "metaprompt", "gradient"], kind),
    )
    payload: Any = {
        "trajectories": trajectories,
        "prompt": {
            "name": settings.prompt_name,
            "prompt": current_prompt.prompt_text,
            "update_instructions": (
                "Make minimal changes. Preserve deterministic-metrics dependence, "
                "do not allow personalized financial advice, and preserve required output sections."
            ),
            "when_to_update": "Only update when quality issues are observed in feedback or output format.",
        },
    }
    optimized = await optimizer.ainvoke(payload)
    new_prompt = _extract_prompt_text(optimized)
    if not new_prompt:
        return PromptUpdateResult(False, "optimizer_returned_no_prompt", current_prompt.version, current_prompt.version)

    metadata = {
        "optimizer_kind": settings.prompt_optimizer_kind,
        "optimizer_reason": decision.reason,
        "history_examples_used": len(historical_examples),
    }
    reason = f"auto_prompt_update:{decision.reason}"
    persisted, updated = await create_new_prompt_version(
        settings.database_url,
        prompt_name=settings.prompt_name,
        new_prompt_text=new_prompt,
        changed_by="value-agent",
        reason=reason,
        run_id=run_id,
        input_snapshot=user_input,
        output_snapshot=run_output,
        feedback_snapshot={"score": feedback.score, "note": feedback.note},
        metadata=metadata,
    )
    return PromptUpdateResult(
        updated=updated,
        reason=reason if updated else "optimizer_prompt_identical",
        previous_version=current_prompt.version,
        current_version=persisted.version,
    )
