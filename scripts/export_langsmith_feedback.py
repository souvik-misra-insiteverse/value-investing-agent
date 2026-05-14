"""Export LangSmith runs to feedback JSONL for prompt optimization.

This module now provides reusable functions that can be invoked by runtime code.
The CLI mode is preserved for backfill/recovery workflows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LangSmith runs into feedback.jsonl format.")
    parser.add_argument(
        "--project",
        default=None,
        help="LangSmith project name. Defaults to LANGSMITH_PROJECT from .env.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of root runs to fetch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("feedback.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--skip-without-feedback",
        action="store_true",
        help="Only export runs that already have LangSmith feedback attached.",
    )
    parser.add_argument(
        "--feedback-key",
        default=None,
        help="If set, only use feedback rows with this key (for example: user_score).",
    )
    return parser.parse_args()


def _json_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return str(value)


def _extract_user_content(inputs: Any, metadata: dict[str, Any] | None) -> str:
    if isinstance(inputs, dict):
        if isinstance(inputs.get("messages"), list) and inputs["messages"]:
            return _json_string(inputs["messages"][0], default="")
        for key in ("query", "question", "input", "prompt", "ticker"):
            if key in inputs and inputs[key] is not None:
                if key == "ticker":
                    return f"Screen {str(inputs[key]).upper().strip()} for value-investing eligibility."
                return _json_string(inputs[key], default="")
    if metadata and metadata.get("ticker"):
        ticker = str(metadata["ticker"]).upper().strip()
        return f"Screen {ticker} for value-investing eligibility."
    return _json_string(inputs, default="")


def _extract_assistant_content(outputs: Any) -> str:
    if isinstance(outputs, dict):
        for key in ("report", "output_text", "answer", "content", "text"):
            if key in outputs and outputs[key] is not None:
                return _json_string(outputs[key], default="")
        if isinstance(outputs.get("messages"), list) and outputs["messages"]:
            return _json_string(outputs["messages"][-1], default="")
    return _json_string(outputs, default="")


def _choose_feedback(feedback_rows: list[Any], feedback_key: str | None) -> dict[str, Any] | None:
    filtered = []
    for fb in feedback_rows:
        key = getattr(fb, "key", None)
        if feedback_key and key != feedback_key:
            continue
        filtered.append(fb)

    if not filtered:
        return None

    # Prefer explicit numeric score entries first.
    numeric = [fb for fb in filtered if isinstance(getattr(fb, "score", None), (int, float, bool))]
    chosen = numeric[0] if numeric else filtered[0]

    score = getattr(chosen, "score", None)
    if isinstance(score, bool):
        score = int(score)

    note = getattr(chosen, "comment", None)
    if not note:
        value = getattr(chosen, "value", None)
        note = _json_string(value, default="")

    return {"score": score, "note": note or ""}


def export_feedback_records(
    *,
    project_name: str,
    limit: int,
    skip_without_feedback: bool,
    feedback_key: str | None,
    client: Client | None = None,
) -> tuple[list[dict[str, Any]], int]:
    client = client or Client()
    runs = list(
        client.list_runs(
            project_name=project_name,
            is_root=True,
            error=False,
            limit=limit,
            select=["id", "inputs", "outputs", "extra", "start_time"],
        )
    )

    if not runs:
        return [], 0

    run_ids = [run.id for run in runs]
    feedback_rows = list(client.list_feedback(run_ids=run_ids, limit=max(100, limit * 5)))

    feedback_by_run: dict[str, list[Any]] = {}
    for fb in feedback_rows:
        rid = getattr(fb, "run_id", None)
        if rid is None:
            continue
        feedback_by_run.setdefault(str(rid), []).append(fb)

    records: list[dict[str, Any]] = []
    skipped = 0
    for run in runs:
        run_feedback = feedback_by_run.get(str(run.id), [])
        chosen_feedback = _choose_feedback(run_feedback, feedback_key)
        if skip_without_feedback and chosen_feedback is None:
            skipped += 1
            continue

        run_extra = getattr(run, "extra", None) or {}
        run_metadata = run_extra.get("metadata") if isinstance(run_extra, dict) else None
        user_content = _extract_user_content(getattr(run, "inputs", None), run_metadata)
        assistant_content = _extract_assistant_content(getattr(run, "outputs", None))
        if not assistant_content:
            skipped += 1
            continue

        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "feedback": chosen_feedback or {"score": None, "note": ""},
                "run_id": str(run.id),
                "project": project_name,
            }
        )

    return records, skipped


def main() -> None:
    load_dotenv()
    args = parse_args()

    project_name = args.project or os.getenv("LANGSMITH_PROJECT")
    if not project_name:
        raise RuntimeError("Provide --project or set LANGSMITH_PROJECT in .env.")

    records, skipped = export_feedback_records(
        project_name=project_name,
        limit=args.limit,
        skip_without_feedback=args.skip_without_feedback,
        feedback_key=args.feedback_key,
    )
    if not records:
        print(f"No runs found for project '{project_name}'.")
        return

    exported = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            exported += 1

    print(f"Exported {exported} records to {args.output}")
    if skipped:
        print(f"Skipped {skipped} runs (missing required content or feedback filter).")


if __name__ == "__main__":
    main()
