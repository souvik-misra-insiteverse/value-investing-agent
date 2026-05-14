"""Prompt optimization helpers.

This module is now reusable from runtime code. The CLI entry remains only for
backfill/recovery workflows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from value_agent.config import Settings
from value_agent.prompt_optimizer import PromptFeedback, maybe_optimize_and_persist_prompt
from value_agent.prompt_store import load_or_bootstrap_active_prompt
from value_agent.prompts import DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("feedback_jsonl", type=Path)
    parser.add_argument("--kind", default="prompt_memory", choices=["prompt_memory", "metaprompt", "gradient"])
    return parser.parse_args()


def load_feedback_trajectories(feedback_jsonl: Path) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    with feedback_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            trajectories.append(json.loads(line))
    return trajectories


async def apply_feedback_jsonl(
    *,
    settings: Settings,
    records: list[dict[str, Any]],
) -> int:
    updated = 0
    active_prompt = await load_or_bootstrap_active_prompt(
        settings.database_url,
        prompt_name=settings.prompt_name,
        fallback_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    for idx, record in enumerate(records, start=1):
        messages = record.get("messages") or []
        user_message = next((m for m in messages if m.get("role") == "user"), {})
        assistant_message = next((m for m in messages if m.get("role") == "assistant"), {})
        feedback = record.get("feedback") or {}
        result = await maybe_optimize_and_persist_prompt(
            settings=settings,
            current_prompt=active_prompt,
            user_input={"message": user_message.get("content", "")},
            run_output={"report": assistant_message.get("content", "")},
            feedback=PromptFeedback(
                score=feedback.get("score") if isinstance(feedback.get("score"), (int, float)) else None,
                note=str(feedback.get("note") or ""),
            ),
            run_id=str(record.get("run_id") or f"jsonl-{idx}"),
        )
        if result.updated:
            updated += 1
            active_prompt = await load_or_bootstrap_active_prompt(
                settings.database_url,
                prompt_name=settings.prompt_name,
                fallback_prompt=DEFAULT_SYSTEM_PROMPT,
            )

    return updated


async def main() -> None:
    load_dotenv()
    args = parse_args()
    os.environ["PROMPT_OPTIMIZER_KIND"] = args.kind
    settings = Settings.from_env()
    records = load_feedback_trajectories(args.feedback_jsonl)
    updated = await apply_feedback_jsonl(settings=settings, records=records)
    print(f"Applied {updated} prompt updates from {args.feedback_jsonl}")


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
