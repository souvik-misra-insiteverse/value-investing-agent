"""Migrate the active system prompt in Postgres to the current DEFAULT_SYSTEM_PROMPT.

This is Option A for applying a code-side prompt change to an existing database.
The old prompt row is soft-deleted (is_active=false, deleted_at set) and the new
text is inserted as the next version, preserving the full audit trail.

Usage:
    python scripts/update_prompt.py [--dry-run]

Flags:
    --dry-run   Print what would happen without writing anything to Postgres.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from difflib import unified_diff

from dotenv import load_dotenv

from value_agent.config import Settings
from value_agent.prompt_store import (
    create_new_prompt_version,
    load_or_bootstrap_active_prompt,
)
from value_agent.prompts import DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate active system prompt to the current code default.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the diff and exit without writing to the database.",
    )
    return parser.parse_args()


def _diff(old: str, new: str) -> str:
    lines = list(
        unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="db (active)",
            tofile="code (DEFAULT_SYSTEM_PROMPT)",
        )
    )
    return "".join(lines) if lines else "(no diff — texts are identical)"


async def async_main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_env()
    new_text = DEFAULT_SYSTEM_PROMPT.strip()

    # Load whatever is currently active (or bootstrap if the table is empty).
    active = await load_or_bootstrap_active_prompt(
        settings.database_url,
        prompt_name=settings.prompt_name,
        fallback_prompt=new_text,
    )

    diff_output = _diff(active.prompt_text, new_text)
    print(f"Prompt name : {settings.prompt_name}")
    print(f"Current version in DB : v{active.version}")
    print()
    print("--- diff ---")
    print(diff_output)
    print("------------")

    if active.prompt_text.strip() == new_text:
        print("\nDB is already up to date. Nothing to do.")
        return

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return

    persisted, updated = await create_new_prompt_version(
        settings.database_url,
        prompt_name=settings.prompt_name,
        new_prompt_text=new_text,
        changed_by="update_prompt_script",
        reason="code_default_migration",
        run_id=None,
        input_snapshot={},
        output_snapshot={},
        feedback_snapshot={},
        metadata={
            "migration": "update_prompt.py",
            "previous_version": active.version,
        },
    )

    if updated:
        print(f"\nDone. Active prompt is now v{persisted.version}.")
    else:
        print("\nNo change persisted (prompt text was identical after trim).")


def main() -> None:
    import selectors
    import sys

    # psycopg async requires SelectorEventLoop on Windows (ProactorEventLoop is
    # the default on Python 3.8+ / Windows and is incompatible with psycopg).
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
