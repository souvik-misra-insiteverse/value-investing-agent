from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class FeedbackExample:
    """A past (trajectory, feedback) pair loaded from Postgres for use as optimizer memory.

    Implements the exemplar-memory approach from:
    Yan et al. "Efficient and Accurate Prompt Optimization: the Benefit of Memory in
    Exemplar-Guided Reflection." ACL 2025. arXiv:2411.07446
    """

    input_snapshot: dict[str, Any]
    output_snapshot: dict[str, Any]
    feedback_snapshot: dict[str, Any]
    version: int
    optimizer_reason: str


@dataclass(frozen=True)
class PromptRecord:
    id: int
    prompt_name: str
    version: int
    prompt_text: str
    created_at: datetime


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.system_prompts (
    id BIGSERIAL PRIMARY KEY,
    prompt_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by TEXT NOT NULL DEFAULT 'value-agent',
    source_run_id TEXT NULL,
    input_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    feedback_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    deleted_at TIMESTAMPTZ NULL,
    deleted_by TEXT NULL,
    deleted_reason TEXT NULL,
    CONSTRAINT uq_system_prompts_name_version UNIQUE (prompt_name, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_system_prompts_active
ON public.system_prompts (prompt_name)
WHERE is_active = TRUE AND deleted_at IS NULL;
"""

CREATE_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS public.system_prompts_audit (
    audit_id BIGSERIAL PRIMARY KEY,
    prompt_id BIGINT NOT NULL,
    operation TEXT NOT NULL,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    changed_by TEXT NOT NULL DEFAULT current_user,
    old_row JSONB,
    new_row JSONB
);

CREATE OR REPLACE FUNCTION public.fn_audit_system_prompts()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO public.system_prompts_audit (prompt_id, operation, old_row, new_row)
        VALUES (NEW.id, TG_OP, NULL, to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO public.system_prompts_audit (prompt_id, operation, old_row, new_row)
        VALUES (NEW.id, TG_OP, to_jsonb(OLD), to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO public.system_prompts_audit (prompt_id, operation, old_row, new_row)
        VALUES (OLD.id, TG_OP, to_jsonb(OLD), NULL);
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger
        WHERE tgname = 'trg_audit_system_prompts'
    ) THEN
        CREATE TRIGGER trg_audit_system_prompts
        AFTER INSERT OR UPDATE OR DELETE ON public.system_prompts
        FOR EACH ROW
        EXECUTE FUNCTION public.fn_audit_system_prompts();
    END IF;
END;
$$;
"""


async def ensure_prompt_tables(conn: psycopg.AsyncConnection) -> None:
    async with conn.cursor() as cur:
        await cur.execute(CREATE_TABLE_SQL)
        await cur.execute(CREATE_AUDIT_SQL)


async def _fetch_active_prompt_row(
    conn: psycopg.AsyncConnection,
    *,
    prompt_name: str,
    lock_row: bool,
) -> Optional[dict[str, Any]]:
    sql = """
    SELECT id, prompt_name, version, prompt_text, created_at
    FROM public.system_prompts
    WHERE prompt_name = %s AND is_active = TRUE AND deleted_at IS NULL
    ORDER BY version DESC
    LIMIT 1
    """
    if lock_row:
        sql += " FOR UPDATE"

    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(sql, (prompt_name,))
        return await cur.fetchone()


async def _insert_prompt(
    conn: psycopg.AsyncConnection,
    *,
    prompt_name: str,
    version: int,
    prompt_text: str,
    created_by: str,
    reason: str,
    run_id: Optional[str],
    input_snapshot: dict[str, Any],
    output_snapshot: dict[str, Any],
    feedback_snapshot: dict[str, Any],
    metadata: dict[str, Any],
) -> PromptRecord:
    sql = """
    INSERT INTO public.system_prompts (
        prompt_name,
        version,
        prompt_text,
        is_active,
        created_by,
        source_run_id,
        input_snapshot,
        output_snapshot,
        feedback_snapshot,
        metadata
    )
    VALUES (%s, %s, %s, TRUE, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
    RETURNING id, prompt_name, version, prompt_text, created_at
    """
    merged_metadata = dict(metadata)
    merged_metadata.setdefault("change_reason", reason)
    input_json = json.dumps(input_snapshot, ensure_ascii=True, default=str)
    output_json = json.dumps(output_snapshot, ensure_ascii=True, default=str)
    feedback_json = json.dumps(feedback_snapshot, ensure_ascii=True, default=str)
    metadata_json = json.dumps(merged_metadata, ensure_ascii=True, default=str)

    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            sql,
            (
                prompt_name,
                version,
                prompt_text,
                created_by,
                run_id,
                input_json,
                output_json,
                feedback_json,
                metadata_json,
            ),
        )
        row = await cur.fetchone()

    if row is None:
        raise RuntimeError("Failed to insert system prompt row.")

    return PromptRecord(
        id=int(row["id"]),
        prompt_name=str(row["prompt_name"]),
        version=int(row["version"]),
        prompt_text=str(row["prompt_text"]),
        created_at=row["created_at"],
    )


async def load_or_bootstrap_active_prompt(
    database_url: str,
    *,
    prompt_name: str,
    fallback_prompt: str,
    created_by: str = "value-agent",
) -> PromptRecord:
    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        async with conn.transaction():
            await ensure_prompt_tables(conn)
            async with conn.cursor() as cur:
                await cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (prompt_name,))

            row = await _fetch_active_prompt_row(conn, prompt_name=prompt_name, lock_row=True)
            if row is None:
                created = await _insert_prompt(
                    conn,
                    prompt_name=prompt_name,
                    version=1,
                    prompt_text=fallback_prompt.strip(),
                    created_by=created_by,
                    reason="bootstrap_from_default",
                    run_id=None,
                    input_snapshot={},
                    output_snapshot={},
                    feedback_snapshot={},
                    metadata={"seed": "default_prompt"},
                )
                return created

            return PromptRecord(
                id=int(row["id"]),
                prompt_name=str(row["prompt_name"]),
                version=int(row["version"]),
                prompt_text=str(row["prompt_text"]),
                created_at=row["created_at"],
            )


async def create_new_prompt_version(
    database_url: str,
    *,
    prompt_name: str,
    new_prompt_text: str,
    changed_by: str,
    reason: str,
    run_id: Optional[str],
    input_snapshot: dict[str, Any],
    output_snapshot: dict[str, Any],
    feedback_snapshot: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[PromptRecord, bool]:
    cleaned_prompt = new_prompt_text.strip()
    if not cleaned_prompt:
        raise ValueError("Cannot persist an empty system prompt.")

    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        async with conn.transaction():
            await ensure_prompt_tables(conn)
            async with conn.cursor() as cur:
                await cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (prompt_name,))

            active = await _fetch_active_prompt_row(conn, prompt_name=prompt_name, lock_row=True)
            if active is None:
                created = await _insert_prompt(
                    conn,
                    prompt_name=prompt_name,
                    version=1,
                    prompt_text=cleaned_prompt,
                    created_by=changed_by,
                    reason=reason,
                    run_id=run_id,
                    input_snapshot=input_snapshot,
                    output_snapshot=output_snapshot,
                    feedback_snapshot=feedback_snapshot,
                    metadata=metadata,
                )
                return created, True

            if str(active["prompt_text"]).strip() == cleaned_prompt:
                return (
                    PromptRecord(
                        id=int(active["id"]),
                        prompt_name=str(active["prompt_name"]),
                        version=int(active["version"]),
                        prompt_text=str(active["prompt_text"]),
                        created_at=active["created_at"],
                    ),
                    False,
                )

            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE public.system_prompts
                    SET is_active = FALSE,
                        deleted_at = now(),
                        deleted_by = %s,
                        deleted_reason = %s
                    WHERE id = %s
                    """,
                    (changed_by, reason, int(active["id"])),
                )

            created = await _insert_prompt(
                conn,
                prompt_name=prompt_name,
                version=int(active["version"]) + 1,
                prompt_text=cleaned_prompt,
                created_by=changed_by,
                reason=reason,
                run_id=run_id,
                input_snapshot=input_snapshot,
                output_snapshot=output_snapshot,
                feedback_snapshot=feedback_snapshot,
                metadata=metadata,
            )
            return created, True


async def load_recent_feedback_examples(
    database_url: str,
    *,
    prompt_name: str,
    limit: int = 5,
    only_problems: bool = True,
) -> list[FeedbackExample]:
    """Return the most recent past (trajectory, feedback) pairs for a prompt.

    These are used as historical memory to guide the LangMem prompt optimizer,
    following the Exemplar-Guided Reflection with Memory (ERM) approach from:
    Yan et al. "Efficient and Accurate Prompt Optimization: the Benefit of Memory in
    Exemplar-Guided Reflection." ACL 2025. arXiv:2411.07446

    When ``only_problems=True`` (default) we restrict to rows where a quality issue
    was detected (optimizer_reason recorded in metadata), so the optimizer receives
    only informative negative examples and not redundant already-passing runs.

    Args:
        database_url: Postgres connection string.
        prompt_name: The prompt name, e.g. "value_investing_screening_prompt".
        limit: Maximum number of examples to return (k in ERM).
        only_problems: When True, only return rows that triggered a prompt update.

    Returns:
        List of FeedbackExample ordered newest-first, empty if none found.
    """
    if only_problems:
        sql = """
        SELECT
            version,
            input_snapshot,
            output_snapshot,
            feedback_snapshot,
            metadata
        FROM public.system_prompts
        WHERE
            prompt_name = %s
            AND deleted_at IS NOT NULL
            AND (metadata->>'optimizer_reason') IS NOT NULL
            AND input_snapshot  != '{}'::jsonb
            AND output_snapshot != '{}'::jsonb
        ORDER BY created_at DESC
        LIMIT %s
        """
    else:
        sql = """
        SELECT
            version,
            input_snapshot,
            output_snapshot,
            feedback_snapshot,
            metadata
        FROM public.system_prompts
        WHERE
            prompt_name = %s
            AND input_snapshot  != '{}'::jsonb
            AND output_snapshot != '{}'::jsonb
        ORDER BY created_at DESC
        LIMIT %s
        """

    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, (prompt_name, limit))
            rows = await cur.fetchall()

    examples: list[FeedbackExample] = []
    for row in rows:
        def _parse(v: Any) -> dict[str, Any]:
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except Exception:
                    return {}
            return {}

        meta = _parse(row.get("metadata"))
        examples.append(
            FeedbackExample(
                input_snapshot=_parse(row.get("input_snapshot")),
                output_snapshot=_parse(row.get("output_snapshot")),
                feedback_snapshot=_parse(row.get("feedback_snapshot")),
                version=int(row.get("version", 0)),
                optimizer_reason=str(meta.get("optimizer_reason", "")),
            )
        )
    return examples
