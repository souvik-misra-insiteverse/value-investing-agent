from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from .config import Settings
from .graph import make_app
from .prompt_optimizer import PromptFeedback, maybe_optimize_and_persist_prompt
from .prompt_store import PromptRecord, load_or_bootstrap_active_prompt
from .prompts import DEFAULT_SYSTEM_PROMPT
from .llm_judge import judge_report
from langgraph.store.postgres import AsyncPostgresStore
from langchain.embeddings import init_embeddings
# pyrefly: ignore [missing-import]
from langfuse.langchain import CallbackHandler

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the value investing screening agent.")
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Persistent LangGraph thread id. Reuse it to preserve threadwise memory.",
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=None,
        help="Portfolio value used only for rules-based unit sizing.",
    )
    return parser.parse_args()


async def async_main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_env(portfolio_value=args.portfolio_value)
    thread_id = args.thread_id or f"thread-{uuid.uuid4()}"
    active_prompt = await load_or_bootstrap_active_prompt(
        settings.database_url,
        prompt_name=settings.prompt_name,
        fallback_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    initial_state = {
        "ticker": args.ticker.upper().strip(),
        "portfolio_value": settings.default_portfolio_value,
    }
    config = {
        "run_name": thread_id,
        "configurable": {"thread_id": thread_id},
        "tags": ["value-investing", "screening", initial_state["ticker"]],
        "metadata": {
            "langfuse_session_id": thread_id,
            "ticker": initial_state["ticker"],
            "thread_id": thread_id,
            "portfolio_value": settings.default_portfolio_value,
            "prompt_name": active_prompt.prompt_name,
            "prompt_version": active_prompt.version,
        },
    }

    langfuse_handler = CallbackHandler()
    config["callbacks"] = [langfuse_handler]

    async with make_app(settings, system_prompt=active_prompt.prompt_text) as app:
        result = await app.ainvoke(initial_state, config=config)

    console.rule(f"{initial_state['ticker']} value screen")
    console.print(result["report"])
    if "final_prompt" in result:
        console.rule("final prompt used (RAG)")
        console.print(result["final_prompt"])
    console.rule("run metadata")
    console.print(
        {
            "thread_id": thread_id,
            "ticker": initial_state["ticker"],
            "prompt_name": active_prompt.prompt_name,
            "prompt_version": active_prompt.version,
        }
    )

    # Auto-judge the report for correctness and hallucinations
    console.rule("llm judge: evaluating report")
    judge_result = await judge_report(
        model_name=settings.model_name,
        ticker=result.get("ticker", ""),
        company_name=result.get("company_name", ""),
        analysis=result.get("analysis", {}),
        annuals=result.get("annuals", []),
        report=result.get("report", ""),
    )
    console.print(
        {
            "score": judge_result.score,
            "confidence": judge_result.confidence,
            "correctness": judge_result.correctness,
            "hallucinations_detected": judge_result.hallucinations_detected,
            "format_valid": judge_result.format_valid,
            "reasoning": judge_result.reasoning,
        }
    )

    # Auto-update prompt based on judge assessment (no human loop)
    auto_feedback = PromptFeedback(
        score=judge_result.score,
        note=f"Judge: {judge_result.reasoning}",
    )
    update_result = await maybe_optimize_and_persist_prompt(
        settings=settings,
        current_prompt=active_prompt,
        user_input=initial_state,
        run_output={"report": result.get("report", ""), "analysis": result.get("analysis")},
        feedback=auto_feedback,
        run_id=thread_id,
        judge_result=judge_result,
    )
    if update_result.updated:
        console.rule("prompt update")
        console.print(
            {
                "updated": True,
                "previous_version": update_result.previous_version,
                "current_version": update_result.current_version,
                "reason": update_result.reason,
            }
        )

    if judge_result.score > 0.97 and judge_result.format_valid:
        embedding = init_embeddings(settings.embedding_model)
        async with AsyncPostgresStore.from_conn_string(
            settings.database_url,
            index={"dims": 384, "embed": embedding},
        ) as store:
            await store.setup()
            await store.aput(
                namespace=("golden_examples",),
                key=initial_state["ticker"],
                value={"context": result.get("context_pack", ""), "report": result.get("report", "")}
            )
        console.rule("few-shot store")
        console.print(f"Saved {initial_state['ticker']} as a golden example for future RAG.")


def main() -> None:
    # Psycopg async mode is not compatible with the default Proactor loop on Windows.
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
