from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.postgres import AsyncPostgresStore
from langmem import create_memory_store_manager

from .config import Settings
from .fundamentals import collect_annual_fundamentals, compact_fundamentals_table
from .market_data import get_last_price
from .metrics import analyze_value_case
from .prompts import MEMORY_INSTRUCTIONS
from .sec import SecClient


class ValueAgentState(TypedDict, total=False):
    ticker: str
    company_name: str
    cik: str
    portfolio_value: float
    price: Optional[float]
    companyfacts: dict[str, Any]
    annuals: list[dict[str, Any]]
    analysis: dict[str, Any]
    memories: list[str]
    context_pack: str
    report: str
    warnings: list[str]


def build_graph(
    settings: Settings,
    *,
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
    system_prompt: str,
):
    sec_client = SecClient(settings.sec_user_agent)
    llm = init_chat_model(settings.model_name)
    memory_manager = create_memory_store_manager(
        settings.model_name,
        namespace=("value-agent", "{thread_id}"),
        store=store,
        instructions=MEMORY_INSTRUCTIONS,
        query_limit=5,
        enable_deletes=True,
    )

    async def load_memory(state: ValueAgentState, config: RunnableConfig) -> ValueAgentState:
        ticker = state["ticker"].upper().strip()
        memories = await memory_manager.asearch(
            query=f"value investing screening preferences, feedback, thresholds, and prior notes for {ticker}",
            config=config,
        )
        return {"ticker": ticker, "memories": [str(m) for m in memories]}

    async def fetch_sec(state: ValueAgentState) -> ValueAgentState:
        cik, company_name = await sec_client.lookup_cik(state["ticker"])
        facts = await sec_client.company_facts(cik)
        return {"cik": cik, "company_name": company_name, "companyfacts": facts}

    async def fetch_price(state: ValueAgentState) -> ValueAgentState:
        price = await get_last_price(state["ticker"])
        warnings = [] if price else ["Latest market price was unavailable; valuation and units may be incomplete."]
        return {"price": price, "warnings": warnings}

    async def analyze(state: ValueAgentState) -> ValueAgentState:
        annuals = collect_annual_fundamentals(state["companyfacts"], years=10)
        portfolio_value = float(state.get("portfolio_value") or settings.default_portfolio_value)
        analysis = analyze_value_case(
            annuals,
            price=state.get("price"),
            portfolio_value=portfolio_value,
            max_position_pct=settings.max_position_pct,
            first_tranche_pct=settings.first_tranche_pct,
        )
        return {"annuals": annuals, "analysis": analysis, "portfolio_value": portfolio_value}

    async def build_context(state: ValueAgentState) -> ValueAgentState:
        analysis_for_prompt = {
            "ticker": state["ticker"],
            "company_name": state.get("company_name"),
            "cik": state.get("cik"),
            "latest_price": state.get("price"),
            "analysis": state.get("analysis"),
            "warnings": state.get("warnings", []),
        }
        context_pack = (
            "## Prior thread memories\n"
            + ("\n".join(state.get("memories", [])) or "None")
            + "\n\n## Deterministic analysis JSON\n"
            + json.dumps(analysis_for_prompt, indent=2, default=str)
            + "\n\n## Compact annual fundamentals\n"
            + compact_fundamentals_table(state.get("annuals", []))
        )
        return {"context_pack": context_pack}

    async def synthesize(state: ValueAgentState) -> ValueAgentState:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["context_pack"]),
            ]
        )
        return {"report": str(response.content)}

    async def remember(state: ValueAgentState, config: RunnableConfig) -> ValueAgentState:
        user_msg = {
            "role": "user",
            "content": f"Screen ticker {state['ticker']} for value-investing eligibility.",
        }
        assistant_msg = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "ticker": state["ticker"],
                    "company_name": state.get("company_name"),
                    "decision": state.get("analysis", {}).get("decision"),
                    "allocation": state.get("analysis", {}).get("allocation"),
                    "report_summary": state.get("report", "")[:3000],
                },
                default=str,
            ),
        }
        await memory_manager.ainvoke({"messages": [user_msg, assistant_msg]}, config=config)
        return {}

    builder = StateGraph(ValueAgentState)
    builder.add_node("load_memory", load_memory)
    builder.add_node("fetch_sec", fetch_sec)
    builder.add_node("fetch_price", fetch_price)
    builder.add_node("analyze", analyze)
    builder.add_node("build_context", build_context)
    builder.add_node("synthesize", synthesize)
    builder.add_node("remember", remember)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "fetch_sec")
    builder.add_edge("fetch_sec", "fetch_price")
    builder.add_edge("fetch_price", "analyze")
    builder.add_edge("analyze", "build_context")
    builder.add_edge("build_context", "synthesize")
    builder.add_edge("synthesize", "remember")
    builder.add_edge("remember", END)

    return builder.compile(checkpointer=checkpointer, store=store)


@asynccontextmanager
async def make_app(settings: Settings, *, system_prompt: str) -> AsyncIterator[Any]:
    """Create a graph with Postgres-backed checkpoints and long-term memory store."""
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        await checkpointer.setup()
        embedding = init_embeddings(settings.embedding_model)
        async with AsyncPostgresStore.from_conn_string(
            settings.database_url,
            index={"dims": 384, "embed": embedding},
        ) as store:
            await store.setup()
            yield build_graph(settings, checkpointer=checkpointer, store=store, system_prompt=system_prompt)
