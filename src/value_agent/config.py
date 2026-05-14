from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    model_name: str
    embedding_model: str
    database_url: str
    sec_user_agent: str
    default_portfolio_value: float
    max_position_pct: float
    first_tranche_pct: float
    prompt_name: str
    auto_prompt_optimization: bool
    prompt_optimizer_kind: str

    @classmethod
    def from_env(cls, *, portfolio_value: Optional[float] = None) -> "Settings":
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/value_agent?sslmode=disable",
        )
        sec_user_agent = os.getenv("SEC_USER_AGENT", "")
        if not sec_user_agent or "your.email@example.com" in sec_user_agent:
            raise RuntimeError(
                "Set SEC_USER_AGENT to an app name and contact email, e.g. "
                "ValueInvestingAgent/0.1 analyst@example.com"
            )

        default_portfolio = portfolio_value
        if default_portfolio is None:
            default_portfolio = float(os.getenv("DEFAULT_PORTFOLIO_VALUE", "10000"))

        return cls(
            model_name=os.getenv("MODEL_NAME", "openai:gpt-4.1-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "openai:text-embedding-3-small"),
            database_url=database_url,
            sec_user_agent=sec_user_agent,
            default_portfolio_value=float(default_portfolio),
            max_position_pct=float(os.getenv("MAX_POSITION_PCT", "0.05")),
            first_tranche_pct=float(os.getenv("FIRST_TRANCHE_PCT", "0.25")),
            prompt_name=os.getenv("PROMPT_NAME", "value_investing_screening_prompt"),
            auto_prompt_optimization=os.getenv("AUTO_PROMPT_OPTIMIZATION", "true").lower() == "true",
            prompt_optimizer_kind=os.getenv("PROMPT_OPTIMIZER_KIND", "prompt_memory"),
        )
