from __future__ import annotations

from typing import Any

from .sec import annual_series, is_eps_unit, is_shares_unit, is_usd_unit

TAG_CANDIDATES = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "eps_diluted": ["EarningsPerShareDiluted", "IncomeLossFromContinuingOperationsPerDilutedShare"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "shares_outstanding": ["EntityCommonStockSharesOutstanding"],
}

DURATION_KEYS = {"revenue", "net_income", "operating_cash_flow", "eps_diluted", "shares_diluted"}
DEI_KEYS = {"shares_outstanding"}
EPS_KEYS = {"eps_diluted"}
SHARE_KEYS = {"shares_diluted", "shares_outstanding"}


def collect_annual_fundamentals(companyfacts: dict[str, Any], *, years: int = 10) -> list[dict[str, Any]]:
    """Build a compact 10-year annual fundamentals table from SEC companyfacts."""
    extracted: dict[str, dict[int, dict[str, Any]]] = {}

    for key, tags in TAG_CANDIDATES.items():
        taxonomy = "dei" if key in DEI_KEYS else "us-gaap"
        unit_predicate = is_eps_unit if key in EPS_KEYS else is_shares_unit if key in SHARE_KEYS else is_usd_unit
        extracted[key] = annual_series(
            companyfacts,
            taxonomy,
            tags,
            unit_predicate,
            duration=key in DURATION_KEYS,
        )

    all_years = sorted({fy for series in extracted.values() for fy in series})[-years:]
    rows: list[dict[str, Any]] = []
    for fy in all_years:
        row: dict[str, Any] = {"fiscal_year": fy, "sources": {}}
        for key, series in extracted.items():
            fact = series.get(fy)
            row[key] = fact["val"] if fact else None
            if fact:
                row["sources"][key] = {
                    "tag": fact["tag"],
                    "form": fact["form"],
                    "filed": fact["filed"],
                    "accn": fact["accn"],
                }
        rows.append(row)
    return rows


def compact_fundamentals_table(annuals: list[dict[str, Any]]) -> str:
    headers = [
        "FY",
        "Revenue",
        "Net income",
        "Assets",
        "Liabilities",
        "Equity",
        "Current ratio inputs",
        "Diluted EPS",
    ]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in annuals:
        current_inputs = f"CA={_fmt(row.get('current_assets'))}; CL={_fmt(row.get('current_liabilities'))}"
        lines.append(
            " | ".join(
                [
                    str(row.get("fiscal_year", "")),
                    _fmt(row.get("revenue")),
                    _fmt(row.get("net_income")),
                    _fmt(row.get("assets")),
                    _fmt(row.get("liabilities")),
                    _fmt(row.get("equity")),
                    current_inputs,
                    _fmt(row.get("eps_diluted")),
                ]
            )
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_v = abs(value_f)
    if abs_v >= 1_000_000_000:
        return f"{value_f / 1_000_000_000:.2f}B"
    if abs_v >= 1_000_000:
        return f"{value_f / 1_000_000:.2f}M"
    return f"{value_f:.2f}"
