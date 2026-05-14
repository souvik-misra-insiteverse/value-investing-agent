from __future__ import annotations

import math
from typing import Any, Optional


def analyze_value_case(
    annuals: list[dict[str, Any]],
    *,
    price: Optional[float],
    portfolio_value: float,
    max_position_pct: float,
    first_tranche_pct: float,
) -> dict[str, Any]:
    """Compute deterministic value-investing metrics and a screening decision."""
    annuals = sorted(annuals, key=lambda r: r["fiscal_year"])
    metrics = _business_metrics(annuals, price)
    decision = _decision(metrics, annuals)
    allocation = _allocation(
        decision,
        price=price,
        portfolio_value=portfolio_value,
        max_position_pct=max_position_pct,
        first_tranche_pct=first_tranche_pct,
    )
    return {"metrics": metrics, "decision": decision, "allocation": allocation}


def _business_metrics(annuals: list[dict[str, Any]], price: Optional[float]) -> dict[str, Any]:
    latest = annuals[-1] if annuals else {}
    years_available = len(annuals)
    net_income_values = [r.get("net_income") for r in annuals if r.get("net_income") is not None]
    positive_net_income_years = sum(1 for v in net_income_values if float(v) > 0)

    shares = _latest_positive(annuals, "shares_outstanding") or _latest_positive(annuals, "shares_diluted")
    equity = _latest_positive(annuals, "equity")
    eps = _latest_positive(annuals, "eps_diluted")
    if eps is None and shares and latest.get("net_income") is not None and shares > 0:
        eps = float(latest["net_income"]) / shares

    book_value_per_share = equity / shares if equity and shares and shares > 0 else None
    current_ratio = _ratio(latest.get("current_assets"), latest.get("current_liabilities"))
    liabilities_to_equity = _ratio(latest.get("liabilities"), latest.get("equity"))
    long_term_debt_to_equity = _ratio(latest.get("long_term_debt"), latest.get("equity"))
    price_to_earnings = _ratio(price, eps) if eps and eps > 0 else None
    price_to_book = _ratio(price, book_value_per_share) if book_value_per_share and book_value_per_share > 0 else None
    graham_number = None
    margin_of_safety = None
    if eps and eps > 0 and book_value_per_share and book_value_per_share > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_per_share)
        if price and price > 0:
            margin_of_safety = graham_number / price - 1

    revenue_cagr = _cagr(annuals, "revenue")
    equity_cagr = _cagr(annuals, "equity")
    net_income_cagr = _cagr(annuals, "net_income")
    operating_cash_flow_positive_years = sum(
        1 for r in annuals if r.get("operating_cash_flow") is not None and float(r["operating_cash_flow"]) > 0
    )

    return {
        "years_available": years_available,
        "latest_fiscal_year": latest.get("fiscal_year"),
        "price": price,
        "positive_net_income_years": positive_net_income_years,
        "operating_cash_flow_positive_years": operating_cash_flow_positive_years,
        "revenue_cagr": revenue_cagr,
        "equity_cagr": equity_cagr,
        "net_income_cagr": net_income_cagr,
        "current_ratio": current_ratio,
        "liabilities_to_equity": liabilities_to_equity,
        "long_term_debt_to_equity": long_term_debt_to_equity,
        "eps": eps,
        "book_value_per_share": book_value_per_share,
        "price_to_earnings": price_to_earnings,
        "price_to_book": price_to_book,
        "graham_number": graham_number,
        "margin_of_safety": margin_of_safety,
    }


def _decision(metrics: dict[str, Any], annuals: list[dict[str, Any]]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    failures: list[str] = []

    years = metrics["years_available"]
    if years >= 8:
        score += 10
        reasons.append(f"Has {years} annual SEC data points.")
    else:
        failures.append(f"Only {years} annual SEC data points were available; target is at least 8.")

    positive_income = metrics["positive_net_income_years"]
    if positive_income >= min(8, years):
        score += 20
        reasons.append(f"Profitable in {positive_income}/{years} available years.")
    else:
        failures.append(f"Profitable in only {positive_income}/{years} available years.")

    ocf_positive = metrics["operating_cash_flow_positive_years"]
    if years and ocf_positive >= max(1, min(7, years - 1)):
        score += 10
        reasons.append(f"Operating cash flow positive in {ocf_positive}/{years} available years.")
    else:
        failures.append("Operating cash-flow record is weak or incomplete.")

    revenue_cagr = metrics["revenue_cagr"]
    if revenue_cagr is not None and revenue_cagr > 0.03:
        score += 15
        reasons.append(f"Revenue CAGR is {_pct(revenue_cagr)}.")
    elif revenue_cagr is not None and revenue_cagr > 0:
        score += 8
        reasons.append(f"Revenue grew, but CAGR is modest at {_pct(revenue_cagr)}.")
    else:
        failures.append("Revenue did not show adequate long-term growth.")

    equity_cagr = metrics["equity_cagr"]
    if equity_cagr is not None and equity_cagr > 0.02:
        score += 10
        reasons.append(f"Book equity CAGR is {_pct(equity_cagr)}.")
    else:
        failures.append("Book equity growth was not strong enough.")

    current_ratio = metrics["current_ratio"]
    lte = metrics["long_term_debt_to_equity"]
    liabilities_to_equity = metrics["liabilities_to_equity"]
    if current_ratio is not None and current_ratio >= 1.5:
        score += 10
        reasons.append(f"Latest current ratio is {current_ratio:.2f}.")
    else:
        failures.append("Latest current ratio is below the 1.5 defensive threshold or unavailable.")

    debt_ok = False
    if lte is not None:
        debt_ok = lte <= 0.5
        debt_text = f"long-term debt/equity is {lte:.2f}"
    elif liabilities_to_equity is not None:
        debt_ok = liabilities_to_equity <= 1.0
        debt_text = f"liabilities/equity is {liabilities_to_equity:.2f}"
    else:
        debt_text = "debt ratios are unavailable"
    if debt_ok:
        score += 10
        reasons.append(f"Balance sheet leverage is acceptable: {debt_text}.")
    else:
        failures.append(f"Balance sheet leverage is not conservative enough: {debt_text}.")

    pe = metrics["price_to_earnings"]
    pb = metrics["price_to_book"]
    mos = metrics["margin_of_safety"]
    valuation_points = 0
    if pe is not None and pe <= 15:
        valuation_points += 8
    if pb is not None and pb <= 1.5:
        valuation_points += 8
    if mos is not None and mos >= 0.25:
        valuation_points += 9
    score += valuation_points
    if valuation_points >= 17:
        reasons.append("Valuation passes at least two conservative value checks.")
    elif valuation_points > 0:
        failures.append("Valuation passes only part of the conservative value screen.")
    else:
        failures.append("Valuation does not pass P/E, P/B, or Graham-number margin-of-safety checks.")

    eligible = (
        score >= 70
        and years >= 8
        and positive_income >= min(8, years)
        and (mos is not None and mos >= 0.25)
        and debt_ok
    )

    return {
        "eligible": eligible,
        "score": score,
        "score_max": 100,
        "label": "Eligible" if eligible else "Not eligible / watchlist",
        "reasons": reasons,
        "failures": failures,
        "rules": {
            "profitability": "Profitable in at least 8 years or all available years if fewer than 8.",
            "growth": "Positive revenue and book-equity trend, preferably >3% and >2% CAGR.",
            "balance_sheet": "Current ratio >= 1.5 and long-term debt/equity <= 0.5, or liabilities/equity <= 1.0 when long-term debt is unavailable.",
            "valuation": "P/E <= 15, P/B <= 1.5, and/or Graham-number margin of safety >= 25%.",
        },
    }


def _allocation(
    decision: dict[str, Any],
    *,
    price: Optional[float],
    portfolio_value: float,
    max_position_pct: float,
    first_tranche_pct: float,
) -> dict[str, Any]:
    max_position_value = portfolio_value * max_position_pct
    first_tranche_value = max_position_value * first_tranche_pct
    if not decision["eligible"] or price is None or price <= 0:
        min_units = 0
    else:
        min_units = int(first_tranche_value // price)
        if min_units == 0 and portfolio_value >= price:
            min_units = 1

    if decision["eligible"]:
        hold_period = "3-5 years minimum; review after every 10-K and exit if the thesis breaks or valuation exceeds intrinsic value."
    else:
        hold_period = "Do not buy under this screen; reassess after the next annual filing or after price declines enough to restore margin of safety."

    return {
        "portfolio_value": portfolio_value,
        "max_position_pct": max_position_pct,
        "first_tranche_pct": first_tranche_pct,
        "max_position_value": max_position_value,
        "first_tranche_value": first_tranche_value,
        "minimum_units": min_units,
        "hold_period": hold_period,
        "sizing_note": (
            "Units are calculated from configured portfolio rules, not personal advice: "
            "floor(portfolio_value * max_position_pct * first_tranche_pct / latest_price)."
        ),
    }


def _latest_positive(annuals: list[dict[str, Any]], key: str) -> Optional[float]:
    for row in reversed(annuals):
        value = row.get(key)
        if value is None:
            continue
        value_f = float(value)
        if value_f > 0:
            return value_f
    return None


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        n = float(numerator)
        d = float(denominator)
        if d == 0:
            return None
        return n / d
    except (TypeError, ValueError):
        return None


def _cagr(annuals: list[dict[str, Any]], key: str) -> Optional[float]:
    vals = [(int(r["fiscal_year"]), float(r[key])) for r in annuals if r.get(key) is not None and float(r[key]) > 0]
    if len(vals) < 2:
        return None
    first_year, first_val = vals[0]
    last_year, last_val = vals[-1]
    periods = last_year - first_year
    if periods <= 0 or first_val <= 0:
        return None
    return (last_val / first_val) ** (1 / periods) - 1


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"
