from __future__ import annotations

import asyncio
from typing import Any, Callable, Iterable, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

SEC_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}


class SecClient:
    """Small async SEC client with polite throttling and explicit User-Agent."""

    def __init__(self, user_agent: str, *, min_interval_seconds: float = 0.12) -> None:
        self._headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        }
        self._min_interval_seconds = min_interval_seconds
        self._last_request_at = 0.0
        self._lock = asyncio.Lock()

    async def _throttle(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            delta = now - self._last_request_at
            if delta < self._min_interval_seconds:
                await asyncio.sleep(self._min_interval_seconds - delta)
            self._last_request_at = loop.time()

    @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=8), stop=stop_after_attempt(3))
    async def _get_json(self, url: str) -> dict[str, Any]:
        await self._throttle()
        async with httpx.AsyncClient(headers=self._headers, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def lookup_cik(self, ticker: str) -> tuple[str, str]:
        """Return (zero-padded CIK, company name) for a ticker."""
        ticker = ticker.upper().strip()
        exchange_payload = await self._get_json(SEC_TICKERS_EXCHANGE_URL)
        result = _find_ticker_in_exchange_payload(exchange_payload, ticker)
        if result:
            return result

        legacy_payload = await self._get_json(SEC_TICKERS_URL)
        result = _find_ticker_in_legacy_payload(legacy_payload, ticker)
        if result:
            return result

        raise ValueError(f"Ticker {ticker!r} was not found in SEC ticker mappings.")

    async def company_facts(self, cik10: str) -> dict[str, Any]:
        cik10 = str(cik10).zfill(10)
        return await self._get_json(SEC_COMPANYFACTS_URL.format(cik=cik10))


UnitPredicate = Callable[[str], bool]


def _find_ticker_in_exchange_payload(payload: dict[str, Any], ticker: str) -> Optional[tuple[str, str]]:
    fields = payload.get("fields", [])
    data = payload.get("data", [])
    if not fields or not data:
        return None
    idx = {name: i for i, name in enumerate(fields)}
    for row in data:
        if str(row[idx.get("ticker", -1)]).upper() == ticker:
            cik = str(row[idx["cik"]]).zfill(10)
            name = str(row[idx.get("name", 1)])
            return cik, name
    return None


def _find_ticker_in_legacy_payload(payload: dict[str, Any], ticker: str) -> Optional[tuple[str, str]]:
    for item in payload.values():
        if str(item.get("ticker", "")).upper() == ticker:
            return str(item["cik_str"]).zfill(10), str(item.get("title", ticker))
    return None


def is_usd_unit(unit: str) -> bool:
    return unit.upper() == "USD"


def is_shares_unit(unit: str) -> bool:
    return unit.lower() in {"shares", "share"}


def is_eps_unit(unit: str) -> bool:
    normalized = unit.lower().replace(" ", "")
    return "usd" in normalized and "share" in normalized


def annual_series(
    companyfacts: dict[str, Any],
    taxonomy: str,
    tags: Iterable[str],
    unit_predicate: UnitPredicate,
    *,
    duration: bool,
) -> dict[int, dict[str, Any]]:
    """Extract annual values by fiscal year from SEC companyfacts.

    Merges tag candidates because companies often switch XBRL tags over time. For a
    fiscal year, the latest filed annual form wins.
    """
    facts = companyfacts.get("facts", {}).get(taxonomy, {})
    output: dict[int, dict[str, Any]] = {}

    for tag_priority, tag in enumerate(tags):
        concept = facts.get(tag)
        if not concept:
            continue
        for unit, values in concept.get("units", {}).items():
            if not unit_predicate(unit):
                continue
            for fact in values:
                if fact.get("form") not in ANNUAL_FORMS:
                    continue
                fy = _safe_int(fact.get("fy"))
                if fy is None:
                    continue
                if duration:
                    if fact.get("fp") not in {"FY", None, ""}:
                        continue
                    if not _looks_like_annual_duration(fact):
                        continue
                val = _safe_float(fact.get("val"))
                if val is None:
                    continue
                candidate = {
                    "fy": fy,
                    "val": val,
                    "tag": tag,
                    "unit": unit,
                    "form": fact.get("form"),
                    "filed": str(fact.get("filed", "")),
                    "end": str(fact.get("end", "")),
                    "accn": str(fact.get("accn", "")),
                    "tag_priority": tag_priority,
                }
                existing = output.get(fy)
                if existing is None or _fact_sort_key(candidate) > _fact_sort_key(existing):
                    output[fy] = candidate
    return output


def _fact_sort_key(fact: dict[str, Any]) -> tuple[str, int]:
    # Latest filing should win. If tied, earlier tag in candidate list wins.
    return (str(fact.get("filed", "")), -int(fact.get("tag_priority", 999)))


def _looks_like_annual_duration(fact: dict[str, Any]) -> bool:
    start = str(fact.get("start", ""))
    end = str(fact.get("end", ""))
    if not start or not end:
        return True
    try:
        from datetime import date

        start_date = date.fromisoformat(start[:10])
        end_date = date.fromisoformat(end[:10])
        days = (end_date - start_date).days
        return 300 <= days <= 450
    except Exception:
        return True


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
