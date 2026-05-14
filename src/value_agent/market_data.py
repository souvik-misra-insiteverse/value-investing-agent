from __future__ import annotations

import asyncio
from typing import Optional


def _last_price_sync(ticker: str) -> Optional[float]:
    import yfinance as yf

    stock = yf.Ticker(ticker)
    # fast_info is cheap when Yahoo returns it, but not all tickers support it.
    try:
        price = stock.fast_info.get("last_price")
        if price is not None and float(price) > 0:
            return float(price)
    except Exception:
        pass

    hist = stock.history(period="5d", auto_adjust=False)
    if hist.empty or "Close" not in hist:
        return None
    close = hist["Close"].dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


async def get_last_price(ticker: str) -> Optional[float]:
    """Fetch latest available market close/last price using yfinance."""
    return await asyncio.to_thread(_last_price_sync, ticker.upper().strip())
