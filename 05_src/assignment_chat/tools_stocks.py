"""
Service 3: Stock Data via yfinance (Function Calling)

Provides real-time stock quotes and historical price data for publicly traded
companies using the yfinance library. This service uses LangGraph function
calling to let the LLM decide when to fetch stock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf
from langchain.tools import tool

from assignment_chat.utils.logger import get_logger

_logs = get_logger(__name__)

# Valid periods for yfinance history
_VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


@tool
def get_stock_quote(ticker: str) -> str:
    """
    Retrieves the current stock price and key financial metrics for a publicly
    traded company.

    Use this tool when the user asks about:
    - Current stock price of a company
    - Basic valuation metrics (P/E ratio, market cap, dividend yield)
    - 52-week high/low range
    - Trading volume and recent price changes
    - Quick company overview by ticker symbol

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL" for Apple, "MSFT" for Microsoft,
                "GOOGL" for Alphabet, "TSLA" for Tesla, "SPY" for S&P 500 ETF).

    Returns:
        Formatted stock quote with current price and key metrics.
    """
    ticker = ticker.upper().strip()
    _logs.debug(f"Fetching stock quote for: {ticker}")

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception as e:
        _logs.error(f"yfinance error for {ticker}: {e}")
        return f"Unable to retrieve data for ticker '{ticker}'. Error: {str(e)}"

    # Check if we got valid data
    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        # Try fast_info as fallback
        try:
            fast = stock.fast_info
            current_price = getattr(fast, "last_price", None)
            if current_price:
                return (
                    f"## {ticker} — Quick Quote\n"
                    f"Current Price: ${current_price:.2f}\n"
                    f"*(Limited data available for this ticker)*"
                )
        except Exception:
            pass
        return (
            f"No data found for ticker '{ticker}'. "
            "Please verify the ticker symbol is correct (e.g., AAPL, MSFT, GOOGL)."
        )

    # Extract key fields with safe defaults
    name = info.get("longName") or info.get("shortName") or ticker
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    open_price = info.get("open") or info.get("regularMarketOpen")
    day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
    day_low = info.get("dayLow") or info.get("regularMarketDayLow")
    week_52_high = info.get("fiftyTwoWeekHigh")
    week_52_low = info.get("fiftyTwoWeekLow")
    volume = info.get("volume") or info.get("regularMarketVolume")
    avg_volume = info.get("averageVolume")
    market_cap = info.get("marketCap")
    pe_ratio = info.get("trailingPE") or info.get("forwardPE")
    pe_type = "Trailing" if info.get("trailingPE") else "Forward" if info.get("forwardPE") else ""
    dividend_yield = info.get("dividendYield")
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    currency = info.get("currency", "USD")

    # Calculate day change
    change_str = ""
    if current_price and prev_close:
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        sign = "+" if change >= 0 else ""
        change_str = f"{sign}{change:.2f} ({sign}{change_pct:.2f}%)"

    # Format market cap
    def fmt_large(n):
        if n is None:
            return "N/A"
        if n >= 1e12:
            return f"${n/1e12:.2f}T"
        if n >= 1e9:
            return f"${n/1e9:.2f}B"
        if n >= 1e6:
            return f"${n/1e6:.2f}M"
        return f"${n:,.0f}"

    def fmt_vol(n):
        if n is None:
            return "N/A"
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        if n >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(n)

    lines = [f"## {name} ({ticker}) — Stock Quote\n"]

    if current_price:
        price_line = f"**Current Price**: {currency} {current_price:.2f}"
        if change_str:
            price_line += f"  |  **Day Change**: {change_str}"
        lines.append(price_line)

    # Price range
    range_parts = []
    if day_low and day_high:
        range_parts.append(f"**Day Range**: {currency} {day_low:.2f} – {day_high:.2f}")
    if week_52_low and week_52_high:
        range_parts.append(f"**52-Week Range**: {currency} {week_52_low:.2f} – {week_52_high:.2f}")
    if range_parts:
        lines.append("  |  ".join(range_parts))

    # Volume
    vol_parts = []
    if volume:
        vol_parts.append(f"**Volume**: {fmt_vol(volume)}")
    if avg_volume:
        vol_parts.append(f"**Avg Volume**: {fmt_vol(avg_volume)}")
    if vol_parts:
        lines.append("  |  ".join(vol_parts))

    # Fundamentals
    fund_parts = []
    if market_cap:
        fund_parts.append(f"**Market Cap**: {fmt_large(market_cap)}")
    if pe_ratio:
        fund_parts.append(f"**{pe_type} P/E**: {pe_ratio:.2f}")
    if dividend_yield:
        fund_parts.append(f"**Dividend Yield**: {dividend_yield*100:.2f}%")
    if fund_parts:
        lines.append("  |  ".join(fund_parts))

    # Sector/industry
    if sector or industry:
        lines.append(f"**Sector**: {sector}  |  **Industry**: {industry}")

    return "\n".join(lines)


@tool
def get_stock_history(ticker: str, period: str = "1mo") -> str:
    """
    Retrieves historical stock price data to analyze trends and performance.

    Use this tool when the user asks about:
    - How a stock has performed over time (past month, quarter, year, etc.)
    - Price trends and whether a stock is in an uptrend or downtrend
    - Volatility and peak/trough analysis
    - Comparing performance relative to a starting price
    - Year-to-date (YTD) performance

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT", "SPY").
        period: Time period for history. Valid values:
                "1d" (1 day), "5d" (5 days), "1mo" (1 month, default),
                "3mo" (3 months), "6mo" (6 months), "1y" (1 year),
                "2y" (2 years), "5y" (5 years), "10y" (10 years),
                "ytd" (year-to-date), "max" (maximum available).

    Returns:
        Summary of historical price performance including start/end price,
        high/low, total return, and volatility indicator.
    """
    ticker = ticker.upper().strip()
    period = period.lower().strip()
    if period not in _VALID_PERIODS:
        period = "1mo"
    _logs.debug(f"Fetching stock history for: {ticker}, period={period}")

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
    except Exception as e:
        _logs.error(f"yfinance history error for {ticker}: {e}")
        return f"Unable to retrieve history for '{ticker}'. Error: {str(e)}"

    if hist.empty:
        return (
            f"No historical data found for '{ticker}' over period '{period}'. "
            "Please check the ticker symbol."
        )

    start_price = float(hist["Close"].iloc[0])
    end_price = float(hist["Close"].iloc[-1])
    period_high = float(hist["High"].max())
    period_low = float(hist["Low"].min())
    total_return = ((end_price - start_price) / start_price) * 100
    avg_daily_vol = float(hist["Volume"].mean()) if "Volume" in hist.columns else None

    # Volatility: daily return std dev
    daily_returns = hist["Close"].pct_change().dropna()
    volatility = float(daily_returns.std() * 100) if len(daily_returns) > 1 else None

    # Trend direction
    if total_return > 5:
        trend = "strong uptrend"
    elif total_return > 1:
        trend = "modest uptrend"
    elif total_return < -5:
        trend = "strong downtrend"
    elif total_return < -1:
        trend = "modest downtrend"
    else:
        trend = "relatively flat"

    sign = "+" if total_return >= 0 else ""
    start_date = hist.index[0].strftime("%Y-%m-%d")
    end_date = hist.index[-1].strftime("%Y-%m-%d")
    num_periods = len(hist)

    lines = [
        f"## {ticker} — Historical Performance ({period})\n",
        f"**Period**: {start_date} to {end_date} ({num_periods} trading days)\n",
        f"**Start Price**: ${start_price:.2f}  |  **End Price**: ${end_price:.2f}",
        f"**Total Return**: {sign}{total_return:.2f}%  |  **Trend**: {trend.title()}",
        f"**Period High**: ${period_high:.2f}  |  **Period Low**: ${period_low:.2f}",
    ]

    if volatility is not None:
        vol_label = "High" if volatility > 2.5 else "Moderate" if volatility > 1.2 else "Low"
        lines.append(
            f"**Daily Volatility**: {volatility:.2f}% std dev ({vol_label} volatility)"
        )

    if avg_daily_vol:
        def fmt_vol(n):
            if n >= 1e6:
                return f"{n/1e6:.1f}M"
            if n >= 1e3:
                return f"{n/1e3:.0f}K"
            return f"{n:.0f}"
        lines.append(f"**Avg Daily Volume**: {fmt_vol(avg_daily_vol)}")

    # Show last 5 closing prices as a mini-table
    recent = hist["Close"].tail(5)
    if len(recent) > 1:
        lines.append("\n**Recent Closing Prices**:")
        for date, price in recent.items():
            lines.append(f"  {date.strftime('%Y-%m-%d')}: ${float(price):.2f}")

    return "\n".join(lines)
