"""Utilities for maintaining the ChatGPT micro-cap portfolio.

This module rewrites the original script to:
- Centralize market data fetching with a robust Yahoo->Stooq fallback
- Ensure ALL price requests go through the same accessor
- Handle empty Yahoo frames (no exception) so fallback actually triggers
- Normalize Stooq output to Yahoo-like columns
- Make weekend handling consistent and testable
- Keep behavior and CSV formats compatible with prior runs

Notes:
- Some tickers/indices are not available on Stooq (e.g., ^RUT). These stay on Yahoo.
- Stooq end date is exclusive; we add +1 day for ranges.
- "Adj Close" is set equal to "Close" for Stooq to match downstream expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, cast,Dict, List, Optional
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import json
import logging

# Optional pandas-datareader import for Stooq access
try:
    import pandas_datareader.data as pdr
    _HAS_PDR = True
except Exception:
    _HAS_PDR = False

# -------- AS-OF override --------
ASOF_DATE: pd.Timestamp | None = None

def set_asof(date: str | datetime.datetime | pd.Timestamp | None) -> None:
    """Set a global 'as of' date so the script treats that day as 'today'. Use 'YYYY-MM-DD' format."""
    global ASOF_DATE
    if date is None:
        print("No prior date passed. Using today's date...")
        ASOF_DATE = None
        return
    ASOF_DATE = pd.Timestamp(date).normalize()
    pure_date = ASOF_DATE.date()

    print(f"Setting date as {pure_date}.")

# Allow env var override:  ASOF_DATE=YYYY-MM-DD python trading_script.py
_env_asof = os.environ.get("ASOF_DATE")
if _env_asof:
    set_asof(_env_asof)

def _effective_now() -> datetime.datetime:
    return (ASOF_DATE.to_pydatetime() if ASOF_DATE is not None else datetime.datetime.now())

# ------------------------------
# Globals / file locations
# ------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # Save files alongside this script by default
PORTFOLIO_CSV = DATA_DIR / "Start Your Own" / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"
DEFAULT_BENCHMARKS = ["IWO", "XBI", "SPY", "IWM"]

# Set up logger for this module
logger = logging.getLogger(__name__)

# Log initial global state configuration (only when run as main script)
def _log_initial_state():
    """Log the initial global file path configuration."""
    logger.info("=== Trading Script Initial Configuration ===")
    logger.info("Script directory: %s", SCRIPT_DIR)
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("Portfolio CSV: %s", PORTFOLIO_CSV)
    logger.info("Trade log CSV: %s", TRADE_LOG_CSV)
    logger.info("Default benchmarks: %s", DEFAULT_BENCHMARKS)
    logger.info("==============================================")

# ------------------------------
# Configuration helpers — benchmark tickers (tickers.json)
# ------------------------------



logger = logging.getLogger(__name__)

def _read_json_file(path: Path) -> Optional[Dict]:
    """Read and parse JSON from `path`. Return dict on success, None if not found or invalid.

    - FileNotFoundError -> return None
    - JSON decode error -> log a warning and return None
    - Other IO errors -> log a warning and return None
    """
    try:
        logger.info("Reading JSON file: %s", path)
        with path.open("r", encoding="utf-8") as fh:
            result = json.load(fh)
            logger.info("Successfully read JSON file: %s", path)
            return result
    except FileNotFoundError:
        logger.info("JSON file not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("tickers.json present but malformed: %s -> %s. Falling back to defaults.", path, exc)
        return None
    except Exception as exc:
        logger.warning("Unable to read tickers.json (%s): %s. Falling back to defaults.", path, exc)
        return None

def load_benchmarks(script_dir: Path | None = None) -> List[str]:
    """Return a list of benchmark tickers.

    Looks for a `tickers.json` file in either:
      - script_dir (if provided) OR the module SCRIPT_DIR, and then
      - script_dir.parent (project root candidate).

    Expected schema:
      {"benchmarks": ["IWO", "XBI", "SPY", "IWM"]}

    Behavior:
    - If file missing or malformed -> return DEFAULT_BENCHMARKS copy.
    - If 'benchmarks' key missing or not a list -> log warning and return defaults.
    - Normalizes tickers (strip, upper) and preserves order while removing duplicates.
    """
    base = Path(script_dir) if script_dir else SCRIPT_DIR
    candidates = [base, base.parent]

    cfg = None
    cfg_path = None
    for c in candidates:
        p = (c / "tickers.json").resolve()
        data = _read_json_file(p)
        if data is not None:
            cfg = data
            cfg_path = p
            break

    if not cfg:
        return DEFAULT_BENCHMARKS.copy()

    benchmarks = cfg.get("benchmarks")
    if not isinstance(benchmarks, list):
        logger.warning("tickers.json at %s missing 'benchmarks' array. Falling back to defaults.", cfg_path)
        return DEFAULT_BENCHMARKS.copy()

    seen = set()
    result: list[str] = []
    for t in benchmarks:
        if not isinstance(t, str):
            continue
        up = t.strip().upper()
        if not up:
            continue
        if up not in seen:
            seen.add(up)
            result.append(up)

    return result if result else DEFAULT_BENCHMARKS.copy()


# ------------------------------
# Date helpers
# ------------------------------

def last_trading_date(today: datetime.datetime | None = None) -> pd.Timestamp:
    """Return last trading date (Mon–Fri), mapping Sat/Sun -> Fri and pre-market hours to previous day.
    
    If running before 9:30 AM ET on a weekday, uses the previous trading day since
    market data for the current day may not be available yet.
    """
    dt = pd.Timestamp(today or _effective_now())
    
    # Handle weekends first
    if dt.weekday() == 5:  # Sat -> Fri
        friday_date = (dt - pd.Timedelta(days=1)).normalize()
        logger.info("Script running on Saturday - using Friday's data (%s) instead of today's date", friday_date.date())
        return friday_date
    if dt.weekday() == 6:  # Sun -> Fri
        friday_date = (dt - pd.Timedelta(days=2)).normalize()
        logger.info("Script running on Sunday - using Friday's data (%s) instead of today's date", friday_date.date())
        return friday_date
    
    # For weekdays, check if we're before market hours or after market close
    # If it's very early or very late, market data for "today" might not be available
    try:
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        dt_et = dt.tz_convert(et_tz) if dt.tz is not None else dt.tz_localize('UTC').tz_convert(et_tz)
        
        # Market hours are roughly 9:30 AM to 4:00 PM ET
        # If it's before 9:30 AM ET or after hours, data may not be available for current day
        market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Use previous trading day if:
        # 1. Before market open, OR
        # 2. After market close (data might be delayed)
        if dt_et.time() < market_open.time():
            reason = f"before market hours ({dt_et.strftime('%H:%M')} ET)"
            use_previous = True
        elif dt_et.time() > market_close.time() and dt_et.hour >= 20:  # After 8 PM ET
            reason = f"late evening ({dt_et.strftime('%H:%M')} ET)"
            use_previous = True
        else:
            use_previous = False
            
        if use_previous:
            prev_date = dt - pd.Timedelta(days=1)
            # If previous day was a weekend, go back further
            while prev_date.weekday() >= 5:  # Sat=5, Sun=6
                prev_date = prev_date - pd.Timedelta(days=1)
            logger.info("Script running %s - using previous trading day (%s)", 
                       reason, prev_date.date())
            return prev_date.normalize()
    except Exception:
        # If timezone handling fails, use a simple hour check
        # Assume if it's before 9 AM or after 9 PM local time, we might need previous day data
        if dt.hour < 9 or dt.hour >= 21:
            prev_date = dt - pd.Timedelta(days=1)
            # If previous day was a weekend, go back further
            while prev_date.weekday() >= 5:  # Sat=5, Sun=6
                prev_date = prev_date - pd.Timedelta(days=1)
            reason = "before 9 AM or after 9 PM local" if dt.hour < 9 else "after 9 PM local"
            logger.info("Script running %s - using previous trading day (%s)", 
                       reason, prev_date.date())
            return prev_date.normalize()
    
    return dt.normalize()

def check_weekend() -> str:
    """Backwards-compatible wrapper returning ISO date string for last trading day."""
    return last_trading_date().date().isoformat()

def trading_day_window(target: datetime.datetime | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) window for the last trading day (Fri on weekends)."""
    d = last_trading_date(target)
    return d, (d + pd.Timedelta(days=1))


# ------------------------------
# Data access layer
# ------------------------------

# Known Stooq symbol remaps for common indices
STOOQ_MAP = {
    "^GSPC": "^SPX",  # S&P 500
    "^DJI": "^DJI",   # Dow Jones
    "^IXIC": "^IXIC", # Nasdaq Composite
    # "^RUT": not on Stooq; keep Yahoo
}

# Symbols we should *not* attempt on Stooq
STOOQ_BLOCKLIST = {"^RUT"}


# ------------------------------
# Data access layer (UPDATED)
# ------------------------------

@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str  # "yahoo" | "stooq-pdr" | "stooq-csv" | "yahoo:<proxy>-proxy" | "empty"

def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten multiIndex frame so we can lazily lookup values by index.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # If the second level is the same ticker for all cols, drop it
            if len(set(df.columns.get_level_values(1))) == 1:
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
            else:
                # multiple tickers: flatten with join
                df = df.copy()
                df.columns = ["_".join(map(str, t)).strip("_") for t in df.columns.to_flat_index()]
        except Exception:
            df = df.copy()
            df.columns = ["_".join(map(str, t)).strip("_") for t in df.columns.to_flat_index()]

    # Ensure all expected columns exist
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return df[cols]

def _yahoo_download(ticker: str, **kwargs: Any) -> pd.DataFrame:
    """Call yfinance.download with a real UA and silence all chatter."""
    import io, logging
    from contextlib import redirect_stderr, redirect_stdout

    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                df = cast(pd.DataFrame, yf.download(ticker, **kwargs))
        except Exception:
            return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _stooq_csv_download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch OHLCV from Stooq CSV endpoint (daily). Good for US tickers and many ETFs."""
    import requests, io
    if ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()
    t = STOOQ_MAP.get(ticker, ticker)

    # Stooq daily CSV: lowercase; equities/ETFs use .us, indices keep ^ prefix
    if not t.startswith("^"):
        sym = t.lower()
        if not sym.endswith(".us"):
            sym = f"{sym}.us"
    else:
        sym = t.lower()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Filter to [start, end) (Stooq end is exclusive)
        df = df.loc[(df.index >= start.normalize()) & (df.index < end.normalize())]

        # Normalize to Yahoo-like schema
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    except Exception:
        return pd.DataFrame()

def _stooq_download(
    ticker: str,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Fetch OHLCV from Stooq via pandas-datareader; returns empty DF on failure."""
    if not _HAS_PDR or ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()

    t = STOOQ_MAP.get(ticker, ticker)
    if not t.startswith("^"):
        t = t.lower()

    try:
        # Ensure pdr is imported locally if not available globally
        if not _HAS_PDR:
            return pd.DataFrame()
        import pandas_datareader.data as pdr_local
        df = cast(pd.DataFrame, pdr_local.DataReader(t, "stooq", start=start, end=end))
        df.sort_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def _weekend_safe_range(period: str | None, start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute a concrete [start, end) window.
    - If explicit start/end provided: use them (add +1 day to end to make it exclusive).
    - If period is '1d': use the last trading day's [Fri, Sat) window on weekends.
    - If period like '2d'/'5d': build a window ending at the last trading day.
    """
    if start or end:
        end_ts = pd.Timestamp(end) if end else last_trading_date() + pd.Timedelta(days=1)
        start_ts = pd.Timestamp(start) if start else (end_ts - pd.Timedelta(days=5))
        return start_ts.normalize(), pd.Timestamp(end_ts).normalize()

    # No explicit dates; derive from period
    if isinstance(period, str) and period.endswith("d"):
        days = int(period[:-1])
    else:
        days = 1

    # Anchor to last trading day (Fri on Sun/Sat)
    end_trading = last_trading_date()
    start_ts = (end_trading - pd.Timedelta(days=days)).normalize()
    end_ts = (end_trading + pd.Timedelta(days=1)).normalize()
    return start_ts, end_ts

def download_price_data(ticker: str, **kwargs: Any) -> FetchResult:
    """
    Robust OHLCV fetch with multi-stage fallbacks:

    Order:
      1) Yahoo Finance via yfinance
      2) Stooq via pandas-datareader
      3) Stooq direct CSV
      4) Index proxies (e.g., ^GSPC->SPY, ^RUT->IWM) via Yahoo
    Returns a DataFrame with columns [Open, High, Low, Close, Adj Close, Volume].
    """
    # Pull out range args, compute a weekend-safe window
    period = kwargs.pop("period", None)
    start = kwargs.pop("start", None)
    end = kwargs.pop("end", None)
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)

    s, e = _weekend_safe_range(period, start, end)

    # ---------- 1) Yahoo (date-bounded) ----------
    df_y = _yahoo_download(ticker, start=s, end=e, **kwargs)
    if isinstance(df_y, pd.DataFrame) and not df_y.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_y)), "yahoo")

    # ---------- 2) Stooq via pandas-datareader ----------
    df_s = _stooq_download(ticker, start=s, end=e)
    if isinstance(df_s, pd.DataFrame) and not df_s.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_s)), "stooq-pdr")

    # ---------- 3) Stooq direct CSV ----------
    df_csv = _stooq_csv_download(ticker, s, e)
    if isinstance(df_csv, pd.DataFrame) and not df_csv.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_csv)), "stooq-csv")

    # ---------- 4) Proxy indices if applicable ----------
    proxy_map = {"^GSPC": "SPY", "^RUT": "IWM"}
    proxy = proxy_map.get(ticker)
    if proxy:
        df_proxy = _yahoo_download(proxy, start=s, end=e, **kwargs)
        if isinstance(df_proxy, pd.DataFrame) and not df_proxy.empty:
            return FetchResult(_normalize_ohlcv(_to_datetime_index(df_proxy)), f"yahoo:{proxy}-proxy")

    # ---------- Nothing worked ----------
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    return FetchResult(empty, "empty")



# ------------------------------
# File path configuration
# ------------------------------

def set_data_dir(data_dir: Path) -> None:
    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV
    logger.info("Setting data directory: %s", data_dir)
    DATA_DIR = Path(data_dir)
    logger.debug("Creating data directory if it doesn't exist: %s", DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    PORTFOLIO_CSV = DATA_DIR / "Start Your Own" / "chatgpt_portfolio_update.csv"
    TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"
    logger.info("Data directory configured - Portfolio CSV: %s, Trade Log CSV: %s", PORTFOLIO_CSV, TRADE_LOG_CSV)


# ------------------------------
# Excel helper functions
# ------------------------------


def save_portfolio_to_csv_with_formatting(df: pd.DataFrame) -> None:
    """Save portfolio data to CSV file with blank lines after TOTAL rows for better readability."""
    try:
        csv_path = DATA_DIR / "Start Your Own" / "chatgpt_portfolio_update.csv"
        logger.info("Writing formatted CSV file: %s", csv_path)

        # Create formatted data with blank lines after TOTAL rows
        formatted_rows = []

        for idx, row in df.iterrows():
            # Add the current row
            formatted_rows.append(row)

            # If this is a TOTAL row, add a blank line after it
            if row['Ticker'] == 'TOTAL':
                # Create a blank row (all NaN/empty values)
                blank_row = pd.Series([None] * len(df.columns), index=df.columns)
                formatted_rows.append(blank_row)

        # Create DataFrame with blank lines
        df_formatted = pd.DataFrame(formatted_rows).reset_index(drop=True)

        # Save to CSV
        df_formatted.to_csv(csv_path, index=False)
        logger.info("Successfully wrote formatted CSV file: %s", csv_path)

    except Exception as e:
        logger.error("Error writing CSV file %s: %s", csv_path, e)
        # Don't raise - this is supplementary to Excel saving
        pass


# ------------------------------
# Input helpers
# ------------------------------

def get_stop_loss_percentage() -> float:
    """Get stop loss percentage from user input with validation.
    
    Returns:
        float: Stop loss percentage (0 means no stop loss)
    """
    try:
        stop_loss_pct = float(input("Enter stop loss percentage (e.g., 8 for 8% below buy price, or 0 to skip): "))
        if stop_loss_pct < 0:
            raise ValueError("Stop loss percentage cannot be negative")
        return stop_loss_pct
    except ValueError:
        raise ValueError("Invalid stop loss percentage")

def calculate_stop_loss_price(buy_price: float, stop_loss_pct: float) -> float:
    """Calculate actual stop loss price from percentage.
    
    Args:
        buy_price: The buy/execution price
        stop_loss_pct: Stop loss percentage (e.g., 8 for 8%)
    
    Returns:
        float: Stop loss price (0 if stop_loss_pct is 0)
    """
    if stop_loss_pct <= 0:
        return 0.0
    return round(buy_price * (1 - stop_loss_pct / 100), 2)

def process_buy_order(
    order_type: str,
    ticker: str, 
    shares: float,
    order_price: float | None,
    stop_loss_pct: float,
    cash: float,
    portfolio_df: pd.DataFrame
) -> tuple[float, pd.DataFrame] | None:
    """Process a buy order of any type (market, limit, executed).
    
    Args:
        order_type: 'm' (market), 'l' (limit), or 'e' (executed)
        ticker: Stock ticker symbol
        shares: Number of shares to buy
        order_price: Limit/execution price (None for market orders)
        stop_loss_pct: Stop loss percentage
        cash: Available cash
        portfolio_df: Current portfolio
        
    Returns:
        Tuple of (updated_cash, updated_portfolio) or None if failed
    """
    today_iso = last_trading_date().date().isoformat()
    
    # Determine execution price based on order type
    if order_type == "m":
        # Market order - use opening price
        s, e = trading_day_window()
        fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
        data = fetch.df
        if data.empty:
            print(f"MOO buy for {ticker} failed: no market data available (source={fetch.source}).")
            return None
        
        o = float(data["Open"].iloc[-1]) if "Open" in data else float(data["Close"].iloc[-1])
        exec_price = round(o, 2)
        source_info = fetch.source
        reason = "MANUAL BUY MOO - Filled"
        
    elif order_type == "l":
        # Limit order - simulate execution
        s, e = trading_day_window()
        fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
        data = fetch.df
        if data.empty:
            print(f"Manual buy for {ticker} failed: no market data available (source={fetch.source}).")
            return None
            
        o = float(data["Open"].iloc[-1]) if "Open" in data else float(data["Close"].iloc[-1])
        h = float(data["High"].iloc[-1])
        l = float(data["Low"].iloc[-1])
        if pd.isna(o):
            o = float(data["Close"].iloc[-1])
            
        # Simulate limit order execution
        if o <= order_price:
            exec_price = o
        elif l <= order_price:
            exec_price = order_price
        else:
            print(f"Buy limit ${order_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled.")
            return None
            
        source_info = fetch.source
        reason = "MANUAL BUY LIMIT - Filled"
        
    elif order_type == "e":
        # Executed order - use exact price provided
        exec_price = order_price
        source_info = "executed"
        reason = "MANUAL BUY EXECUTED - Already Filled"
        
    else:
        print(f"Unknown order type: {order_type}")
        return None
    
    # Calculate stop loss and cost
    stop_loss = calculate_stop_loss_price(exec_price, stop_loss_pct)
    cost_amt = exec_price * shares
    
    # Check if we have enough cash
    if cost_amt > cash:
        print(f"Buy for {ticker} failed: cost {cost_amt:.2f} exceeds cash {cash:.2f}.")
        return None
    
    # Show stop loss info
    if stop_loss_pct > 0:
        print(f"Stop loss set at ${stop_loss:.2f} ({stop_loss_pct}% below buy price of ${exec_price:.2f})")
    
    # Confirmation
    order_desc = {
        "m": "MOO",
        "l": "LIMIT", 
        "e": "EXECUTED"
    }[order_type]
    
    check = input(
        f"You are placing a BUY {order_desc} for {shares} {ticker} at ${exec_price:.2f}.\n"
        f"If this is a mistake, type '1' or, just hit Enter: "
    )
    if check == "1":
        print("Returning...")
        return cash, portfolio_df
    
    # Log the trade
    log = {
        "Date": today_iso,
        "Ticker": ticker,
        "Shares Bought": round(shares, 4),
        "Buy Price": round(exec_price, 2),
        "Cost Basis": round(cost_amt, 2),
        "PnL": 0.0,
        "Reason": reason,
    }
    
    if os.path.exists(TRADE_LOG_CSV):
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df_log = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df_log.empty:
            df_log = pd.DataFrame([log])
        else:
            df_log = pd.concat([df_log, pd.DataFrame([log])], ignore_index=True)
    else:
        df_log = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df_log.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)
    
    # Update portfolio
    if not isinstance(portfolio_df, pd.DataFrame) or portfolio_df.empty:
        portfolio_df = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )
    
    rows = portfolio_df.loc[portfolio_df["ticker"].astype(str).str.upper() == ticker.upper()]
    if rows.empty:
        # New position
        new_trade = {
            "ticker": ticker,
            "shares": round(float(shares), 4),
            "stop_loss": round(float(stop_loss), 2),
            "buy_price": round(float(exec_price), 2),
            "cost_basis": round(float(cost_amt), 2),
        }
        if portfolio_df.empty:
            portfolio_df = pd.DataFrame([new_trade])
        else:
            portfolio_df = pd.concat([portfolio_df, pd.DataFrame([new_trade])], ignore_index=True)
    else:
        # Add to existing position
        idx = rows.index[0]
        cur_shares = float(portfolio_df.at[idx, "shares"])
        cur_cost = float(portfolio_df.at[idx, "cost_basis"])
        new_shares = cur_shares + float(shares)
        new_cost = cur_cost + float(cost_amt)
        avg_price = new_cost / new_shares if new_shares else 0.0
        portfolio_df.at[idx, "shares"] = round(new_shares, 4)
        portfolio_df.at[idx, "cost_basis"] = round(new_cost, 2)
        portfolio_df.at[idx, "buy_price"] = round(avg_price, 2)
        portfolio_df.at[idx, "stop_loss"] = round(float(stop_loss), 2)
    
    # Update cash
    cash -= cost_amt
    
    # Success message
    if order_type == "m":
        print(f"Manual BUY MOO for {ticker} filled at ${exec_price:.2f} ({source_info}).")
    elif order_type == "l":
        print(f"Manual BUY LIMIT for {ticker} filled at ${exec_price:.2f} ({source_info}).")
    else:  # executed
        print(f"Executed BUY for {ticker} logged at ${exec_price:.2f}.")
    
    return cash, portfolio_df

# ------------------------------
# Portfolio operations
# ------------------------------

def _ensure_df(portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]]) -> pd.DataFrame:
    if isinstance(portfolio, pd.DataFrame):
        return portfolio.copy()
    if isinstance(portfolio, (dict, list)):
        df = pd.DataFrame(portfolio)
        # Ensure proper columns exist even for empty DataFrames
        if df.empty:
            logger.debug("Creating empty portfolio DataFrame with proper column structure")
            df = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        return df
    raise TypeError("portfolio must be a DataFrame, dict, or list[dict]")

def process_portfolio(
    portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]],
    cash: float,
    interactive: bool = True,
) -> tuple[pd.DataFrame, float]:
    today_iso = last_trading_date().date().isoformat()
    portfolio_df = _ensure_df(portfolio)

    results: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0

    # ------- Interactive trade entry (supports MOO) -------
    if interactive:
        while True:
            print(portfolio_df)
            action = input(
                f""" You have {cash} in cash.
Would you like to log a manual trade? Enter 'b' for buy, 's' for sell, or press Enter to continue: """
            ).strip().lower()

            if action == "b":
                ticker = input("Enter ticker symbol: ").strip().upper()
                order_type = input("Order type? 'm' = market-on-open, 'l' = limit, 'e' = executed (already filled): ").strip().lower()

                try:
                    shares = float(input("Enter number of shares: "))
                    if shares <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid share amount. Buy cancelled.")
                    continue

                # Get order-type-specific inputs
                try:
                    if order_type == "m":
                        stop_loss_pct = get_stop_loss_percentage()
                        order_price = None  # Will be determined from market data
                        
                    elif order_type == "l":
                        order_price = float(input("Enter buy LIMIT price: "))
                        if order_price <= 0:
                            raise ValueError("Buy price must be positive")
                        stop_loss_pct = get_stop_loss_percentage()
                        
                    elif order_type == "e":
                        order_price = float(input("Enter the actual execution price: "))
                        if order_price <= 0:
                            raise ValueError("Execution price must be positive")
                        stop_loss_pct = get_stop_loss_percentage()
                        
                    else:
                        print("Unknown order type. Use 'm', 'l', or 'e'.")
                        continue
                        
                except ValueError as e:
                    print(f"Invalid input. {order_type.upper()} buy cancelled.")
                    continue

                # Process the buy order using unified function
                result = process_buy_order(
                    order_type=order_type,
                    ticker=ticker,
                    shares=shares,
                    order_price=order_price,
                    stop_loss_pct=stop_loss_pct,
                    cash=cash,
                    portfolio_df=portfolio_df
                )
                
                if result is not None:
                    cash, portfolio_df = result
                continue

            if action == "s":
                # Show current holdings for selection
                if portfolio_df.empty:
                    print("No positions to sell.")
                    continue

                print("\n📊 Current Holdings:")
                print("=" * 50)
                holdings = []
                for idx, stock in portfolio_df.iterrows():
                    ticker_name = str(stock["ticker"]).upper()
                    shares_held = float(stock["shares"]) if not pd.isna(stock["shares"]) else 0.0
                    buy_price = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0

                    if shares_held > 0:  # Only show positions with shares
                        holdings.append({
                            'ticker': ticker_name,
                            'shares': shares_held,
                            'buy_price': buy_price,
                            'index': len(holdings)
                        })
                        print(f"{len(holdings)}. {ticker_name:<6} | {shares_held:>8.4f} shares @ ${buy_price:.2f}")

                if not holdings:
                    print("No positions available to sell.")
                    continue

                print("=" * 50)

                try:
                    # Get user selection
                    selection = input(f"\nSelect position to sell (1-{len(holdings)}) or 'c' to cancel: ").strip().lower()

                    if selection == 'c':
                        print("Sell cancelled.")
                        continue

                    selection_num = int(selection)
                    if selection_num < 1 or selection_num > len(holdings):
                        print("Invalid selection. Sell cancelled.")
                        continue

                    # Get selected ticker info
                    selected = holdings[selection_num - 1]
                    ticker = selected['ticker']
                    max_shares = selected['shares']

                    print(f"\n📉 Selling {ticker} (Max: {max_shares:.4f} shares)")

                    # Get sell details with default to max shares
                    shares_input = input(f"Enter number of shares to sell (max {max_shares:.4f}, press Enter for all): ").strip()
                    
                    if shares_input == "":
                        # Default to selling all shares
                        shares = max_shares
                        print(f"Defaulting to sell all {shares:.4f} shares")
                    else:
                        try:
                            shares = float(shares_input)
                        except ValueError:
                            print("Invalid number format. Sell cancelled.")
                            continue
                    
                    if shares <= 0 or shares > max_shares:
                        print(f"Invalid share amount. Must be between 0 and {max_shares:.4f}. Sell cancelled.")
                        continue

                    sell_price = float(input("Enter sell LIMIT price: "))
                    if sell_price <= 0:
                        print("Invalid sell price. Sell cancelled.")
                        continue

                except (ValueError, KeyboardInterrupt):
                    print("Invalid input. Manual sell cancelled.")
                    continue

                cash, portfolio_df = log_manual_sell(
                    sell_price, shares, ticker, cash, portfolio_df
                )
                continue

            break  # proceed to pricing

    # ------- Check trade log to identify buys/sells today -------
    today_buys = set()  # Track tickers bought today
    today_sells = {}    # Track tickers sold today {ticker: total_proceeds}
    today_buy_costs = {}  # Track buy costs today {ticker: total_cost}

    if TRADE_LOG_CSV.exists():
        try:
            trade_log = pd.read_csv(TRADE_LOG_CSV)

            # Find buys today
            today_buy_entries = trade_log[
                (trade_log["Date"] == today_iso) &
                (trade_log["Reason"].astype(str).str.contains("BUY", case=False, na=False))
            ]
            today_buys = set(today_buy_entries["Ticker"].astype(str).str.upper())

            # Calculate buy costs for cash flow
            for _, buy_entry in today_buy_entries.iterrows():
                ticker = str(buy_entry["Ticker"]).upper()
                cost = float(buy_entry["Cost Basis"]) if pd.notna(buy_entry["Cost Basis"]) else 0
                today_buy_costs[ticker] = today_buy_costs.get(ticker, 0) + cost

            # Find sells today and calculate proceeds
            today_sell_entries = trade_log[
                (trade_log["Date"] == today_iso) &
                (pd.notna(trade_log["Shares Sold"])) &
                (trade_log["Shares Sold"] > 0)
            ]

            for _, sell_entry in today_sell_entries.iterrows():
                ticker = str(sell_entry["Ticker"]).upper()
                shares_sold = float(sell_entry["Shares Sold"])
                sell_price = float(sell_entry["Sell Price"]) if pd.notna(sell_entry["Sell Price"]) else 0
                proceeds = shares_sold * sell_price
                today_sells[ticker] = today_sells.get(ticker, 0) + proceeds

        except Exception as e:
            logger.warning("Could not read trade log for trade detection: %s", e)

    # ------- Daily pricing + stop-loss execution -------
    s, e = trading_day_window()
    for _, stock in portfolio_df.iterrows():
        ticker = str(stock["ticker"]).upper()
        shares = float(stock["shares"]) if not pd.isna(stock["shares"]) else 0.0
        cost = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0
        cost_basis = float(stock["cost_basis"]) if not pd.isna(stock["cost_basis"]) else cost * shares
        stop = float(stock["stop_loss"]) if not pd.isna(stock["stop_loss"]) else 0.0

        fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
        data = fetch.df

        if data.empty:
            print(f"No data for {ticker} (source={fetch.source}).")
            row = {
                "Date": today_iso, "Ticker": ticker, "Shares": round(shares, 4),
                "Buy Price": round(cost, 2), "Cost Basis": round(cost_basis, 2), "Stop Loss": round(stop, 2),
                "Current Price": "", "Total Value": "", "PnL": "", "PnL %": "",
                "Action": "NO DATA", "Cash Balance": "", "Total Equity": "", "Notes": "",
            }
            results.append(row)
            continue

        o = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
        h = float(data["High"].iloc[-1])
        l = float(data["Low"].iloc[-1])
        c = float(data["Close"].iloc[-1])
        if np.isnan(o):
            o = c

        # Check if stop loss level was breached (for alerting purposes only)
        price = round(c, 2)
        value = round(price * shares, 2)
        pnl = round((price - cost) * shares, 2)

        # Alert if stop loss was triggered but don't auto-execute
        if stop and l <= stop:
            print(f"⚠️  ALERT: {ticker} stop loss triggered! Low: ${l:.2f} vs Stop Loss: ${stop:.2f}")
            print(f"   Current price: ${price:.2f} - Consider manual sale if desired.")
            action_note = "STOP LOSS ALERT"
        else:
            action_note = ""

        # Determine action and cash flow
        ticker_upper = ticker.upper()

        # Determine action (never auto-sell due to stop loss)
        if ticker_upper in today_buys:
            action = "BUY"
            # Show negative cash flow for purchases
            cash_flow = f"-{today_buy_costs.get(ticker_upper, 0):.2f}" if today_buy_costs.get(ticker_upper, 0) > 0 else ""
            # Count in portfolio total (position held)
            total_value += value
            total_pnl += pnl
            position_value = value
        elif ticker_upper in today_sells:
            action = "SELL - Manual"  # Will be overridden if stop loss triggered
            # Show positive cash flow for sales
            cash_flow = f"+{today_sells.get(ticker_upper, 0):.2f}"
            # Count remaining position value (for partial sales)
            total_value += value
            total_pnl += pnl
            position_value = value  # Show remaining position value, not 0
        else:
            action = "HOLD"
            cash_flow = ""
            # Count in portfolio total (position held)
            total_value += value
            total_pnl += pnl
            position_value = value

        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0

        row = {
            "Date": today_iso, "Ticker": ticker, "Shares": round(shares, 4),
            "Buy Price": round(cost, 2), "Cost Basis": round(cost_basis, 2), "Stop Loss": round(stop, 2),
            "Current Price": price, "Total Value": round(position_value, 2), "PnL": pnl, "PnL %": round(pnl_pct, 2),
            "Action": action, "Cash Balance": cash_flow, "Total Equity": "", "Notes": action_note,
        }

        results.append(row)

    total_row = {
        "Date": today_iso, "Ticker": "TOTAL", "Shares": "", "Buy Price": "",
        "Cost Basis": "", "Stop Loss": "", "Current Price": "",
        "Total Value": round(total_value, 2), "PnL": round(total_pnl, 2), "PnL %": "",
        "Action": "", "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2), "Notes": "",
    }
    results.append(total_row)

    df_out = pd.DataFrame(results)

    # Read existing CSV data and remove today's entries to avoid duplicates
    if PORTFOLIO_CSV.exists():
        existing = _read_csv_with_encoding_fallback(PORTFOLIO_CSV)
        # Remove blank lines for processing
        existing = existing.dropna(how='all')
        existing = existing[existing["Date"] != str(today_iso)]
        print("Saving results to CSV...")
        df_out = pd.concat([existing, df_out], ignore_index=True)
    else:
        print("Creating new CSV file...")

    # Save to CSV with blank lines after TOTAL rows for better readability
    save_portfolio_to_csv_with_formatting(df_out)

    return portfolio_df, cash



# ------------------------------
# Trade logging
# ------------------------------

def log_sell(
    ticker: str,
    shares: float,
    price: float,
    cost: float,
    pnl: float,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    today = check_weekend()
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Sold": shares,
        "Sell Price": price,
        "Cost Basis": cost,
        "PnL": pnl,
        "Reason": "AUTOMATED SELL - STOPLOSS TRIGGERED",
    }
    print(f"{ticker} stop loss was met. Selling all shares.")
    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)
    return portfolio

def log_manual_buy(
    buy_price: float,
    shares: float,
    ticker: str,
    stoploss: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    today = check_weekend()

    if interactive:
        check = input(
            f"You are placing a BUY LIMIT for {shares} {ticker} at ${buy_price:.2f}.\n"
            f"If this is a mistake, type '1' or, just hit Enter: "
        )
        if check == "1":
            print("Returning...")
            return cash, chatgpt_portfolio

    if not isinstance(chatgpt_portfolio, pd.DataFrame) or chatgpt_portfolio.empty:
        chatgpt_portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

    s, e = trading_day_window()
    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
    data = fetch.df
    if data.empty:
        print(f"Manual buy for {ticker} failed: no market data available (source={fetch.source}).")
        return cash, chatgpt_portfolio

    if "Open" in data.columns and not data["Open"].empty:
        o = float(data["Open"].iloc[-1])
    else:
        o = np.nan
    h = float(data["High"].iloc[-1])
    l = float(data["Low"].iloc[-1])
    if np.isnan(o):
        o = float(data["Close"].iloc[-1])

    if o <= buy_price:
        exec_price = o
    elif l <= buy_price:
        exec_price = buy_price
    else:
        print(f"Buy limit ${buy_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled.")
        return cash, chatgpt_portfolio

    cost_amt = exec_price * shares
    if cost_amt > cash:
        print(f"Manual buy for {ticker} failed: cost {cost_amt:.2f} exceeds cash balance {cash:.2f}.")
        return cash, chatgpt_portfolio

    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price": exec_price,
        "Cost Basis": cost_amt,
        "PnL": 0.0,
        "Reason": "MANUAL BUY LIMIT - Filled",
    }
    if os.path.exists(TRADE_LOG_CSV):
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)

    rows = chatgpt_portfolio.loc[chatgpt_portfolio["ticker"].str.upper() == ticker.upper()]
    if rows.empty:
        if chatgpt_portfolio.empty:
            chatgpt_portfolio = pd.DataFrame([{
                "ticker": ticker,
                "shares": float(shares),
                "stop_loss": float(stoploss),
                "buy_price": float(exec_price),
                "cost_basis": float(cost_amt),
            }])
        else:
            chatgpt_portfolio = pd.concat(
                [chatgpt_portfolio, pd.DataFrame([{
                    "ticker": ticker,
                    "shares": float(shares),
                    "stop_loss": float(stoploss),
                    "buy_price": float(exec_price),
                    "cost_basis": float(cost_amt),
                }])],
                ignore_index=True
            )
    else:
        idx = rows.index[0]
        cur_shares = float(chatgpt_portfolio.at[idx, "shares"])
        cur_cost = float(chatgpt_portfolio.at[idx, "cost_basis"])
        new_shares = cur_shares + float(shares)
        new_cost = cur_cost + float(cost_amt)
        chatgpt_portfolio.at[idx, "shares"] = new_shares
        chatgpt_portfolio.at[idx, "cost_basis"] = new_cost
        chatgpt_portfolio.at[idx, "buy_price"] = new_cost / new_shares if new_shares else 0.0
        chatgpt_portfolio.at[idx, "stop_loss"] = float(stoploss)

    cash -= cost_amt
    print(f"Manual BUY LIMIT for {ticker} filled at ${exec_price:.2f} ({fetch.source}).")
    return cash, chatgpt_portfolio

def log_executed_buy(
    exec_price: float,
    shares: float,
    ticker: str,
    stoploss: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    """Log a buy transaction that was already executed at a specific price.
    No market data simulation - records the exact price provided.
    """
    today = check_weekend()

    if interactive:
        check = input(
            f"Logging EXECUTED buy: {shares} {ticker} at ${exec_price:.2f}.\n"
            f"If this is a mistake, type '1' or, just hit Enter: "
        )
        if check == "1":
            print("Returning...")
            return cash, chatgpt_portfolio

    if not isinstance(chatgpt_portfolio, pd.DataFrame) or chatgpt_portfolio.empty:
        chatgpt_portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

    cost_amt = exec_price * shares
    if cost_amt > cash:
        print(f"Executed buy for {ticker} failed: cost {cost_amt:.2f} exceeds cash balance {cash:.2f}.")
        return cash, chatgpt_portfolio

    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price": exec_price,
        "Cost Basis": cost_amt,
        "PnL": 0.0,
        "Reason": "MANUAL BUY EXECUTED - Already Filled",
    }
    if os.path.exists(TRADE_LOG_CSV):
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)

    rows = chatgpt_portfolio.loc[chatgpt_portfolio["ticker"].str.upper() == ticker.upper()]
    if rows.empty:
        if chatgpt_portfolio.empty:
            chatgpt_portfolio = pd.DataFrame([{
                "ticker": ticker,
                "shares": float(shares),
                "stop_loss": float(stoploss),
                "buy_price": float(exec_price),
                "cost_basis": float(cost_amt),
            }])
        else:
            chatgpt_portfolio = pd.concat(
                [chatgpt_portfolio, pd.DataFrame([{
                    "ticker": ticker,
                    "shares": float(shares),
                    "stop_loss": float(stoploss),
                    "buy_price": float(exec_price),
                    "cost_basis": float(cost_amt),
                }])],
                ignore_index=True
            )
    else:
        idx = rows.index[0]
        cur_shares = float(chatgpt_portfolio.at[idx, "shares"])
        cur_cost = float(chatgpt_portfolio.at[idx, "cost_basis"])
        new_shares = cur_shares + float(shares)
        new_cost = cur_cost + float(cost_amt)
        chatgpt_portfolio.at[idx, "shares"] = new_shares
        chatgpt_portfolio.at[idx, "cost_basis"] = new_cost
        chatgpt_portfolio.at[idx, "buy_price"] = new_cost / new_shares if new_shares else 0.0
        chatgpt_portfolio.at[idx, "stop_loss"] = float(stoploss)

    cash -= cost_amt
    print(f"Executed BUY for {ticker} logged at ${exec_price:.2f}.")
    return cash, chatgpt_portfolio

def log_manual_sell(
    sell_price: float,
    shares_sold: float,
    ticker: str,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    reason: str | None = None,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    today = check_weekend()
    if interactive:
        reason = input(
            f"""You are placing a SELL LIMIT for {shares_sold} {ticker} at ${sell_price:.2f}.
If this is a mistake, enter 1, or hit Enter."""
        )
    if reason == "1":
        print("Returning...")
        return cash, chatgpt_portfolio
    elif reason is None:
        reason = ""

    if ticker not in chatgpt_portfolio["ticker"].values:
        print(f"Manual sell for {ticker} failed: ticker not in portfolio.")
        return cash, chatgpt_portfolio

    ticker_row = chatgpt_portfolio[chatgpt_portfolio["ticker"] == ticker]
    total_shares = float(ticker_row["shares"].item())
    # Add small tolerance for floating point precision issues
    if shares_sold > (total_shares + 1e-6):
        print(f"Manual sell for {ticker} failed: trying to sell {shares_sold} shares but only own {total_shares}.")
        return cash, chatgpt_portfolio

    s, e = trading_day_window()
    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
    data = fetch.df
    if data.empty:
        print(f"Manual sell for {ticker} failed: no market data available (source={fetch.source}).")
        return cash, chatgpt_portfolio

    o = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
    h = float(data["High"].iloc[-1])
    l = float(data["Low"].iloc[-1])
    if np.isnan(o):
        o = float(data["Close"].iloc[-1])

    if o >= sell_price:
        exec_price = o
    elif h >= sell_price:
        exec_price = sell_price
    else:
        print(f"Sell limit ${sell_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled.")
        return cash, chatgpt_portfolio

    buy_price = float(ticker_row["buy_price"].item())
    cost_basis = buy_price * shares_sold
    pnl = exec_price * shares_sold - cost_basis

    log = {
        "Date": today, "Ticker": ticker,
        "Shares Bought": "", "Buy Price": "",
        "Cost Basis": cost_basis, "PnL": pnl,
        "Reason": f"MANUAL SELL LIMIT - {reason}", "Shares Sold": shares_sold,
        "Sell Price": exec_price,
    }
    if os.path.exists(TRADE_LOG_CSV):
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)


    if total_shares == shares_sold:
        chatgpt_portfolio = chatgpt_portfolio[chatgpt_portfolio["ticker"] != ticker]
    else:
        row_index = ticker_row.index[0]
        chatgpt_portfolio.at[row_index, "shares"] = total_shares - shares_sold
        chatgpt_portfolio.at[row_index, "cost_basis"] = (
            chatgpt_portfolio.at[row_index, "shares"] * chatgpt_portfolio.at[row_index, "buy_price"]
        )

    cash += shares_sold * exec_price
    print(f"Manual SELL LIMIT for {ticker} filled at ${exec_price:.2f} ({fetch.source}).")
    return cash, chatgpt_portfolio



# ------------------------------
# Reporting / Metrics
# ------------------------------

def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics (incl. CAPM)."""
    portfolio_dict: list[dict[Any, Any]] = chatgpt_portfolio.to_dict(orient="records")
    today = check_weekend()

    rows: list[list[str]] = []
    header = ["Ticker", "Close", "% Chg", "Volume"]

    end_d = last_trading_date()                           # Fri on weekends
    start_d = (end_d - pd.Timedelta(days=4)).normalize()  # go back enough to capture 2 sessions even around holidays

    benchmarks = load_benchmarks()  # reads tickers.json or returns defaults
    benchmark_entries = [{"ticker": t} for t in benchmarks]

    for stock in portfolio_dict + benchmark_entries:
        ticker = str(stock["ticker"]).upper()
        try:
            fetch = download_price_data(ticker, start=start_d, end=(end_d + pd.Timedelta(days=1)), progress=False)
            data = fetch.df
            if data.empty or len(data) < 2:
                rows.append([ticker, "—", "—", "—"])
                continue

            price = float(data["Close"].iloc[-1])
            last_price = float(data["Close"].iloc[-2])
            volume = float(data["Volume"].iloc[-1])

            percent_change = ((price - last_price) / last_price) * 100
            rows.append([ticker, f"{price:,.2f}", f"{percent_change:+.2f}%", f"{int(volume):,}"])
        except Exception as e:
            raise Exception(f"Download for {ticker} failed. {e} Try checking internet connection.")

    # Read portfolio history from CSV
    chatgpt_df = load_full_portfolio_history()

    # Use only TOTAL rows, sorted by date
    totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        print("\n" + "=" * 64)
        print(f"Daily Results — {today}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 12, 9, 15]
        print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
        print("-" * sum(colw) + "-" * 3)
        for r in rows:
            print(f"{str(r[0]):<{colw[0]}} {str(r[1]):>{colw[1]}} {str(r[2]):>{colw[2]}} {str(r[3]):>{colw[3]}}")
        print("\n[ Portfolio Snapshot ]")
        print(chatgpt_portfolio)
        print(f"Cash balance: ${cash:,.2f}")
        return

    totals["Date"] = pd.to_datetime(totals["Date"], format="mixed", errors="coerce")  # tolerate ISO strings
    totals = totals.sort_values("Date")

    # Calculate CURRENT total equity from actual portfolio instead of using stale CSV
    current_portfolio_value = 0.0
    if not chatgpt_portfolio.empty:
        s, e = trading_day_window()
        for _, stock in chatgpt_portfolio.iterrows():
            ticker = str(stock["ticker"]).upper()
            shares = float(stock["shares"]) if not pd.isna(stock["shares"]) else 0.0
            if shares > 0:
                try:
                    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
                    if not fetch.df.empty:
                        current_price = float(fetch.df["Close"].iloc[-1])
                        position_value = shares * current_price
                        current_portfolio_value += position_value
                except Exception:
                    pass  # Skip if we can't get price data
    
    final_equity = current_portfolio_value + cash  # Use CALCULATED equity, not CSV
    equity_series = totals.set_index("Date")["Total Equity"].astype(float).sort_index()

    # --- Max Drawdown ---
    running_max = equity_series.cummax()
    drawdowns = (equity_series / running_max) - 1.0
    max_drawdown = float(drawdowns.min())  # most negative value
    mdd_date = drawdowns.idxmin()

    # Daily simple returns (portfolio)
    r = equity_series.pct_change().dropna()
    n_days = len(r)
    if n_days < 2:
        print("\n" + "=" * 64)
        print(f"Daily Results — {today}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 12, 9, 15]
        print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
        print("-" * sum(colw) + "-" * 3)
        for rrow in rows:
            print(f"{str(rrow[0]):<{colw[0]}} {str(rrow[1]):>{colw[1]}} {str(rrow[2]):>{colw[2]}} {str(rrow[3]):>{colw[3]}}")
        print("\n[ Portfolio Snapshot ]")
        print(chatgpt_portfolio)
        print(f"Cash balance: ${cash:,.2f}")
        print(f"Latest ChatGPT Equity: ${final_equity:,.2f}")
        if hasattr(mdd_date, "date") and not isinstance(mdd_date, (str, int)):
            mdd_date_str = mdd_date.date()
        elif hasattr(mdd_date, "strftime") and not isinstance(mdd_date, (str, int)):
            mdd_date_str = mdd_date.strftime("%Y-%m-%d")
        else:
            mdd_date_str = str(mdd_date)
        print(f"Maximum Drawdown: {max_drawdown:.2%} (on {mdd_date_str})")
        return

    # Risk-free config
    rf_annual = 0.045
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    rf_period = (1 + rf_daily) ** n_days - 1

    # Stats
    mean_daily = float(r.mean())
    std_daily = float(r.std(ddof=1))

    # Downside deviation (MAR = rf_daily)
    downside = (r - rf_daily).clip(upper=0)
    downside_std = float((downside.pow(2).mean()) ** 0.5) if not downside.empty else np.nan

    # Total return over the window
    r_numeric = pd.to_numeric(r, errors="coerce")
    r_numeric = r_numeric[~r_numeric.isna()].astype(float)
    # Filter out any non-finite values to ensure only valid floats are used
    r_numeric = r_numeric[np.isfinite(r_numeric)]
    # Only use numeric values for the calculation
    if len(r_numeric) > 0:
        arr = np.asarray(r_numeric.values, dtype=float)
        period_return = float(np.prod(1 + arr) - 1) if arr.size > 0 else float('nan')
    else:
        period_return = float('nan')

    # Sharpe / Sortino
    sharpe_period = (period_return - rf_period) / (std_daily * np.sqrt(n_days)) if std_daily > 0 else np.nan
    sharpe_annual = ((mean_daily - rf_daily) / std_daily) * np.sqrt(252) if std_daily > 0 else np.nan
    sortino_period = (period_return - rf_period) / (downside_std * np.sqrt(n_days)) if downside_std and downside_std > 0 else np.nan
    sortino_annual = ((mean_daily - rf_daily) / downside_std) * np.sqrt(252) if downside_std and downside_std > 0 else np.nan

    # -------- CAPM: Beta & Alpha (vs ^GSPC) --------
    start_date = equity_series.index.min() - pd.Timedelta(days=1)
    end_date = equity_series.index.max() + pd.Timedelta(days=1)

    spx_fetch = download_price_data("^GSPC", start=start_date, end=end_date, progress=False)
    spx = spx_fetch.df

    beta = np.nan
    alpha_annual = np.nan
    r2 = np.nan
    n_obs = 0

    if not spx.empty and len(spx) >= 2:
        spx = spx.reset_index().set_index("Date").sort_index()
        mkt_ret = spx["Close"].astype(float).pct_change().dropna()

        # Align portfolio & market returns
        common_idx = r.index.intersection(list(mkt_ret.index))
        if len(common_idx) >= 2:
            rp = (r.reindex(common_idx).astype(float) - rf_daily)   # portfolio excess
            rm = (mkt_ret.reindex(common_idx).astype(float) - rf_daily)  # market excess

            x = np.asarray(rm.values, dtype=float).ravel()
            y = np.asarray(rp.values, dtype=float).ravel()

            n_obs = x.size
            rm_std = float(np.std(x, ddof=1)) if n_obs > 1 else 0.0
            if rm_std > 0:
                beta, alpha_daily = np.polyfit(x, y, 1)
                alpha_annual = (1 + float(alpha_daily)) ** 252 - 1

                corr = np.corrcoef(x, y)[0, 1]
                r2 = float(corr ** 2)

    # $X normalized S&P 500 over same window (uses remembered starting equity)
    spx_norm_fetch = download_price_data(
        "^GSPC",
        start=equity_series.index.min(),
        end=equity_series.index.max() + pd.Timedelta(days=1),
        progress=False,
    )
    spx_norm = spx_norm_fetch.df
    spx_value = np.nan

    # Use the first Total Equity value from the series as starting equity
    # This eliminates the need to ask every time
    starting_equity = equity_series.iloc[0] if not equity_series.empty else np.nan

    if not spx_norm.empty and not np.isnan(starting_equity):
        initial_price = float(spx_norm["Close"].iloc[0])
        price_now = float(spx_norm["Close"].iloc[-1])
        spx_value = (starting_equity / initial_price) * price_now

    # -------- Pretty Printing --------
    print("\n" + "=" * 64)
    print(f"Daily Results — {today}")
    print("=" * 64)

    # Price & Volume table
    print("\n[ Price & Volume ]")
    colw = [10, 12, 9, 15]
    print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
    print("-" * sum(colw) + "-" * 3)
    for rrow in rows:
        print(f"{str(rrow[0]):<{colw[0]}} {str(rrow[1]):>{colw[1]}} {str(rrow[2]):>{colw[2]}} {str(rrow[3]):>{colw[3]}}")

    # Performance metrics
    def fmt_or_na(x: float | int | None, fmt: str) -> str:
        return (fmt.format(x) if not (x is None or (isinstance(x, float) and np.isnan(x))) else "N/A")

    print("\n[ Risk & Return ]")
    if hasattr(mdd_date, "date") and not isinstance(mdd_date, (str, int)):
        mdd_date_str = mdd_date.date()
    elif hasattr(mdd_date, "strftime") and not isinstance(mdd_date, (str, int)):
        mdd_date_str = mdd_date.strftime("%Y-%m-%d")
    else:
        mdd_date_str = str(mdd_date)
    print(f"{'Max Drawdown:':32} {fmt_or_na(max_drawdown, '{:.2%}'):>15}   on {mdd_date_str}")
    print(f"{'Sharpe Ratio (period):':32} {fmt_or_na(sharpe_period, '{:.4f}'):>15}")
    print(f"{'Sharpe Ratio (annualized):':32} {fmt_or_na(sharpe_annual, '{:.4f}'):>15}")
    print(f"{'Sortino Ratio (period):':32} {fmt_or_na(sortino_period, '{:.4f}'):>15}")
    print(f"{'Sortino Ratio (annualized):':32} {fmt_or_na(sortino_annual, '{:.4f}'):>15}")

    print("\n[ CAPM vs Benchmarks ]")
    if not np.isnan(beta):
        print(f"{'Beta (daily) vs ^GSPC:':32} {beta:>15.4f}")
        print(f"{'Alpha (annualized) vs ^GSPC:':32} {alpha_annual:>15.2%}")
        print(f"{'R² (fit quality):':32} {r2:>15.3f}   {'Obs:':>6} {n_obs}")
        if n_obs < 60 or (not np.isnan(r2) and r2 < 0.20):
            print("  Note: Short sample and/or low R² — alpha/beta may be unstable.")
    else:
        print("Beta/Alpha: insufficient overlapping data.")

    print("\n[ Snapshot ]")
    print(f"{'Latest ChatGPT Equity:':32} ${final_equity:>14,.2f}")
    if not np.isnan(spx_value):
        try:
            print(f"{f'${starting_equity:.0f} in S&P 500 (same window):':32} ${spx_value:>14,.2f}")
        except Exception:
            pass
    print(f"{'Cash Balance:':32} ${cash:>14,.2f}")

    # Show today's transactions to explain cash changes
    print("\n[ Today's Transactions ]")
    if TRADE_LOG_CSV.exists():
        try:
            trade_log = pd.read_csv(TRADE_LOG_CSV)
            today_trades = trade_log[trade_log["Date"] == today]

            if not today_trades.empty:
                print("Transaction history explaining cash flow:")
                for _, trade in today_trades.iterrows():
                    ticker = trade["Ticker"]
                    reason = trade["Reason"]

                    if "BUY" in reason:
                        shares = trade["Shares Bought"] if pd.notna(trade["Shares Bought"]) else 0
                        buy_price = trade["Buy Price"] if pd.notna(trade["Buy Price"]) else 0
                        cost = trade["Cost Basis"] if pd.notna(trade["Cost Basis"]) else 0
                        print(f"  📤 BUY:  {ticker} | {shares:.4f} shares @ ${buy_price:.2f} = -${cost:.2f}")
                    elif "SELL" in reason:
                        shares = trade["Shares Sold"] if pd.notna(trade["Shares Sold"]) else 0
                        sell_price = trade["Sell Price"] if pd.notna(trade["Sell Price"]) else 0
                        proceeds = shares * sell_price
                        pnl = trade["PnL"] if pd.notna(trade["PnL"]) else 0
                        pnl_sign = "+" if pnl >= 0 else ""
                        print(f"  📥 SELL: {ticker} | {shares:.4f} shares @ ${sell_price:.2f} = +${proceeds:.2f} (PnL: {pnl_sign}${pnl:.2f})")

                # Calculate net cash impact
                total_inflow = 0
                total_outflow = 0
                for _, trade in today_trades.iterrows():
                    if "BUY" in trade["Reason"]:
                        cost = trade["Cost Basis"] if pd.notna(trade["Cost Basis"]) else 0
                        total_outflow += cost
                    elif "SELL" in trade["Reason"]:
                        shares = trade["Shares Sold"] if pd.notna(trade["Shares Sold"]) else 0
                        sell_price = trade["Sell Price"] if pd.notna(trade["Sell Price"]) else 0
                        total_inflow += shares * sell_price

                net_cash_flow = total_inflow - total_outflow
                flow_direction = "+" if net_cash_flow >= 0 else ""
                print(f"  💰 NET:  Cash flow today = {flow_direction}${net_cash_flow:.2f} (${total_inflow:.2f} in, ${total_outflow:.2f} out)")
            else:
                print("No transactions today.")

        except Exception as e:
            print(f"Could not read trade log: {e}")
    else:
        print("No trade log found.")

    print("\n[ Holdings ]")
    print(chatgpt_portfolio)

    # New section: Performance of each held ticker since purchase
    print("\n[ Performance Since Purchase ]")
    if not chatgpt_portfolio.empty and len(chatgpt_portfolio) > 0:
        print("Ticker    Buy Price  Current Price  Total Change    % Change")
        print("-" * 62)

        s, e = trading_day_window()
        for _, stock in chatgpt_portfolio.iterrows():
            ticker = str(stock["ticker"]).upper()
            shares = float(stock["shares"]) if not pd.isna(stock["shares"]) else 0.0
            buy_price = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0

            # Only show positions with shares (skip sold positions)
            if shares > 0:
                # Get current price
                fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
                data = fetch.df

                if not data.empty:
                    current_price = float(data["Close"].iloc[-1])
                    total_change = current_price - buy_price
                    pct_change = (total_change / buy_price * 100) if buy_price > 0 else 0

                    # Format the output with proper alignment
                    change_sign = "+" if total_change >= 0 else ""
                    pct_sign = "+" if pct_change >= 0 else ""

                    print(f"{ticker:<8}  ${buy_price:>8.2f}  ${current_price:>12.2f}  {change_sign}${total_change:>9.2f}    {pct_sign}{pct_change:>6.2f}%")
                else:
                    print(f"{ticker:<8}  ${buy_price:>8.2f}  {'N/A':>12}  {'N/A':>10}    {'N/A':>7}")
    else:
        print("No current holdings to display.")

    print("\n[ Your Instructions ]")
    print(
        "You are a professional-grade portfolio analyst. You have a portfolio, and above is your current portfolio: \n"
        "(insert `[ Holdings ]` & `[ Snapshot ]` portion of last daily prompt above).\n"

        "Use this info to make decisions regarding your portfolio. You have complete control over every decision. Make any changes you believe are beneficial—no approval required.\n"
        "Deep research is not permitted. Act at your discretion to achieve the best outcome.\n"
        "If you do not make a clear indication to change positions IMMEDIATELY after this message, the portfolio remains unchanged for tomorrow.\n"
        "You are encouraged to use the internet to check current prices (and related up-to-date info) for potential buys.\n"
        "My goal is Aggressive Alpha/Momentum. I will not tolerate ranging stocks like MSFT (as of Mid October 2025), which should be considered for divestment if needed.\n"
         "I am only interested in high-volatility, explosive growth opportunities.\n"
        "The Energy sector should be avoided for the short term as it is not aligned with my goal. This decision should be re-evaluate at the begining of November 2025.\n"
        "Market size of the stocks you inspect should not be less then 500M USD\n"
        "\n"
        "*Paste everything above into ChatGPT*"
    )


# ------------------------------
# Configuration Management
# ------------------------------

def load_config() -> dict:
    """Load configuration from config.json file."""
    config_path = DATA_DIR / "config.json"

    default_config = {
        "initial_cash": 612.00,
        "last_updated": "2025-10-03",
        "settings": {
            "default_benchmarks": ["IWO", "XBI", "SPY", "IWM"],
            "portfolio_csv": "Start Your Own/chatgpt_portfolio_update.csv",
            "trade_log_csv": "chatgpt_trade_log.csv"
        }
    }

    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            # Create default config file
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default configuration file: {config_path}")
            return default_config
    except Exception as e:
        print(f"Error loading config file, using defaults: {e}")
        return default_config

def update_config_cash(new_amount: float) -> None:
    """Update the initial cash amount in the configuration file."""
    config_path = DATA_DIR / "config.json"
    config = load_config()
    config["initial_cash"] = new_amount
    config["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Updated initial cash amount to ${new_amount:.2f} in config file.")
    except Exception as e:
        print(f"Error updating config file: {e}")

# ------------------------------
# Orchestration
# ------------------------------

def _read_csv_with_encoding_fallback(csv_path: Path) -> pd.DataFrame:
    """Robust CSV reader with encoding fallback logic."""
    logger.info("Reading CSV file: %s", csv_path)
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning("UTF-8 encoding failed, trying latin-1 encoding")
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
        except UnicodeDecodeError:
            logger.warning("latin-1 encoding failed, trying cp1252 encoding")
            df = pd.read_csv(csv_path, encoding='cp1252')
    logger.info("Successfully read CSV file: %s", csv_path)
    return df

def load_full_portfolio_history() -> pd.DataFrame:
    """Load the complete portfolio history from CSV file for performance analysis."""
    if PORTFOLIO_CSV.exists():
        df = _read_csv_with_encoding_fallback(PORTFOLIO_CSV)
        # Remove blank lines
        df = df.dropna(how='all')
        return df
    else:
        return pd.DataFrame()

def load_latest_portfolio_state() -> tuple[pd.DataFrame | list[dict[str, Any]], float]:
    """Load the most recent portfolio snapshot and cash balance from CSV file."""
    if PORTFOLIO_CSV.exists():
        df = _read_csv_with_encoding_fallback(PORTFOLIO_CSV)
        # Remove blank lines
        df = df.dropna(how='all')
    else:
        df = pd.DataFrame()

    if df.empty:
        portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        print("Portfolio CSV file not found or empty. Creating new portfolio.")

        # Load initial cash from config file
        config = load_config()
        cash = config["initial_cash"]
        print(f"Using initial cash amount from config: ${cash:.2f}")
        print("(Use --update-cash command line option to change this value)")

        return portfolio, cash

    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"], format="mixed", errors="coerce")

    latest_date = non_total["Date"].max()
    latest_tickers = non_total[non_total["Date"] == latest_date].copy()
    
    # Get previous day's positions to build current portfolio from
    previous_dates = non_total[non_total["Date"] < latest_date]["Date"].unique()
    current_positions = {}
    
    if len(previous_dates) > 0:
        # Start with previous day's portfolio
        previous_date = max(previous_dates)
        previous_day = non_total[non_total["Date"] == previous_date].copy()
        previous_day = previous_day[~previous_day["Action"].astype(str).str.startswith("SELL")]
        
        for _, row in previous_day.iterrows():
            ticker = str(row["Ticker"]).upper()
            current_positions[ticker] = {
                'ticker': ticker,
                'shares': float(row["Shares"]) if pd.notna(row["Shares"]) else 0.0,
                'cost_basis': float(row["Cost Basis"]) if pd.notna(row["Cost Basis"]) else 0.0,
                'buy_price': float(row["Buy Price"]) if pd.notna(row["Buy Price"]) else 0.0,
                'stop_loss': float(row["Stop Loss"]) if pd.notna(row["Stop Loss"]) else 0.0
            }
    
    # Apply today's BUY transactions from CSV
    buy_transactions = latest_tickers[latest_tickers["Action"].astype(str).str.contains("BUY")]
    for _, row in buy_transactions.iterrows():
        ticker = str(row["Ticker"]).upper()
        shares = float(row["Shares"]) if pd.notna(row["Shares"]) else 0.0
        buy_price = float(row["Buy Price"]) if pd.notna(row["Buy Price"]) else 0.0
        cost = shares * buy_price
        stop_loss = float(row["Stop Loss"]) if pd.notna(row["Stop Loss"]) else 0.0
        
        if ticker not in current_positions:
            current_positions[ticker] = {
                'ticker': ticker,
                'shares': shares,
                'cost_basis': cost,
                'buy_price': buy_price,
                'stop_loss': stop_loss
            }
        else:
            # Add to existing position
            old_shares = current_positions[ticker]['shares']
            old_cost = current_positions[ticker]['cost_basis']
            new_shares = old_shares + shares
            new_cost = old_cost + cost
            new_avg_price = new_cost / new_shares if new_shares > 0 else 0.0
            
            current_positions[ticker]['shares'] = round(new_shares, 4)
            current_positions[ticker]['cost_basis'] = round(new_cost, 2)
            current_positions[ticker]['buy_price'] = round(new_avg_price, 2)
            current_positions[ticker]['stop_loss'] = stop_loss
    
    # Apply today's SELL transactions from trade log
    if TRADE_LOG_CSV.exists():
        try:
            trade_log = pd.read_csv(TRADE_LOG_CSV)
            today_iso = latest_date.date().isoformat()
            today_sells = trade_log[
                (trade_log["Date"] == today_iso) &
                (pd.notna(trade_log["Shares Sold"])) &
                (trade_log["Shares Sold"] > 0)
            ]
            
            for _, sell_row in today_sells.iterrows():
                ticker = str(sell_row["Ticker"]).upper()
                shares_sold = float(sell_row["Shares Sold"])
                
                if ticker in current_positions:
                    current_shares = current_positions[ticker]['shares']
                    
                    if shares_sold >= current_shares:
                        # Selling all shares - remove position
                        del current_positions[ticker]
                    else:
                        # Partial sell - reduce position proportionally
                        remaining_shares = current_shares - shares_sold
                        current_cost = current_positions[ticker]['cost_basis']
                        remaining_cost = current_cost * (remaining_shares / current_shares)
                        
                        current_positions[ticker]['shares'] = round(remaining_shares, 4)
                        current_positions[ticker]['cost_basis'] = round(remaining_cost, 2)
                        # buy_price stays the same
        except Exception as e:
            logger.warning("Could not process trade log for sells: %s", e)
    
    # Convert to list format expected by rest of function
    latest_tickers = pd.DataFrame(list(current_positions.values()))
    
    # Skip the old filtering logic since we've rebuilt the portfolio correctly
    # Convert DataFrame to list of dictionaries for return
    if not latest_tickers.empty:
        latest_tickers = latest_tickers.to_dict('records')
    else:
        latest_tickers = []

    df_total = df[df["Ticker"] == "TOTAL"].copy()
    df_total["Date"] = pd.to_datetime(df_total["Date"], format="mixed", errors="coerce")

    # Find the latest TOTAL row with a valid (non-NaN) cash balance
    valid_total = df_total[pd.notna(df_total["Cash Balance"])]
    if valid_total.empty:
        # No valid cash balance found - use config file
        config = load_config()
        cash = config["initial_cash"]
        print(f"No valid cash balance found in CSV. Using initial cash from config: ${cash:.2f}")
        print("(Use --update-cash command line option to change this value)")
    else:
        latest = valid_total.sort_values("Date").iloc[-1]
        cash = float(latest["Cash Balance"])
    return latest_tickers, cash


def main(data_dir: Path | None = None) -> None:
    """Check versions, then run the trading script."""
    if data_dir is not None:
        set_data_dir(data_dir)

    chatgpt_portfolio, cash = load_latest_portfolio_state()
    chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash)
    daily_results(chatgpt_portfolio, cash)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Optional data directory")
    parser.add_argument("--asof", default=None, help="Treat this YYYY-MM-DD as 'today' (e.g., 2025-08-27)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level (default: INFO)")
    parser.add_argument("--update-cash", type=float, default=None,
                       help="Update the initial cash amount in config file (e.g., --update-cash 612.00)")
    args = parser.parse_args()


    # Configure logging level
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format=' %(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
    )

    # Log initial global state and command-line arguments
    _log_initial_state()
    logger.info("Script started with arguments: %s", vars(args))

    # Handle --update-cash option first
    if args.update_cash is not None:
        if args.data_dir:
            set_data_dir(Path(args.data_dir))
        update_config_cash(args.update_cash)
        exit(0)  # Exit after updating config

    if args.asof:
        set_asof(args.asof)

    main(Path(args.data_dir) if args.data_dir else None)
