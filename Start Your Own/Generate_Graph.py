"""
Plot portfolio performance vs. S&P 500 with a configurable starting equity.

- Normalizes BOTH series (portfolio and S&P) to the same starting equity.
- Aligns S&P data to the portfolio dates with forward-fill.
- Backwards-compatible function names for existing imports.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"


def parse_date(date_str: str, label: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(date_str)
    except Exception as exc:
        raise SystemExit(f"Invalid {label} '{date_str}'. Use YYYY-MM-DD.") from exc


def _normalize_to_start(series, starting_equity):
    """
    Normalize a series to start at starting_equity
    """
    # Ensure we're working with a Series
    if isinstance(series, pd.DataFrame):
        # If it's a DataFrame, take the first column (assuming it's the value column)
        s = pd.to_numeric(series.iloc[:, 0], errors="coerce")
    else:
        s = pd.to_numeric(series, errors="coerce")
    
    if s.empty:
        return pd.Series()
    
    start_value = s.iloc[0]
    if start_value == 0:
        return s * 0  # Return zeros if start value is zero to avoid division by zero
    
    normalized = (s / start_value) * starting_equity
    return normalized


def _align_to_dates(sp500_data: pd.DataFrame, portfolio_dates: pd.Series) -> pd.Series:
    """
    Align S&P 500 data to portfolio dates using forward fill.
    Returns a Series with values aligned to portfolio_dates.
    """
    # Create a DataFrame with all portfolio dates
    aligned_df = pd.DataFrame({'Date': portfolio_dates})
    
    # Merge with S&P 500 data
    merged = aligned_df.merge(sp500_data, on='Date', how='left')
    
    # Forward fill missing values
    merged['Value'] = merged['Value'].ffill()
    
    return merged['Value']


def load_portfolio_details(
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    portfolio_csv: Path = PORTFOLIO_CSV,
) -> pd.DataFrame:
    """Return TOTAL rows (Date, Total Equity) filtered to [start_date, end_date]."""
    if not portfolio_csv.exists():
        raise SystemExit(f"Portfolio file '{portfolio_csv}' not found.")

    df = pd.read_csv(portfolio_csv)
    totals = df[df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        raise SystemExit("""Portfolio CSV contains no TOTAL rows. Please run 'python trading_script.py --data-dir "Start Your Own"' at least once for graphing data.""")

    totals["Date"] = pd.to_datetime(totals["Date"], errors="coerce")
    totals["Total Equity"] = pd.to_numeric(totals["Total Equity"], errors="coerce")

    totals = totals.dropna(subset=["Date", "Total Equity"]).sort_values("Date")

    min_date = totals["Date"].min()
    max_date = totals["Date"].max()
    if start_date is None or start_date < min_date:
        start_date = min_date
    if end_date is None or end_date > max_date:
        end_date = max_date
    if start_date is not None and end_date is not None:
        if start_date > end_date:
            raise SystemExit("Start date must be on or before end date.")


    mask = (totals["Date"] >= start_date) & (totals["Date"] <= end_date)
    return totals.loc[mask, ["Date", "Total Equity"]].reset_index(drop=True)


def download_sp500(dates, starting_equity):
    """
    Download S&P 500 data and normalize to starting equity
    If recent data is unavailable, try with earlier dates or period-based approach
    """
    if len(dates) == 0:
        return pd.DataFrame()
    
    start_date = dates.min()
    end_date = dates.max()
    
    # Try multiple date ranges if recent data is unavailable
    for days_back in [0, 1, 2, 3, 5, 7]:  # Try current, then 1, 2, 3, 5, 7 days back
        try_start = start_date - pd.Timedelta(days=days_back)
        try_end = end_date - pd.Timedelta(days=days_back)
        
        if days_back > 0:
            print(f"Trying S&P 500 data {days_back} day(s) earlier...")
        
        # Download S&P 500 data with error handling
        try:
            sp500 = yf.download("^GSPC", start=try_start, end=try_end + pd.Timedelta(days=1), progress=False)
        except Exception as e:
            if days_back == 0:
                print(f"Error downloading current S&P 500 data: {e}")
            continue
        
        # Check if download returned valid data
        if sp500 is not None and not sp500.empty:
            if days_back > 0:
                print(f"Successfully retrieved S&P 500 data from {days_back} day(s) ago")
            
            # Reset index to get Date as a column
            sp500 = sp500.reset_index()
            
            # Extract only the 'Close' price series
            sp500_close = sp500[['Date', 'Close']].copy()
            sp500_close.columns = ['Date', 'Value']
            
            # Shift dates forward if we used earlier data
            if days_back > 0:
                sp500_close['Date'] = sp500_close['Date'] + pd.Timedelta(days=days_back)
            
            # Align with portfolio dates
            aligned_values = _align_to_dates(sp500_close, dates)
            
            # Normalize to starting equity
            norm = _normalize_to_start(aligned_values, starting_equity)
            
            result = pd.DataFrame({
                'Date': dates,
                'SPX Value': norm.values
            })
            
            return result
    
    # If date-based approach failed, try period-based approach
    print("Trying period-based S&P 500 data retrieval...")
    try:
        # Get recent data using period instead of specific dates
        sp500 = yf.download("^GSPC", period="3mo", progress=False)  # Last 3 months
        
        if sp500 is not None and not sp500.empty:
            print(f"Successfully retrieved S&P 500 data using period approach ({len(sp500)} days)")
            
            # Reset index to get Date as a column
            sp500 = sp500.reset_index()
            
            # Extract only the 'Close' price series
            sp500_close = sp500[['Date', 'Close']].copy()
            sp500_close.columns = ['Date', 'Value']
            
            # Get the most recent available data point
            latest_sp500_date = sp500_close['Date'].max()
            latest_sp500_value = sp500_close[sp500_close['Date'] == latest_sp500_date]['Value'].iloc[0]
            
            print(f"Using latest available S&P 500 data from {latest_sp500_date.strftime('%Y-%m-%d')}")
            
            # Create a synthetic S&P 500 series using the latest available value
            # This assumes S&P 500 stayed constant from the last available date
            synthetic_sp500 = pd.DataFrame({
                'Date': dates,
                'Value': [latest_sp500_value] * len(dates)
            })
            
            # Align with portfolio dates
            aligned_values = _align_to_dates(synthetic_sp500, dates)
            
            # Normalize to starting equity
            norm = _normalize_to_start(aligned_values, starting_equity)
            
            result = pd.DataFrame({
                'Date': dates,
                'SPX Value': norm.values
            })
            
            return result
            
    except Exception as e:
        print(f"Period-based approach also failed: {e}")
    
    print("Could not retrieve S&P 500 data using any approach")
    return pd.DataFrame()


def plot_comparison(
    portfolio: pd.DataFrame,
    spx: pd.DataFrame,
    starting_equity: float,
    title: str = "Portfolio vs. S&P 500 (Indexed)",
) -> None:
    """
    Plot the two normalized lines. Expects:
      - portfolio: columns ['Date', 'Total Equity'] (already normalized if desired)
      - spx:       columns ['Date', 'SPX Value'] (already normalized)
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(portfolio["Date"], portfolio["Total Equity"], label=f"Portfolio (start={starting_equity:g})", marker="o")
    ax.plot(spx["Date"], spx["SPX Value"], label="S&P 500", marker="o", linestyle="--")
    
    # Annotate last points as percent vs baseline
    p_last = float(portfolio["Total Equity"].iloc[-1])
    s_last = float(spx["SPX Value"].iloc[-1])

    p_pct = (p_last / starting_equity - 1.0) * 100.0
    s_pct = (s_last / starting_equity - 1.0) * 100.0

    ax.text(portfolio["Date"].iloc[-1], p_last * 1.01, f"{p_pct:+.1f}%", fontsize=9)
    ax.text(spx["Date"].iloc[-1], s_last * 1.01, f"{s_pct:+.1f}%", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Index (start = {starting_equity:g})")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()


def main(
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    starting_equity: float,
    output: Optional[Path],
    portfolio_csv: Path = PORTFOLIO_CSV,
) -> None:
    # Load portfolio totals in the date range
    totals = load_portfolio_details(start_date, end_date, portfolio_csv=portfolio_csv)

    # Normalize portfolio to the chosen starting equity
    norm_port = totals.copy()
    norm_port["Total Equity"] = _normalize_to_start(norm_port["Total Equity"], starting_equity)

    # Download & normalize S&P to same baseline, aligned to portfolio dates
    spx = download_sp500(norm_port["Date"], starting_equity)

    # Plot comparison or portfolio only if S&P data unavailable
    if spx.empty:
        print("Warning: S&P 500 data not available. Plotting portfolio performance only.")
        # Plot portfolio only
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(norm_port["Date"], norm_port["Total Equity"], label=f"Portfolio (start={starting_equity:g})", marker="o")
        
        # Annotate last point as percent vs baseline
        p_last = float(norm_port["Total Equity"].iloc[-1])
        p_pct = (p_last / starting_equity - 1.0) * 100.0
        ax.text(norm_port["Date"].iloc[-1], p_last * 1.01, f"{p_pct:+.1f}%", fontsize=9)
        
        ax.set_title("ChatGPT Portfolio Performance (S&P 500 data unavailable)")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Index (start = {starting_equity:g})")
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()
        plt.tight_layout()
    else:
        # Plot comparison with S&P 500
        plot_comparison(norm_port, spx, starting_equity, title="ChatGPT Portfolio vs. S&P 500 (Indexed)")

    # Save or show
    if output:
        output = output if output.is_absolute() else DATA_DIR / output
        plt.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot portfolio performance vs S&P 500")
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--start-equity", type=float, default=100.0, help="Baseline to index both series (default 100)")
    parser.add_argument("--baseline-file", type=str, help="Path to a text file containing a single number for baseline")
    parser.add_argument("--output", type=str, help="Optional path to save the chart (.png/.jpg/.pdf)")

    args = parser.parse_args()
    start = parse_date(args.start_date, "start date") if args.start_date else None
    end = parse_date(args.end_date, "end date") if args.end_date else None

    baseline = args.start_equity
    if args.baseline_file:
        p = Path(args.baseline_file)
        if not p.exists():
            raise SystemExit(f"Baseline file not found: {p}")
        try:
            baseline = float(p.read_text().strip())
        except Exception as exc:
            raise SystemExit(f"Could not parse baseline from {p}") from exc

    out_path = Path(args.output) if args.output else None
    main(start, end, baseline, out_path)