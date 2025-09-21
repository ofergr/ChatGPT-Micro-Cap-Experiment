import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from pathlib import Path  # NEW

DATA_DIR = "Scripts and CSV Files"
PORTFOLIO_CSV = f"{DATA_DIR}/chatgpt_portfolio_update.csv"

# Save path in project root
RESULTS_PATH = Path("Results.png")  # NEW


def load_portfolio_totals() -> pd.DataFrame:
    """Load portfolio equity history including a baseline row."""
    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)
    chatgpt_totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    chatgpt_totals["Date"] = pd.to_datetime(chatgpt_totals["Date"])
    chatgpt_totals["Total Equity"] = pd.to_numeric(
        chatgpt_totals["Total Equity"], errors="coerce"
    )

    baseline_date = pd.Timestamp("2025-06-27")
    baseline_equity = 100.0
    baseline_row = pd.DataFrame({"Date": [baseline_date], "Total Equity": [baseline_equity]})

    out = pd.concat([baseline_row, chatgpt_totals], ignore_index=True).sort_values("Date")
    out = out.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return out


def download_sp500(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Download S&P 500 prices and normalise to a $100 baseline (at 2025-06-27 close=6173.07)."""
    sp500 = yf.download("^SPX", start=start_date, end=end_date + pd.Timedelta(days=1),
                        progress=False, auto_adjust=True)
    sp500 = sp500.reset_index()
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)

    spx_27_price = 6173.07  # 2025-06-27 close (baseline)
    scaling_factor = 100.0 / spx_27_price
    sp500["SPX Value ($100 Invested)"] = sp500["Close"] * scaling_factor
    return sp500[["Date", "SPX Value ($100 Invested)"]]


def find_largest_gain(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """
    Largest rise from a local minimum to the subsequent peak.
    Returns (start_date, end_date, gain_pct).
    """
    df = df.sort_values("Date")
    min_val = float(df["Total Equity"].iloc[0])
    min_date = pd.Timestamp(df["Date"].iloc[0])
    peak_val = min_val
    peak_date = min_date
    best_gain = 0.0
    best_start = min_date
    best_end = peak_date

    # iterate rows 1..end
    for date, val in df[["Date", "Total Equity"]].iloc[1:].itertuples(index=False):
        val = float(val)
        date = pd.Timestamp(date)

        # extend peak while rising
        if val > peak_val:
            peak_val = val
            peak_date = date
            continue

        # fall → close previous run
        if val < peak_val:
            gain = (peak_val - min_val) / min_val * 100.0
            if gain > best_gain:
                best_gain = gain
                best_start = min_date
                best_end = peak_date
            # reset min/peak at this valley
            min_val = val
            min_date = date
            peak_val = val
            peak_date = date

    # final run (if last segment ends on a rise)
    gain = (peak_val - min_val) / min_val * 100.0
    if gain > best_gain:
        best_gain = gain
        best_start = min_date
        best_end = peak_date

    return best_start, best_end, best_gain


def compute_drawdown(df: pd.DataFrame) -> tuple[pd.Timestamp, float, float]:
    """
    Compute running max and drawdown (%). Return (dd_date, dd_value, dd_pct).
    """
    df = df.sort_values("Date").copy()
    df["Running Max"] = df["Total Equity"].cummax()
    df["Drawdown %"] = (df["Total Equity"] / df["Running Max"] - 1.0) * 100.0
    row = df.loc[df["Drawdown %"].idxmin()]
    return pd.Timestamp(row["Date"]), float(row["Total Equity"]), float(row["Drawdown %"])


def main() -> dict:
    """Generate and display the comparison graph; return metrics."""
    chatgpt_totals = load_portfolio_totals()

    start_date = pd.Timestamp("2025-06-27")
    end_date = chatgpt_totals["Date"].max()
    sp500 = download_sp500(start_date, end_date)

    # metrics
    largest_start, largest_end, largest_gain = find_largest_gain(chatgpt_totals)
    dd_date, dd_value, dd_pct = compute_drawdown(chatgpt_totals)

    # plotting
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.plot(
        chatgpt_totals["Date"],
        chatgpt_totals["Total Equity"],
        label="ChatGPT ($100 Invested)",
        marker="o",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        sp500["Date"],
        sp500["SPX Value ($100 Invested)"],
        label="S&P 500 ($100 Invested)",
        marker="o",
        color="orange",
        linestyle="--",
        linewidth=2,
    )

    # annotate largest gain
    largest_peak_value = float(
        chatgpt_totals.loc[chatgpt_totals["Date"] == largest_end, "Total Equity"].iloc[0]
    )
    plt.text(
        largest_end,
        largest_peak_value + 0.3,
        f"+{largest_gain:.1f}% largest gain",
        color="green",
        fontsize=9,
    )

    # annotate final P/Ls
    final_date = chatgpt_totals["Date"].iloc[-1]
    final_chatgpt = float(chatgpt_totals["Total Equity"].iloc[-1])
    final_spx = float(sp500["SPX Value ($100 Invested)"].iloc[-1])
    plt.text(final_date, final_chatgpt + 0.5, f"{final_chatgpt - 100.0:.1f}%", color="blue", fontsize=9)
    plt.text(final_date, final_spx + 0.9, f"+{final_spx - 100.0:.1f}%", color="orange", fontsize=9)

    # label ATYR's catalyst failure
    plt.text(

        pd.to_datetime("2025-09-13") + pd.Timedelta(days=0.5),
        125,
        f"ATYR falls ~80%",
        color="red",
        fontsize=9,
    )
    # annotate max drawdown
    plt.text(
        dd_date + pd.Timedelta(days=0.5),
        dd_value - 0.5,
        f"{dd_pct:.1f}%",
        color="red",
        fontsize=9,
    )

    plt.title("ChatGPT's Micro Cap Portfolio vs. S&P 500")
    plt.xlabel("Date")
    plt.ylabel("Value of $100 Investment")
    plt.xticks(rotation=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Auto-save to project root ---
    plt.savefig(RESULTS_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved chart to: {RESULTS_PATH.resolve()}")

    plt.show()

    return {
        "largest_run_start": largest_start,
        "largest_run_end": largest_end,
        "largest_run_gain_pct": largest_gain,
        "max_drawdown_date": dd_date,
        "max_drawdown_equity": dd_value,
        "max_drawdown_pct": dd_pct,
    }


if __name__ == "__main__":
    print("generating graph...")

    metrics = main()
    ls = metrics["largest_run_start"].date()
    le = metrics["largest_run_end"].date()
    lg = metrics["largest_run_gain_pct"]
    dd_d = metrics["max_drawdown_date"].date()
    dd_e = metrics["max_drawdown_equity"]
    dd_p = metrics["max_drawdown_pct"]
    print(f"Largest run: {ls} → {le}, +{lg:.2f}%")
    print(f"Max drawdown: {dd_p:.2f}% on {dd_d} (equity {dd_e:.2f})")