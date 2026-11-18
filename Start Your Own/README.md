# Start Your Own

This folder lets you run the trading experiment on your own computer. It contains two small scripts and the CSV files they produce.

Run the commands below from the repository root. The scripts automatically
save their CSV data inside this folder.

## Overview

 **Install dependencies:**
   ```bash
   # Recommended: Use a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   pip install -r requirements.txt
   ```

**Processing Portfolio:**
   ```bash
   # ALWAYS include a CSV file of history

   python trading_script.py --data-dir "Start Your Own"
   ```

**To Save Prior Days:**
   ```bash

   # Save data with specific date
   python trading_script.py --asof 2025-08-27 --data-dir "Start Your Own"
   ```

**Generate performance graphs:**
   ```bash
   python "Start Your Own/Generate_Graph.py"
   ```

### Argument Table for 'Generate_Graph.py'

| Argument            | Type   | Default          | Description                                                        |
|---------------------|--------|------------|--------------------------------------------------------------------------|
| `--start-date`      | str    | Start date in CSV| Start date in `YYYY-MM-DD` format                                  |
| `--end-date`        | str    | End date in CSV| End date in `YYYY-MM-DD` format                                      |
| `--start-equity`    | float  | 100.0   | Baseline to index both series (default 100)                                 |
| `--output`          | str    | —       | Optional path to save the chart (`.png` / `.jpg` / `.pdf`)                  |

## ProcessPortfolio.py

### IMPORTANT

Always run the program after the market closes at 4:00 PM EST, otherwise it will default to using the previous day’s data.

Because the program relies on past data, orders for a given day are generated after that day’s trading session and must be placed on the following trading day. This prevents lookahead bias. For example, When I receive orders from ChatGPT, I run the program and input the orders the close the day after.  

This script updates your portfolio and logs trades.

**Information**
   - The program uses past data from 'chatgpt_portfolio_update.csv' to automatically grab today's portfolio.
   - If 'chatgpt_portfolio_update.csv' is empty (meaning no past trading days logged), you will required to enter your starting cash.
   - From here, you can set up your portfolio or make any changes.
   - The script asks if you want to record manual buys or sells.
   - After you hit 'Enter' all calculations for the day are made.
   - Results are saved to `chatgpt_portfolio_update.csv` and any trades are added to `chatgpt_trade_log.csv`.
   - In the terminal, daily results are printed and automatically copied to your clipboard (see below). Paste them into the LLM.
To automate prompts, check out the [Automation Guide](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Other/AUTOMATION_README.md)

### Clipboard

The daily report (not the trade prompts) is copied to the clipboard at the end of the run. Pass `--no-copy` to disable. No extra Python packages are needed.

| Where you run it | How it copies | Setup |
|------------------|---------------|-------|
| macOS | `pbcopy` | None |
| Windows | `clip` | None |
| Linux (Wayland) | `wl-copy` | `sudo apt install wl-clipboard` |
| Linux (X11) | `xclip` or `xsel` | `sudo apt install xclip` |
| SSH, terminal supports OSC 52 (iTerm2, kitty, WezTerm, Windows Terminal, Ghostty) | Terminal escape sequence | None. Inside tmux: `set -g set-clipboard on` |
| SSH from MobaXterm | `xclip` over forwarded X11 | Install `xclip` on the server and enable X11-Forwarding for the session. `echo $DISPLAY` should print a value. |

If the script says the OSC 52 copy is unconfirmed and your paste is empty, your terminal doesn't support it (MobaXterm doesn't). Use the X11 route above or select the text with the mouse.
## Generate_Graph.py

This script draws a graph of your portfolio versus the S&P 500.

**Program will ALWAYS use 'Start Your Own/chatgpt_portfolio_update.csv' for data.**

1. **Ensure you have portfolio data**
   - Run `ProcessPortfolio.py` at least once so `chatgpt_portfolio_update.csv` has data.

2. **Run the graph script**
   ```bash
   python "Start Your Own/Generate_Graph.py" --start-equity 100
   ```
   
3. **View the chart**
   - A window opens showing your portfolio value vs. S&P 500. Results will be adjusted for baseline equity.

All of this is still VERY NEW, so there are bugs. Please reach out if you find an issue or have a question.

Both scripts are designed for beginners, feel free to experiment and modify them as you learn.
