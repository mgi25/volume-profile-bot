# Volume Profile Bot

This repository contains a MetaTrader 5 (MT5) trading bot along with a small
utility for displaying live trading ranges. The project is written in Python
and relies on the `MetaTrader5` package to interact with an MT5 terminal.

The main script **bot.py** implements an automated strategy using volume
profile statistics. Another script **range.py** plots live "LuxAlgo" ranges for
a selected symbol, which can help visualize market structure.

## Prerequisites

- Python 3.8 or later
- MetaTrader 5 terminal installed and running
- The following Python packages:
  - `MetaTrader5`
  - `pandas`
  - `numpy`
  - `statsmodels`
  - `ta`
  - `matplotlib` (for `range.py`)

You can install the required packages with:

```bash
pip install MetaTrader5 pandas numpy statsmodels ta matplotlib
```

## Configuration

`bot.py` expects valid MT5 account credentials. You can provide these in two
ways:

1. **Environment variables** – recommended for local development:
   - `MT5_LOGIN` – account number
   - `MT5_PASSWORD` – account password
   - `MT5_SERVER` – MT5 server name
2. **Hard coded values** – update the constants at the top of `bot.py` (not
   recommended for production).

The environment variables, if set, override the constants defined in the
script.

## Usage

After installing the prerequisites and configuring your credentials, run the
following commands from the repository root:

- **Start trading bot**
  ```bash
  python bot.py
  ```
  The bot connects to your MT5 terminal and begins executing the volume profile
  strategy.

- **Display live range plot**
  ```bash
  python range.py
  ```
  A Matplotlib window will open showing live ranges detected by the LuxAlgo
  logic.

Logs are written to `live_bot.log` and trade statistics are appended to
`ml_trading_log.csv`.

## Notes

This code is for educational purposes. Trading involves risk and no guarantee
of profitability is provided. Ensure that you understand and test the strategy
before using it with a live account.

