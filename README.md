# Volume Profile Bot

This repository contains trading bot scripts for MetaTrader 5.

## Installation

Use pip to install the required Python packages:

```bash
pip install -r requirements.txt
```

## Contents

- `bot.py` - Main trading bot implementation.
- `range.py` - Example script for range-based analysis.

This repository contains a trading bot that relies on the MetaTrader5 (MT5) Python API.

## Environment Variables

Before running `bot.py`, set the following environment variables so the bot can
authenticate with your MT5 account:

- `MT5_LOGIN` – numeric login ID for the MT5 account
- `MT5_PASSWORD` – password associated with the login
- `MT5_SERVER` – name of the MT5 trading server

These variables are read at runtime and are required for initialization.

