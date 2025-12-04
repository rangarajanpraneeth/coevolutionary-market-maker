#!/usr/bin/env python3

import argparse
import os
import yfinance as yf


def normalize_period(p):
    p = p.lower().strip()
    if p == "max":
        return "max"
    if p[-1] not in {"d", "w", "m", "y"}:
        raise ValueError("Period must end with d, w, m, or y (ex: 6m, 3w, 1y).")
    num = p[:-1]
    unit = p[-1]
    if not num.isdigit():
        raise ValueError("Period must start with a number (ex: 6m, 3w, 1y).")
    mapping = {"d": "d", "w": "wk", "m": "mo", "y": "y"}
    return num + mapping[unit]


def fetch_ohlcv(
    symbol, filename=None, period="6m", interval="1d", start=None, end=None
):
    symbol = symbol.upper()
    print(f"\n=== Fetching OHLCV for {symbol} ===")
    print(f"Interval: {interval}")

    if start and end:
        print(f"Date Range: {start} → {end}")
        df = yf.download(symbol, start=start, end=end, interval=interval)
    else:
        yf_period = normalize_period(period)
        print(f"Period: {yf_period}")
        df = yf.download(symbol, period=yf_period, interval=interval)

    if df.empty:
        raise ValueError(
            f"No data returned for {symbol}. Check ticker, period, interval, or date range."
        )

    df.reset_index(inplace=True)

    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Downloaded data missing '{col}' column — got {df.columns.tolist()}"
            )

    if filename is None:
        filename = symbol.lower()
    if filename.lower().endswith(".csv"):
        filename = filename[:-4]

    out_dir = os.path.join("data", symbol)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{filename}.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved → {out_path} ({len(df)} rows)\n")
    return out_path, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV into data/<symbol>/")
    parser.add_argument(
        "-s", "--symbol", required=True, help="Ticker (ex: NVDA, BTC-USD)"
    )
    parser.add_argument("-n", "--name", help="Output filename (no .csv)")
    parser.add_argument(
        "-p", "--period", default="6m", help="Lookback (ex: 5d, 3w, 6m, 1y, max)"
    )
    parser.add_argument(
        "-i", "--interval", default="1d", help="Candle size (ex: 1m, 5m, 15m, 1h, 1d)"
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    fetch_ohlcv(
        symbol=args.symbol,
        filename=args.name,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )
