#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import mplfinance as mpf


def plot_ohlcv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.set_index("Date").sort_index()

    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in ohlcv_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    try:
        freq = pd.infer_freq(df.index)
        if freq:
            df = df.asfreq(freq, method="ffill")
    except Exception:
        pass

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    add_plots = [
        mpf.make_addplot(df["EMA20"], color="dodgerblue", width=1),
        mpf.make_addplot(df["EMA50"], color="orange", width=1),
    ]

    folder = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(folder, f"{base}_ohlcv.png")

    mpf.plot(
        df,
        type="candle",
        volume=True,
        addplot=add_plots,
        style="charles",
        title=base.upper(),
        ylabel="Price",
        ylabel_lower="Volume",
        savefig=out_path,
        tight_layout=True,
    )

    print(f"Saved chart → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Path to OHLCV CSV file")
    args = parser.parse_args()
    plot_ohlcv(args.file)
