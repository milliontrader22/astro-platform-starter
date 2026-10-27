"""Download Binance spot klines from data.binance.vision (monthly zips) into one CSV.

Usage: python fetch_data.py BTCUSDT 5m 2022-01 2026-09 [out.csv]
Then top up the current month from api.binance.com if reachable.
"""
import io
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "qav", "trades", "tbbav", "tbqav", "ignore"]


def months(start, end):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print("retry", url, e)
            time.sleep(2 ** (k + 1))


def to_df(raw: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(zf.open(zf.namelist()[0]), header=None, names=COLS)
    if not str(df.iloc[0, 0]).isdigit():  # some files carry a header row
        df = df.iloc[1:].astype({"open_time": "int64"})
    ts = df["open_time"].astype("int64")
    ts = ts.where(ts < 10**14, ts // 1000)  # 2025+ files use microseconds
    return pd.DataFrame({"time": ts, "open": df["open"].astype(float), "high": df["high"].astype(float),
                         "low": df["low"].astype(float), "close": df["close"].astype(float),
                         "volume": df["volume"].astype(float)})


def recent(symbol, interval, start_ms):
    rows = []
    while True:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}"
               f"&startTime={start_ms}&limit=1000")
        import json
        data = json.loads(get(url))
        if not data:
            break
        rows += [[d[0], float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])] for d in data]
        start_ms = data[-1][0] + 1
        if len(data) < 1000:
            break
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])


def main():
    symbol, interval, start, end = sys.argv[1:5]
    out = Path(sys.argv[5] if len(sys.argv) > 5 else f"data/{symbol}-{interval}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for mo in months(start, end):
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{mo}.zip"
        try:
            parts.append(to_df(get(url)))
            print("ok", mo)
        except Exception as e:  # noqa: BLE001
            print("skip", mo, e)
    df = pd.concat(parts)
    # current month: daily archives (api.binance.com may be geo-blocked)
    last = pd.to_datetime(df["time"].max(), unit="ms") + pd.Timedelta(days=1)
    for day in pd.date_range(last.normalize(), pd.Timestamp.utcnow().tz_localize(None).normalize()):
        url = f"https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip"
        try:
            df = pd.concat([df, to_df(get(url, tries=1))])
        except Exception:  # noqa: BLE001
            pass
    try:
        df = pd.concat([df, recent(symbol, interval, int(df["time"].max()) + 1)])
    except Exception as e:  # noqa: BLE001
        print("no live top-up:", e)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df.to_csv(out, index=False)
    print(out, len(df), pd.to_datetime(df.time.min(), unit="ms"), "->", pd.to_datetime(df.time.max(), unit="ms"))


if __name__ == "__main__":
    main()
