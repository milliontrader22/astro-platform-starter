"""Parameter sweep with in-sample / out-of-sample split.

Usage: python run_sweep.py data/BTCUSDT-5m.csv [data/ETHUSDT-5m.csv] [split=2025-01-01]
Ranks variants on in-sample, then reports the same variants out-of-sample.
"""
import sys

import pandas as pd

from fm_backtest import Params, arrays, run, sweep
from fm_engine import generate, load

GRID = dict(
    entry=["cisd", "tspot_edge", "tspot_deep"],
    max_stage=[3, 4],
    tp_mode=["dev", "r"],
    tp=[1.0, 1.5, 2.0, 2.5, 3.0],
    sl_buf=[0.0, 0.001],
    exit_invalid=[True, False],
    side=["both", "long", "short"],
)
SESSIONS = {"all": (), "asia": tuple(range(0, 7)), "london": tuple(range(7, 12)),
            "ny": tuple(range(12, 20)), "lon+ny": tuple(range(7, 20))}


def main():
    df = load(sys.argv[1])
    pair = load(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].endswith(".csv") else None
    split = pd.Timestamp(next((a.split("=")[1] for a in sys.argv if a.startswith("split=")), "2025-01-01"), tz="UTC")
    setups = generate(df, 60, 6, pair)
    arrs = arrays(df)
    cut = int((df["dt"] < split).sum())
    is_set = [s for s in setups if s.create_bar < cut]
    oos_set = [s for s in setups if s.create_bar >= cut]
    print(f"{len(setups)} setups | IS {len(is_set)} | OOS {len(oos_set)} | split {split.date()}")

    res = []
    for name, hrs in SESSIONS.items():
        r = sweep(is_set, arrs, GRID, Params(sess=hrs))
        r["sess"] = name
        res.append(r)
    filters = [dict(need_smt=True), dict(need_psp=True)] if pair is not None else []
    for f in filters:
        r = sweep(is_set, arrs, GRID, Params(**f))
        r["sess"] = "all+" + "+".join(f)
        res.append(r)
    res = pd.concat(res, ignore_index=True)
    res["score"] = res["avg_r"] * res["trades"] ** 0.5   # favours edge that holds over many trades
    top = res.sort_values("score", ascending=False).head(25).copy()

    oos = []
    for _, row in top.iterrows():
        kw = {k: row[k] for k in GRID}
        kw["sess"] = SESSIONS.get(row["sess"], ())
        if "need_smt" in row["sess"]:
            kw["need_smt"] = True
        if "need_psp" in row["sess"]:
            kw["need_psp"] = True
        _, st = run(oos_set, Params(**kw), arrs)
        oos.append({f"oos_{k}": v for k, v in st.items()})
    out = pd.concat([top.reset_index(drop=True), pd.DataFrame(oos)], axis=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(out.round(3).to_string())
    res.to_csv("sweep_results.csv", index=False)
    out.to_csv("sweep_top.csv", index=False)


if __name__ == "__main__":
    main()
