"""Two-stage parameter sweep with in-sample / out-of-sample split.

Usage: python run_sweep.py data/BTCUSDT-5m.csv data/ETHUSDT-5m.csv [split=2025-01-01]
Stage 1 sweeps the model options (both sides, all hours) on in-sample setups.
Stage 2 refines the best stage-1 variants by side and session.
The top variants are then re-run, unchanged, on the out-of-sample period.
"""
import itertools
import os
import pickle
import sys
from multiprocessing import Pool

import pandas as pd

from fm_backtest import Params, arrays, run
from fm_engine import generate, load

GRID = dict(
    entry=["cisd", "tspot_edge", "tspot_deep"],
    max_stage=[3, 4],
    tp_mode=["dev", "r"],
    tp=[1.0, 1.5, 2.0, 3.0],
    exit_invalid=[True, False],
    min_stop=[0.0, 0.003, 0.005, 0.008],
    bias=["none", "ema", "do"],
    conf=["none", "smt", "psp"],
)
SESSIONS = {"all": (), "asia": tuple(range(0, 7)), "london": tuple(range(7, 12)),
            "ny": tuple(range(12, 20)), "lon+ny": tuple(range(7, 20))}
MIN_TRADES = 100

_G = {}


def params(kw):
    kw = dict(kw)
    conf = kw.pop("conf", "none")
    sess = kw.pop("sess", "all")
    return Params(**kw, need_smt=conf == "smt", need_psp=conf == "psp", sess=SESSIONS[sess])


def _work(kw):
    _, st = run(_G["setups"], params(kw), _G["arrs"])
    return {**kw, **st} if st else None


def _init(setups, arrs):
    _G["setups"], _G["arrs"] = setups, arrs


def pmap(setups, arrs, combos):
    with Pool(os.cpu_count(), initializer=_init, initargs=(setups, arrs)) as pool:
        return pd.DataFrame([r for r in pool.map(_work, combos, chunksize=8) if r])


def score(df):
    df = df[df["trades"] >= MIN_TRADES].copy()
    df["score"] = df["avg_r"] * df["trades"] ** 0.5   # t-stat-like: edge that holds over many trades
    return df.sort_values("score", ascending=False)


def main():
    df = load(sys.argv[1])
    pair = load(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].endswith(".csv") else None
    split = pd.Timestamp(next((a.split("=")[1] for a in sys.argv if a.startswith("split=")), "2025-01-01"), tz="UTC")
    cache = "data/setups.pkl"
    setups = pickle.load(open(cache, "rb")) if os.path.exists(cache) else generate(df, 60, 6, pair)
    arrs = arrays(df)
    cut = int((df["dt"] < split).sum())
    is_set = [s for s in setups if s.create_bar < cut]
    oos_set = [s for s in setups if s.create_bar >= cut]
    print(f"{len(setups)} setups | IS {len(is_set)} | OOS {len(oos_set)} | split {split.date()}", flush=True)

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    s1 = score(pmap(is_set, arrs, combos))
    s1.to_csv("sweep_stage1.csv", index=False)
    print("stage 1:", len(combos), "variants", flush=True)

    combos2 = [{**{k: r[k] for k in GRID}, "side": side, "sess": sess}
               for _, r in s1.head(40).iterrows() for side in ("both", "long", "short") for sess in SESSIONS]
    s2 = score(pmap(is_set, arrs, combos2))
    s2.to_csv("sweep_stage2.csv", index=False)

    keys = list(GRID) + ["side", "sess"]
    top = s2.drop_duplicates(keys).head(30).reset_index(drop=True)
    oos = pmap(oos_set, arrs, [{k: r[k] for k in keys} for _, r in top.iterrows()])
    out = top.merge(oos, on=keys, suffixes=("", "_oos"))
    out.to_csv("sweep_top.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
