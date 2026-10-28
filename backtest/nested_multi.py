"""Validate fixed nested variants on other coins (no re-tuning). SMT pair: BTC for alts, ETH for BTC."""
import itertools, os, sys
import numpy as np, pandas as pd
import fm_backtest as fb
import nested as N
from fm_engine import generate, load

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"]
LON_NY = tuple(range(7, 20))
VARIANTS = {   # ids from nested_5m.csv on BTC
    "A 15m-4H C3 | 5m T-Spot C3-C4 SMT | htf -2": dict(ctx="15m-4H", ctx_stage=3, ctx_smt=False, entry="tspot_deep", ltf_stage=4, ltf_smt=True, mode="htf", tp=2.0, min_stop=0.005, sess=()),
    "B 15m-4H C3 | 5m T-Spot C3 SMT | htf -1":    dict(ctx="15m-4H", ctx_stage=3, ctx_smt=False, entry="tspot_deep", ltf_stage=3, ltf_smt=True, mode="htf", tp=1.0, min_stop=0.005, sess=()),
    "C 15m-4H C4 | 5m T-Spot C3-C4 SMT | htf -1": dict(ctx="15m-4H", ctx_stage=4, ctx_smt=False, entry="tspot_deep", ltf_stage=4, ltf_smt=True, mode="htf", tp=1.0, min_stop=0.005, sess=()),
    "D same as A, stop>=0.3%":                    dict(ctx="15m-4H", ctx_stage=3, ctx_smt=False, entry="tspot_deep", ltf_stage=4, ltf_smt=True, mode="htf", tp=2.0, min_stop=0.003, sess=()),
    "E 1H-1D C3 | 5m T-Spot C3 | mix -2 lon+ny":  dict(ctx="1H-1D", ctx_stage=3, ctx_smt=False, entry="tspot_deep", ltf_stage=3, ltf_smt=False, mode="mix", tp=2.0, min_stop=0.005, sess=LON_NY),
}
btc, eth = load("data/BTCUSDT-5m.csv"), load("data/ETHUSDT-5m.csv")
allt = {v: [] for v in VARIANTS}
for sym in SYMS:
    df = btc if sym == "BTCUSDT" else load(f"data/{sym}-5m.csv")
    pair = eth if sym == "BTCUSDT" else btc
    tag = "" if sym == "BTCUSDT" else f"_{sym}"
    ltf_S = N.cached(f"setups{tag}" if tag else "setups", lambda: generate(df, 60, 6, pair))
    ctx = {}
    for name, tf, htf in [("1H-1D", 60, 1440), ("15m-4H", 15, 240)]:
        b, e = N.resample(df, tf), N.resample(pair, tf)
        S = N.cached(f"setups_{name}{tag}", lambda: generate(b, htf, 6, e))
        for stg in (3, 4):
            ctx[(name, stg, False)] = N.contexts(S, b["time"].to_numpy(), tf, stg, False)
    A = fb.arrays(df); t = df["time"].to_numpy()
    order = sorted(range(len(ltf_S)), key=lambda i: ltf_S[i].create_bar)
    N.G.clear(); N.G.update(ltf_S=ltf_S, A=A, t=t, ctx=ctx, order=order, max_hold=288)
    keys = {(v["entry"], v["ltf_stage"], v["ltf_smt"], v["sess"]) for v in VARIANTS.values()}
    N.G["entries"] = dict(N.entries_for(k) for k in keys)
    line = f"{sym:9s}"
    for name, v in VARIANTS.items():
        r = N.run_variant(v)
        line += f" | {name[0]}: " + (f"n={r['n']} {r['avg_r']:+.2f} ({r['is_r']:+.2f}/{r['oos_r']:+.2f})" if r else "n<30")
        if r:
            allt[name].append(r)
    print(line, flush=True)
print("\nPer variant across coins (trade-weighted):")
for name, rs in allt.items():
    n = sum(r["n"] for r in rs); avg = sum(r["avg_r"] * r["n"] for r in rs) / max(1, n)
    pos = sum(r["avg_r"] > 0 for r in rs)
    print(f"  {name:46s} coins={len(rs)} positive={pos} trades={n} ({sum(r['per_day'] for r in rs):.2f}/day) avgR={avg:+.3f}")
