"""Nested fractals: HTF setup (1H-1D / 15m-4H) gives direction, LTF setup (5m-1H / 1m-15m) gives the entry.

Usage: python nested.py [5m|1m]
Context = an HTF setup whose CISD has confirmed (known at that bar's close) and that is still alive
(no XC2, no XC on its current candle, still within ctx_stage). Only LTF setups in the same direction
are traded, one position at a time. Stop/target modes:
  ltf : LTF C2 extreme / LTF CISD deviation
  htf : HTF C2 extreme / HTF CISD deviation
  mix : LTF C2 extreme / HTF CISD deviation
"""
import bisect
import itertools
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import fm_backtest as fb
from fm_engine import generate, load

MIN = 60_000
SPLIT = pd.Timestamp("2025-01-01", tz="UTC").value // 10**6


def resample(df, minutes):
    g = df["time"] // (minutes * MIN)
    out = df.groupby(g).agg(time=("time", "first"), open=("open", "first"), high=("high", "max"),
                            low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).reset_index(drop=True)
    out["dt"] = pd.to_datetime(out["time"], unit="ms", utc=True)
    return out


def cached(name, fn):
    path = f"data/{name}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    obj = fn()
    pickle.dump(obj, open(path, "wb"))
    return obj


def contexts(S, t, tf_min, max_stage, need_smt):
    """Active windows [start_ms, end_ms) per HTF setup, split by direction."""
    out = {True: [], False: []}
    for s in S:
        if s.confirm_bar < 0 or np.isnan(s.cisd_level) or (need_smt and not s.smt):
            continue
        start = t[max(s.confirm_bar, s.create_bar - 1)] + tf_min * MIN   # known at that bar's close
        ends = [s.end_bar] + list(s.invalid.values())
        if max_stage + 1 in s.stages:
            ends.append(s.stages[max_stage + 1][0])
        end_bar = min(ends)
        if end_bar >= len(t):
            end_bar = len(t) - 1
        end = t[end_bar]
        if end > start:
            d = 1 if s.is_bull else -1
            out[s.is_bull].append((start, end, s.extreme, s.cisd_level, (s.cisd_level - s.extreme) * d))
    for k in out:
        out[k].sort()
    return out


def active(ctx_list, starts, ts):
    """Most recent context window containing ts (or None)."""
    i = bisect.bisect_right(starts, ts) - 1
    while i >= 0:
        st, en, *rest = ctx_list[i]
        if en > ts:
            return ctx_list[i]
        i -= 1
        if ts - st > 7 * 24 * 60 * MIN:
            break
    return None


def walk(entry_bar, entry_px, sl, tp, d, O, H, L, max_hold, maker_in):
    n = len(O)
    for j in range(entry_bar, min(n, entry_bar + max_hold)):
        o = O[j]
        if j > entry_bar and (o - sl) * d <= 0:
            return j, o, "sl"
        if (L[j] <= sl) if d > 0 else (H[j] >= sl):
            return j, sl, "sl"
        if (H[j] >= tp) if d > 0 else (L[j] <= tp):
            return j, tp, "tp"
    j = min(n - 1, entry_bar + max_hold - 1)
    return j, O[j], "time"


G = {}


def _init(g):
    G.update(g)


def entries_for(key):
    entry, stage, smt, sess = key
    p = fb.Params(entry=entry, max_stage=stage, tp_mode="r", tp=1.0, exit_invalid=False, need_smt=smt, sess=sess)
    out = []
    for i in G["order"]:
        s = G["ltf_S"][i]
        e = fb.sim_setup(s, p, *G["A"])
        if e is not None:
            out.append((i, e["entry_bar"], e["entry"]))
    return key, out


def run_variant(v):
    ltf_S, A, t = G["ltf_S"], G["A"], G["t"]
    O, H, L = A[0], A[1], A[2]
    ctx = G["ctx"][(v["ctx"], v["ctx_stage"], v["ctx_smt"])]
    starts = {k: [c[0] for c in ctx[k]] for k in ctx}
    rows, busy = [], -1
    for i, eb, ep in G["entries"][(v["entry"], v["ltf_stage"], v["ltf_smt"], v["sess"])]:
        if eb <= busy:
            continue
        s = ltf_S[i]
        c = active(ctx[s.is_bull], starts[s.is_bull], t[eb])
        if c is None:
            continue
        d = 1 if s.is_bull else -1
        _, _, h_ext, h_cisd, h_rng = c
        sl = h_ext if v["mode"] == "htf" else s.extreme
        if v["mode"] == "ltf":
            tp = s.cisd_level + (s.cisd_level - s.extreme) * v["tp"]
        else:
            tp = h_cisd + d * h_rng * v["tp"]
        risk = (ep - sl) * d
        if risk <= 0 or risk / ep < v["min_stop"] or (tp - ep) * d <= risk * 0.5:
            continue
        xb, xp, why = walk(eb, ep, sl, tp, d, O, H, L, G["max_hold"], v["entry"] != "cisd")
        fee_in = fb.FEE_TAKER if v["entry"] == "cisd" else fb.FEE_MAKER
        fee_out = fb.FEE_MAKER if why == "tp" else fb.FEE_TAKER
        r = ((xp - ep) * d - fee_in * ep - fee_out * xp) / risk
        rows.append((t[eb], r, why))
        busy = xb
    if len(rows) < 30:
        return None
    tr = pd.DataFrame(rows, columns=["t", "r", "why"])
    oos = tr.t >= SPLIT
    days = (t[-1] - t[0]) / (1440 * MIN)
    return {**{k: (str(x) if k == "sess" else x) for k, x in v.items()}, "n": len(tr), "per_day": len(tr) / days,
            "win": (tr.r > 0).mean(), "avg_r": tr.r.mean(), "is_r": tr[~oos].r.mean(), "oos_r": tr[oos].r.mean(),
            "is_n": int((~oos).sum()), "oos_n": int(oos.sum()),
            "years": tr.groupby(pd.to_datetime(tr.t, unit="ms").dt.year).r.sum().round(1).to_dict()}


def main():
    ltf = sys.argv[1] if len(sys.argv) > 1 else "5m"
    btc5, eth5 = load("data/BTCUSDT-5m.csv"), load("data/ETHUSDT-5m.csv")
    ctx_raw = {}
    for name, tf, htf in [("1H-1D", 60, 1440), ("15m-4H", 15, 240)]:
        b, e = resample(btc5, tf), resample(eth5, tf)
        S = cached(f"setups_{name}", lambda: generate(b, htf, 6, e))
        ctx_raw[name] = (S, b["time"].to_numpy(), tf)
    if ltf == "5m":
        df, ltf_S = btc5, cached("setups", lambda: generate(btc5, 60, 6, eth5))
        max_hold = 288
    else:
        df = load("data/BTCUSDT-1m.csv")
        ltf_S = cached("setups_1m-15m", lambda: generate(df, 15, 6, load("data/ETHUSDT-1m.csv")))
        max_hold = 600
    A = fb.arrays(df)
    t = df["time"].to_numpy()
    ctx = {}
    for name, (S, ht, tf) in ctx_raw.items():
        for stg, smt in itertools.product([3, 4], [False, True]):
            ctx[(name, stg, smt)] = contexts(S, ht, tf, stg, smt)
    order = sorted(range(len(ltf_S)), key=lambda i: ltf_S[i].create_bar)
    print(f"LTF {ltf}: {len(ltf_S)} setups; contexts:",
          {k: (len(v[True]), len(v[False])) for k, v in ctx.items() if k[1] == 4 and not k[2]}, flush=True)

    grid = dict(ctx=["1H-1D", "15m-4H"], ctx_stage=[3, 4], ctx_smt=[False, True], entry=["cisd", "tspot_deep"],
                ltf_stage=[3, 4], ltf_smt=[False, True], mode=["ltf", "htf", "mix"], tp=[1.0, 2.0],
                min_stop=[0.0, 0.003, 0.005], sess=[(), tuple(range(7, 20))])
    combos = [dict(zip(grid, x)) for x in itertools.product(*grid.values())]
    g = dict(ltf_S=ltf_S, A=A, t=t, ctx=ctx, order=order, max_hold=max_hold)
    keys = list(itertools.product(grid["entry"], grid["ltf_stage"], grid["ltf_smt"], grid["sess"]))
    with Pool(os.cpu_count(), initializer=_init, initargs=(g,)) as pool:
        g["entries"] = dict(pool.map(entries_for, keys))
    print("entries ready", flush=True)
    with Pool(os.cpu_count(), initializer=_init, initargs=(g,)) as pool:
        res = [r for r in pool.map(run_variant, combos, chunksize=16) if r]
    R = pd.DataFrame(res)
    R.to_csv(f"nested_{ltf}.csv", index=False)
    R["robust"] = R[["is_r", "oos_r"]].min(axis=1)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 90)
    cols = ["ctx", "ctx_stage", "ctx_smt", "entry", "ltf_stage", "ltf_smt", "mode", "tp", "min_stop", "sess",
            "n", "per_day", "win", "avg_r", "is_r", "oos_r", "years"]
    print(f"\n{len(R)} variants. Top by min(IS, OOS) avgR, >= 1 trade/day:")
    print(R[R.per_day >= 1].sort_values("robust", ascending=False)[cols].head(15).round(3).to_string())
    print("\nTop by min(IS, OOS) avgR, any frequency (n >= 100):")
    print(R[R.n >= 100].sort_values("robust", ascending=False)[cols].head(15).round(3).to_string())
    print("\nShare of variants positive in both IS and OOS:", round(((R.is_r > 0) & (R.oos_r > 0)).mean(), 3))


if __name__ == "__main__":
    main()
