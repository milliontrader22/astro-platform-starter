"""Intraday scalping search: weekdays only, New York-time windows, news filter, flat at window end.

Usage: python scalp.py 1m|5m
Stage 1 sweeps the model (context, entry, stop/target mode, target, window); stage 2 refines the best
by news filter, min stop and session exit. Ranked on 2022-24, checked on 2025-26 and by year.
News filter (no calendar needed): no entries 08:25-09:00 ET (CPI/NFP/PPI...) and 13:55-14:30 ET (FOMC),
optionally no trading at all on FOMC days and NFP Fridays.
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import fm_backtest as fb
import nested as N
from fm_engine import generate, load

MIN = 60_000
FOMC = """2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09""".split()
WINDOWS = {"ny_am": (570, 720), "ny": (570, 960), "ny_ext": (480, 960), "london": (120, 300), "lon_ny": (120, 720)}
TARGETS = [("r", 1.0), ("r", 1.5), ("r", 2.0), ("dev", 2.0)]
SPLIT = pd.Timestamp("2025-01-01", tz="UTC").value // 10**6
G = {}


def et_arrays(df):
    et = df["dt"].dt.tz_convert("America/New_York")
    mins = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    wd = et.dt.weekday.to_numpy()
    date = et.dt.strftime("%Y-%m-%d").to_numpy()
    day_id = pd.factorize(date)[0]
    fomc = np.isin(date, FOMC)
    nfp = (wd == 4) & (et.dt.day.to_numpy() <= 7)
    blackout = ((mins >= 505) & (mins < 540)) | ((mins >= 835) & (mins < 870))
    return dict(mins=mins, wd=wd, day=day_id, event=fomc | nfp, blackout=blackout)


def walk(eb, ep, sl, tp, d, end_min, sess_exit):
    O, H, L = G["A"][0], G["A"][1], G["A"][2]
    mins, day = G["et"]["mins"], G["et"]["day"]
    n = len(O)
    for j in range(eb, min(n, eb + G["max_hold"])):
        if sess_exit and j > eb and (day[j] != day[eb] or mins[j] >= end_min):
            return j, O[j], "end"
        o = O[j]
        if j > eb and (o - sl) * d <= 0:
            return j, o, "sl"
        if (L[j] <= sl) if d > 0 else (H[j] >= sl):
            return j, sl, "sl"
        if (H[j] >= tp) if d > 0 else (L[j] <= tp):
            return j, tp, "tp"
    j = min(n - 1, eb + G["max_hold"] - 1)
    return j, O[j], "time"


def entries_for(key):
    entry, stage, smt = key
    p = fb.Params(entry=entry, max_stage=stage, tp_mode="r", tp=1.0, exit_invalid=False, need_smt=smt)
    out = []
    for i in G["order"]:
        e = fb.sim_setup(G["S"][i], p, *G["A"])
        if e is not None:
            out.append((i, e["entry_bar"], e["entry"]))
    return key, out


def run_variant(v, keep=False):
    S, t, et = G["S"], G["t"], G["et"]
    w0, w1 = WINDOWS[v["window"]]
    ctx = G["ctx"].get((v["ctx"], 4)) if v["ctx"] != "none" else None
    starts = {k: [c[0] for c in ctx[k]] for k in ctx} if ctx else None
    kind, tpx = v["target"]
    rows, busy = [], -1
    for i, eb, ep in G["entries"][(v["entry"], v["stage"], v["smt"])]:
        if eb <= busy:
            continue
        m = et["mins"][eb]
        if et["wd"][eb] >= 5 or not (w0 <= m < w1):
            continue
        if v["news"] != "none" and et["blackout"][eb]:
            continue
        if v["news"] == "strict" and et["event"][eb]:
            continue
        s = S[i]
        d = 1 if s.is_bull else -1
        if ctx is not None:
            c = N.active(ctx[s.is_bull], starts[s.is_bull], t[eb])
            if c is None:
                continue
            _, _, h_ext, h_cisd, h_rng = c
        sl = h_ext if v["mode"] == "htf" else s.extreme
        risk = (ep - sl) * d
        if risk <= 0 or risk / ep < v["min_stop"]:
            continue
        if kind == "r":
            tp = ep + d * risk * tpx
        elif v["mode"] == "ltf":
            tp = s.cisd_level + (s.cisd_level - s.extreme) * tpx
        else:
            tp = h_cisd + d * h_rng * tpx
        if (tp - ep) * d <= risk * 0.5:
            continue
        xb, xp, why = walk(eb, ep, sl, tp, d, w1, v["sess_exit"])
        fee_in = fb.FEE_TAKER if v["entry"] == "cisd" else fb.FEE_MAKER
        fee_out = fb.FEE_MAKER if why == "tp" else fb.FEE_TAKER
        r = ((xp - ep) * d - fee_in * ep - fee_out * xp) / risk
        rows.append((t[eb], r, why, et["day"][eb]))
        busy = xb
    if len(rows) < 100:
        return None
    tr = pd.DataFrame(rows, columns=["t", "r", "why", "day"])
    if keep:
        return tr
    oos = tr.t >= SPLIT
    yr = pd.to_datetime(tr.t, unit="ms").dt.year
    ys = tr.groupby(yr).r.sum()
    return {**v, "target": f"{kind}{tpx}", "n": len(tr), "per_wday": len(tr) / G["wdays"],
            "days_traded": tr.day.nunique() / G["wdays"], "win": (tr.r > 0).mean(), "avg_r": tr.r.mean(),
            "is_r": tr[~oos].r.mean(), "oos_r": tr[oos].r.mean(), "pos_years": int((ys > 0).sum()),
            "r_per_wday": tr.r.sum() / G["wdays"], "years": ys.round(1).to_dict()}


def _init(g):
    G.update(g)


def setup_globals(ltf):
    btc5, eth5 = load("data/BTCUSDT-5m.csv"), load("data/ETHUSDT-5m.csv")
    if ltf == "5m":
        df, S, max_hold = btc5, N.cached("setups", lambda: generate(btc5, 60, 6, eth5)), 288
    else:
        df = load("data/BTCUSDT-1m.csv")
        S = N.cached("setups_1m-15m", lambda: generate(df, 15, 6, load("data/ETHUSDT-1m.csv")))
        max_hold = 1440
    ctx = {}
    for name, tf, htf in [("1H-1D", 60, 1440), ("15m-4H", 15, 240)]:
        b, e = N.resample(btc5, tf), N.resample(eth5, tf)
        Sx = N.cached(f"setups_{name}", lambda: generate(b, htf, 6, e))
        ctx[(name, 4)] = N.contexts(Sx, b["time"].to_numpy(), tf, 4, False)
    et = et_arrays(df)
    wdays = len(np.unique(et["day"][et["wd"] < 5]))
    return dict(S=S, A=fb.arrays(df), t=df["time"].to_numpy(), et=et, ctx=ctx, max_hold=max_hold, wdays=wdays,
                order=sorted(range(len(S)), key=lambda i: S[i].create_bar))


def pmap(fn, items):
    with Pool(os.cpu_count(), initializer=_init, initargs=(G,)) as pool:
        return pool.map(fn, items, chunksize=8)


def main():
    ltf = sys.argv[1] if len(sys.argv) > 1 else "5m"
    G.update(setup_globals(ltf))
    keys = list(itertools.product(["cisd", "tspot_deep", "tspot_edge"], [3, 4], [False, True]))
    G["entries"] = dict(pmap(entries_for, keys))
    print(f"{ltf}: {len(G['S'])} setups, {G['wdays']} weekdays", flush=True)

    ctx_modes = [("none", "ltf"), ("15m-4H", "ltf"), ("15m-4H", "htf"), ("15m-4H", "mix"),
                 ("1H-1D", "ltf"), ("1H-1D", "htf"), ("1H-1D", "mix")]
    s1 = [dict(ctx=c, mode=m, entry=e, stage=st, smt=sm, target=tg, window=w, news="blackout", min_stop=ms, sess_exit=True)
          for (c, m), (e, st, sm), tg, w, ms in itertools.product(ctx_modes, keys, TARGETS, WINDOWS, [0.0, 0.002])]
    R1 = pd.DataFrame([r for r in pmap(run_variant, s1) if r])
    R1["score"] = R1["is_r"] * np.sqrt(R1["n"])
    print(f"stage 1: {len(s1)} variants, {len(R1)} with >=100 trades", flush=True)

    best = R1[R1.per_wday >= 0.7].sort_values("score", ascending=False).head(40)
    s2 = []
    for _, r in best.iterrows():
        base = {k: r[k] for k in ["ctx", "mode", "entry", "stage", "smt", "window"]}
        base["target"] = next(tg for tg in TARGETS if f"{tg[0]}{tg[1]}" == r["target"])
        for news, ms, se in itertools.product(["none", "blackout", "strict"], [0.0, 0.001, 0.002, 0.003], [True, False]):
            s2.append({**base, "news": news, "min_stop": ms, "sess_exit": se})
    R2 = pd.DataFrame([r for r in pmap(run_variant, s2) if r])
    R = pd.concat([R1, R2], ignore_index=True).drop_duplicates(
        subset=["ctx", "mode", "entry", "stage", "smt", "target", "window", "news", "min_stop", "sess_exit"])
    R["score"] = R["is_r"] * np.sqrt(R["n"])
    R["robust"] = R[["is_r", "oos_r"]].min(axis=1)
    R.to_csv(f"scalp_{ltf}.csv", index=False)

    pd.set_option("display.width", 340); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)
    cols = ["ctx", "mode", "entry", "stage", "smt", "target", "window", "news", "min_stop", "sess_exit", "n", "per_wday",
            "days_traded", "win", "avg_r", "is_r", "oos_r", "pos_years", "r_per_wday", "years"]
    daily = R[R.per_wday >= 0.7]
    print(f"\n{len(R)} variants; {len(daily)} trade >= 0.7 per weekday; "
          f"of those positive IS and OOS: {((daily.is_r > 0) & (daily.oos_r > 0)).sum()}")
    print("\nDaily variants ranked by 2022-24 (selection), with 2025-26 check:")
    print(daily.sort_values("score", ascending=False)[cols].head(15).round(3).to_string())
    print("\nDaily variants ranked by min(IS, OOS):")
    print(daily.sort_values("robust", ascending=False)[cols].head(15).round(3).to_string())


if __name__ == "__main__":
    main()
