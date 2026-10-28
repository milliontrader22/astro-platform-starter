"""Scale-in (2-3 entries, equal or progressive size) on the daily BTC scalping framework of scalp.py.

Usage: python ladder.py 1m|5m
Levels are fractions of the original risk r0 = |E1 - LTF C2 extreme| measured against the trade.
R is normalised by TOTAL risk: losing every filled entry at the final stop with all entries filled = -1R
(fees included), so results compare 1:1 with single-entry runs.
Conservative fills: in a bar, adverse fills and the stop are processed first; no target on a bar that
filled a new entry.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

import fm_backtest as fb
import nested as N
import scalp as SC

LADDERS = {  # (entry levels incl. E1 at 0, final stop)
    "L2a": ((0.0, 0.5), 1.0), "L2b": ((0.0, 1.0), 2.0),
    "L3a": ((0.0, 0.5, 1.0), 1.5), "L3b": ((0.0, 1.0, 1.5), 2.0), "L3c": ((0.0, 1.0, 2.0), 3.0),
    "L1": ((0.0,), 1.0),  # single entry baseline
}
SIZES = {"eq": (1, 1, 1), "mild": (1, 1.5, 2), "mart": (1, 2, 4)}
TPS = [("e1", 0.5), ("e1", 1.0), ("e1", 1.5), ("avg", 0.5), ("avg", 1.0)]   # in r0 units


def walk_ladder(eb, e1, r0, d, ladder, sizes, tp_spec, end_min, sess_exit):
    O, H, L = SC.G["A"][0], SC.G["A"][1], SC.G["A"][2]
    mins, day = SC.G["et"]["mins"], SC.G["et"]["day"]
    levels, stop_k = LADDERS[ladder]
    w = np.array(SIZES[sizes][:len(levels)], float)
    px = e1 - d * r0 * np.array(levels)
    stop = e1 - d * r0 * stop_k
    total_risk = float(np.sum(w * np.abs(px - stop)))          # money lost if all filled and stopped
    filled = [0]
    kind, k = tp_spec
    fees = fb.FEE_MAKER * w[0] * px[0]

    def target():
        if kind == "e1":
            return e1 + d * r0 * k
        avg = np.sum(w[filled] * px[filled]) / np.sum(w[filled])
        return avg + d * r0 * k

    def pnl(exit_px, fee):
        q = w[filled]
        return float(np.sum(q * (exit_px - px[filled]) * d) - fees - fee * exit_px * np.sum(q))

    n = len(O)
    for j in range(eb, min(n, eb + SC.G["max_hold"])):
        if sess_exit and j > eb and (day[j] != day[eb] or mins[j] >= end_min):
            return j, pnl(O[j], fb.FEE_TAKER) / total_risk, "end", len(filled)
        new_fill = False
        for i in range(1, len(levels)):
            if i not in filled and ((L[j] <= px[i]) if d > 0 else (H[j] >= px[i])):
                filled.append(i); new_fill = True
                fees += fb.FEE_MAKER * w[i] * px[i]
        o = O[j]
        if j > eb and (o - stop) * d <= 0:
            return j, pnl(o, fb.FEE_TAKER) / total_risk, "sl", len(filled)
        if (L[j] <= stop) if d > 0 else (H[j] >= stop):
            return j, pnl(stop, fb.FEE_TAKER) / total_risk, "sl", len(filled)
        tp = target()
        if not new_fill and ((H[j] >= tp) if d > 0 else (L[j] <= tp)):
            return j, pnl(tp, fb.FEE_MAKER) / total_risk, "tp", len(filled)
    j = min(n - 1, eb + SC.G["max_hold"] - 1)
    return j, pnl(O[j], fb.FEE_TAKER) / total_risk, "time", len(filled)


def run(v, keep=False):
    S, t, et = SC.G["S"], SC.G["t"], SC.G["et"]
    w0, w1 = SC.WINDOWS[v["window"]]
    ctx = SC.G["ctx"].get((v["ctx"], 4)) if v["ctx"] != "none" else None
    starts = {k: [c[0] for c in ctx[k]] for k in ctx} if ctx else None
    rows, busy = [], -1
    for i, eb, ep in SC.G["entries"][(v["entry"], v["stage"], False)]:
        if eb <= busy:
            continue
        m = et["mins"][eb]
        if et["wd"][eb] >= 5 or not (w0 <= m < w1) or et["blackout"][eb]:
            continue
        s = S[i]
        d = 1 if s.is_bull else -1
        if ctx is not None and N.active(ctx[s.is_bull], starts[s.is_bull], t[eb]) is None:
            continue
        r0 = (ep - s.extreme) * d
        if r0 <= 0 or r0 / ep < v["min_stop"]:
            continue
        xb, r, why, nf = walk_ladder(eb, ep, r0, d, v["ladder"], v["sizes"], v["tp"], w1, v["sess_exit"])
        rows.append((t[eb], r, why, nf, et["day"][eb]))
        busy = xb
    if len(rows) < 100:
        return None
    tr = pd.DataFrame(rows, columns=["t", "r", "why", "fills", "day"])
    if keep:
        return tr
    oos = tr.t >= SC.SPLIT
    ys = tr.groupby(pd.to_datetime(tr.t, unit="ms").dt.year).r.sum()
    eq = np.cumprod(1 + 0.01 * tr.r.to_numpy())
    return {**v, "tp": f"{v['tp'][0]}{v['tp'][1]}", "n": len(tr), "per_wday": len(tr) / SC.G["wdays"],
            "win": (tr.r > 0).mean(), "avg_r": tr.r.mean(), "is_r": tr[~oos].r.mean(), "oos_r": tr[oos].r.mean(),
            "full_fill": (tr.fills == len(LADDERS[v["ladder"]][0])).mean(), "worst": tr.r.min(),
            "dd1": (1 - eq / np.maximum.accumulate(eq)).max(), "pos_years": int((ys > 0).sum()),
            "years": ys.round(1).to_dict()}


def main():
    ltf = sys.argv[1] if len(sys.argv) > 1 else "5m"
    SC.G.update(SC.setup_globals(ltf))
    keys = list(itertools.product(["cisd", "tspot_deep", "tspot_edge"], [3, 4], [False]))
    SC.G["entries"] = dict(SC.pmap(SC.entries_for, keys))
    print(f"{ltf}: {len(SC.G['S'])} setups, {SC.G['wdays']} weekdays", flush=True)
    grid = dict(ctx=["none", "15m-4H"], entry=["cisd", "tspot_deep", "tspot_edge"], stage=[3, 4],
                window=["ny", "ny_am", "lon_ny"], ladder=list(LADDERS), sizes=list(SIZES), tp=TPS,
                min_stop=[0.0, 0.002], sess_exit=[True, False])
    combos = [dict(zip(grid, x)) for x in itertools.product(*grid.values())]
    combos = [c for c in combos if not (c["ladder"] == "L1" and c["sizes"] != "eq")]
    print("variants:", len(combos), flush=True)
    R = pd.DataFrame([r for r in SC.pmap(run, combos) if r])
    R["robust"] = R[["is_r", "oos_r"]].min(axis=1)
    R["score"] = R["is_r"] * np.sqrt(R["n"])
    R.to_csv(f"ladder_{ltf}.csv", index=False)
    pd.set_option("display.width", 340); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)
    cols = ["ctx", "entry", "stage", "window", "ladder", "sizes", "tp", "min_stop", "sess_exit", "n", "per_wday", "win",
            "avg_r", "is_r", "oos_r", "full_fill", "dd1", "pos_years", "years"]
    daily = R[R.per_wday >= 0.7]
    print(f"\n{len(R)} variants; daily (>=0.7/weekday): {len(daily)}; positive IS and OOS: "
          f"{((daily.is_r > 0) & (daily.oos_r > 0)).sum()}")
    print("\nBy ladder / sizes (daily variants, mean avgR):")
    print(daily.groupby(["ladder", "sizes"]).agg(n=("avg_r", "size"), win=("win", "mean"), avg_r=("avg_r", "mean"),
                                                 best=("robust", "max")).round(3).to_string())
    print("\nDaily, ranked on 2022-24 only:")
    print(daily.sort_values("score", ascending=False)[cols].head(12).round(3).to_string())
    print("\nDaily, ranked by min(IS, OOS):")
    print(daily.sort_values("robust", ascending=False)[cols].head(12).round(3).to_string())


if __name__ == "__main__":
    main()
