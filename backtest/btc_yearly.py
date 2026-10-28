"""Year-by-year results on BTCUSDT for the best candidate variants (1% risk per trade, compounding)."""
import numpy as np, pandas as pd
import fm_backtest as fb
import nested as N
from fm_engine import generate, load
from run_sweep import params

btc, eth = load("data/BTCUSDT-5m.csv"), load("data/ETHUSDT-5m.csv")
A = fb.arrays(btc); t = btc["time"].to_numpy()
S = N.cached("setups", lambda: generate(btc, 60, 6, eth))
BASE = dict(entry="tspot_deep", max_stage=3, tp_mode="dev", tp=3.0, exit_invalid=False, min_stop=0.005,
            bias="none", conf="smt", side="both", sess="lon+ny")
out = {}
for name, kw in {"1. Estrategia Pine (defaults)": BASE, "2. + filtro apertura diaria": {**BASE, "bias": "do"},
                 "3. stop minimo 0.4%": {**BASE, "min_stop": 0.004}}.items():
    tr, _ = fb.run(S, params(kw), A)
    out[name] = pd.DataFrame({"t": t[tr.entry_bar], "r": tr.r_eff.to_numpy()})

ctx = {}
for name, tf, htf in [("1H-1D", 60, 1440), ("15m-4H", 15, 240)]:
    b, e = N.resample(btc, tf), N.resample(eth, tf)
    Sx = N.cached(f"setups_{name}", lambda: generate(b, htf, 6, e))
    for stg in (3, 4):
        ctx[(name, stg, False)] = N.contexts(Sx, b["time"].to_numpy(), tf, stg, False)
N.G.update(ltf_S=S, A=A, t=t, ctx=ctx, order=sorted(range(len(S)), key=lambda i: S[i].create_bar), max_hold=288)
NV = {"4. Anidado 15m-4H + entrada 5m": dict(ctx="15m-4H", ctx_stage=3, ctx_smt=False, entry="tspot_deep", ltf_stage=4,
                                             ltf_smt=True, mode="htf", tp=2.0, min_stop=0.005, sess=()),
      "5. Anidado 1H-1D + entrada 5m": dict(ctx="1H-1D", ctx_stage=3, ctx_smt=False, entry="tspot_deep", ltf_stage=3,
                                            ltf_smt=False, mode="mix", tp=2.0, min_stop=0.005, sess=tuple(range(7, 20)))}
N.G["entries"] = dict(N.entries_for((v["entry"], v["ltf_stage"], v["ltf_smt"], v["sess"])) for v in NV.values())
# rerun the nested walk keeping per-trade rows
import types
for name, v in NV.items():
    rows = []
    orig = N.walk
    res = N.run_variant(v)
    # recompute rows: run_variant only returns aggregates, so replicate quickly
    ctxv = ctx[(v["ctx"], v["ctx_stage"], v["ctx_smt"])]; starts = {k: [c[0] for c in ctxv[k]] for k in ctxv}
    busy = -1
    O, H, L = A[0], A[1], A[2]
    for i, eb, ep in N.G["entries"][(v["entry"], v["ltf_stage"], v["ltf_smt"], v["sess"])]:
        if eb <= busy: continue
        s = S[i]; c = N.active(ctxv[s.is_bull], starts[s.is_bull], t[eb])
        if c is None: continue
        d = 1 if s.is_bull else -1; _, _, h_ext, h_cisd, h_rng = c
        sl = h_ext if v["mode"] == "htf" else s.extreme
        tp = s.cisd_level + (s.cisd_level - s.extreme) * v["tp"] if v["mode"] == "ltf" else h_cisd + d * h_rng * v["tp"]
        risk = (ep - sl) * d
        if risk <= 0 or risk / ep < v["min_stop"] or (tp - ep) * d <= risk * 0.5: continue
        xb, xp, why = N.walk(eb, ep, sl, tp, d, O, H, L, 288, True)
        fee_out = fb.FEE_MAKER if why == "tp" else fb.FEE_TAKER
        r = ((xp - ep) * d - fb.FEE_MAKER * ep - fee_out * xp) / risk
        rows.append((t[eb], r * min(1.0, 10 * (risk / ep) / 0.01))); busy = xb
    out[name] = pd.DataFrame(rows, columns=["t", "r"])

for name, df in out.items():
    df["year"] = pd.to_datetime(df.t, unit="ms").dt.year
    print(f"\n{name}")
    print(f"  {'Año':6s} {'Ops':>4s} {'Acierto':>8s} {'R/op':>7s} {'R total':>8s} {'Rent. 1%':>9s} {'DD máx':>7s}")
    for y, g in list(df.groupby("year")) + [("TOTAL", df)]:
        eq = np.cumprod(1 + 0.01 * g.r.to_numpy()); dd = (1 - eq / np.maximum.accumulate(eq)).max()
        print(f"  {str(y):6s} {len(g):4d} {np.mean(g.r > 0):8.0%} {g.r.mean():+7.2f} {g.r.sum():+8.1f} {eq[-1] - 1:+9.1%} {dd:7.1%}")
