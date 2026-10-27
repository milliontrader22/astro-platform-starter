"""Same rules on several coins, no per-coin tuning. SMT pair: BTC for alts, ETH for BTC."""
import os, pickle, sys
import numpy as np, pandas as pd
import fm_backtest as fb
from fm_engine import generate, load
from run_sweep import params

SYMS = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"]
BASE = dict(entry="tspot_deep", max_stage=3, tp_mode="dev", tp=3.0, exit_invalid=False, min_stop=0.005,
            bias="none", conf="smt", side="both", sess="lon+ny")
VARIANTS = {"BASE": BASE, "BASE+trend": {**BASE, "bias": "ema"}, "BASE stop0.4": {**BASE, "min_stop": 0.004},
            "BASE C3-C4": {**BASE, "max_stage": 4}}
btc = load("data/BTCUSDT-5m.csv")
eth = load("data/ETHUSDT-5m.csv")
split = pd.Timestamp("2025-01-01", tz="UTC")
all_trades = {v: [] for v in VARIANTS}
for sym in SYMS:
    path = f"data/{sym}-5m.csv"
    if not os.path.exists(path):
        print("missing", sym); continue
    df = btc if sym == "BTCUSDT" else load(path)
    cache = f"data/setups_{sym}.pkl" if sym != "BTCUSDT" else "data/setups.pkl"
    if os.path.exists(cache):
        S = pickle.load(open(cache, "rb"))
    else:
        S = generate(df, 60, 6, eth if sym == "BTCUSDT" else btc)
        pickle.dump(S, open(cache, "wb"))
    A = fb.arrays(df); days = len(df) / 288
    cut = int((df["dt"] < split).sum())
    line = f"{sym:9s}"
    for v, kw in VARIANTS.items():
        tr, st = fb.run(S, params(kw), A)
        if not st:
            line += f" | {v}: -"; continue
        tr["sym"] = sym; tr["t"] = df["time"].to_numpy()[tr.entry_bar]
        all_trades[v].append(tr)
        is_r, oos_r = tr[tr.entry_bar < cut].r.mean(), tr[tr.entry_bar >= cut].r.mean()
        line += f" | {v}: n={st['trades']} {st['avg_r']:+.2f}R (IS {is_r:+.2f} / OOS {oos_r:+.2f})"
    print(line, flush=True)

print("\nPortfolio (all coins together, 1 position per coin):")
for v, lst in all_trades.items():
    if not lst:
        continue
    t = pd.concat(lst).sort_values("t")
    days = (t.t.max() - t.t.min()) / 86_400_000
    oos = t.t >= split.value // 10**6
    yr = pd.to_datetime(t.t, unit="ms").dt.year
    eq = np.cumprod(1 + 0.005 * t.r.clip(lower=-1.5))   # 0.5% risk per trade
    dd = (1 - eq / np.maximum.accumulate(eq)).max()
    print(f"  {v:13s} n={len(t)} ({len(t)/days:.2f}/dia) win={np.mean(t.r>0):.2f} avgR={t.r.mean():+.3f} "
          f"IS={t[~oos].r.mean():+.3f} OOS={t[oos].r.mean():+.3f} DD@0.5%={dd:.2f} ret@0.5%={eq.iloc[-1]-1:+.2f} | "
          + str(t.groupby(yr).r.sum().round(1).to_dict()))
