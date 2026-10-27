"""Out-of-sample check of a whole family of variants (not one cherry-picked combo)."""
import itertools, pickle, sys
import pandas as pd
from fm_engine import load
from fm_backtest import arrays, run
from run_sweep import params, pmap

df = load("data/BTCUSDT-5m.csv"); arrs = arrays(df)
S = pickle.load(open("data/setups.pkl", "rb"))
cut = int((df["dt"] < pd.Timestamp("2025-01-01", tz="UTC")).sum())
IS = [s for s in S if s.create_bar < cut]; OOS = [s for s in S if s.create_bar >= cut]
fam = dict(entry=["cisd", "tspot_edge", "tspot_deep"], max_stage=[3, 4], tp_mode=["dev", "r"], tp=[2.0, 3.0],
           exit_invalid=[False], min_stop=[0.005], bias=["do", "ema", "none"], conf=["smt", "none"],
           side=["both", "short", "long"], sess=["lon+ny", "all"])
combos = [dict(zip(fam, v)) for v in itertools.product(*fam.values())]
a = pmap(IS, arrs, combos); b = pmap(OOS, arrs, combos)
keys = list(fam)
m = a.merge(b, on=keys, suffixes=("_is", "_oos"))
for k in ["bias", "conf", "side", "sess", "entry", "tp_mode"]:
    g = m.groupby(k).agg(n=("avg_r_is", "size"), is_r=("avg_r_is", "mean"), oos_r=("avg_r_oos", "mean"),
                         oos_pos=("avg_r_oos", lambda x: (x > 0).mean()), oos_trades=("trades_oos", "mean"))
    print(g.round(3), "\n")
sub = m[(m.bias == "do") & (m.conf == "smt")]
print("do+smt family: OOS avgR mean", round(sub.avg_r_oos.mean(), 3), "share positive", round((sub.avg_r_oos > 0).mean(), 2), "n", len(sub))
m.to_csv("family.csv", index=False)
