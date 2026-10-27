"""Year-by-year, parameter-sensitivity and fee stress test for the chosen variant."""
import pickle
import pandas as pd
import fm_backtest as fb
from fm_engine import load
from run_sweep import params

df = load("data/BTCUSDT-5m.csv"); arrs = fb.arrays(df)
S = pickle.load(open("data/setups.pkl", "rb"))
year = df["dt"].dt.year.to_numpy()
BASE = dict(entry="tspot_deep", max_stage=3, tp_mode="dev", tp=3.0, exit_invalid=False, min_stop=0.005,
            bias="none", conf="smt", side="both", sess="lon+ny")


def show(name, kw):
    tr, st = fb.run(S, params(kw), arrs)
    if not st:
        print(name, "no trades"); return
    yr = tr.groupby(year[tr.entry_bar])["r"].agg(["size", "mean", "sum"]).round(2)
    print(f"{name:28s} n={st['trades']:4d} win={st['winrate']:.2f} avgR={st['avg_r']:+.3f} PF={st['pf']:.2f} "
          f"DD={st['max_dd']:.2f} ret={st['ret']:+.2f} | by year R: " + " ".join(f"{y}:{r['sum']:+.1f}({int(r['size'])})" for y, r in yr.iterrows()))
    return tr


tr = show("BASE", BASE)
show("BASE + daily-open bias", {**BASE, "bias": "do"})
show("BASE + 4H EMA bias", {**BASE, "bias": "ema"})
for k, vals in dict(tp=[1.5, 2.0, 2.5, 4.0], min_stop=[0.003, 0.004, 0.006, 0.008], sess=["all", "ny", "london"],
                    max_stage=[4], conf=["none", "psp"], exit_invalid=[True], tp_mode=["r"], side=["long", "short"]).items():
    for v in vals:
        show(f"{k}={v}", {**BASE, k: v})
fb.FEE_TAKER, fb.FEE_MAKER = 0.001, 0.001
show("BASE with SPOT fees 0.1%", BASE)
fb.FEE_TAKER, fb.FEE_MAKER = 0.0005, 0.0002
tr.to_csv("best_trades.csv", index=False)
print(tr.reason.value_counts().to_dict(), "median stop %", round(tr.stop_pct.median() * 100, 2))
