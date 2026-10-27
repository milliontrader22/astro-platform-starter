"""Monday-Friday filter vs all days, on the chosen variant and on high-frequency variants."""
import pickle
import pandas as pd
import fm_backtest as fb
from fm_engine import load
from run_sweep import params

df = load("data/BTCUSDT-5m.csv"); A = fb.arrays(df); S = pickle.load(open("data/setups.pkl", "rb"))
days = len(df) / 288
year = df["dt"].dt.year.to_numpy()
cut = int((df["dt"] < pd.Timestamp("2025-01-01", tz="UTC")).sum())
BASE = dict(entry="tspot_deep", max_stage=3, tp_mode="dev", tp=3.0, exit_invalid=False, min_stop=0.005,
            bias="none", conf="smt", side="both", sess="lon+ny")
VARIANTS = {
    "BASE (best robust)": BASE,
    "BASE + daily open": {**BASE, "bias": "do"},
    "BASE min stop 0.4%": {**BASE, "min_stop": 0.004},
    "BASE all hours": {**BASE, "sess": "all"},
    "BASE no SMT": {**BASE, "conf": "none"},
    "frequent: T-Spot C3-C4, SMT, no filters": dict(entry="tspot_deep", max_stage=4, tp_mode="dev", tp=2.0,
        exit_invalid=False, min_stop=0.0, bias="none", conf="smt", side="both", sess="all"),
    "frequent: CISD, all setups": dict(entry="cisd", max_stage=4, tp_mode="dev", tp=2.0, exit_invalid=False,
        min_stop=0.0, bias="none", conf="none", side="both", sess="all"),
}
for name, kw in VARIANTS.items():
    for wd in (False, True):
        p = params(kw); p = fb.Params(**{**p.__dict__, "weekdays_only": wd})
        tr, st = fb.run(S, p, A)
        is_r = tr[tr.entry_bar < cut].r.mean(); oos_r = tr[tr.entry_bar >= cut].r.mean()
        yr = tr.groupby(year[tr.entry_bar]).r.sum().round(1).to_dict()
        print(f"{name:42s} {'L-V ' if wd else 'todo'} n={st['trades']:5d} ({st['trades']/days:.2f}/dia) win={st['winrate']:.2f} "
              f"avgR={st['avg_r']:+.3f} PF={st['pf']:.2f} DD={st['max_dd']:.2f} IS={is_r:+.3f} OOS={oos_r:+.3f} | {yr}")
