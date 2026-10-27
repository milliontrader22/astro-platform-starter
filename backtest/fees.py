"""Every setup traded independently (concurrent positions allowed) under several fee levels."""
import pickle, sys
import numpy as np, pandas as pd
import fm_backtest as fb
from fm_engine import load

df = load("data/BTCUSDT-5m.csv"); A = fb.arrays(df); S = pickle.load(open("data/setups.pkl", "rb"))
days = len(df) / 288
yr = df["dt"].dt.year.to_numpy()
cut = int((df["dt"] < pd.Timestamp("2025-01-01", tz="UTC")).sum())
FEES = {"cero": (0, 0), "maker 0 / taker 0.02%": (0.0002, 0.0), "VIP 0.01/0.03": (0.0003, 0.0001),
        "estandar futuros 0.02/0.05": (0.0005, 0.0002), "spot 0.1": (0.001, 0.001)}
CONFIGS = {
    "T-Spot Deep C3-C4, -2 dev": fb.Params(entry="tspot_deep", max_stage=4, tp=2.0, exit_invalid=False),
    "T-Spot Deep C3-C6, -2 dev": fb.Params(entry="tspot_deep", max_stage=6, tp=2.0, exit_invalid=False),
    "T-Spot Edge C3-C4, -2 dev": fb.Params(entry="tspot_edge", max_stage=4, tp=2.0, exit_invalid=False),
    "CISD C3-C4, -2 dev": fb.Params(entry="cisd", max_stage=4, tp=2.0, exit_invalid=False),
    "T-Spot Deep C3-C4, objetivo C2": fb.Params(entry="tspot_deep", max_stage=4, tp_mode="c2", exit_invalid=False),
    "T-Spot Deep C3-C4, objetivo C1": fb.Params(entry="tspot_deep", max_stage=4, tp_mode="c1", exit_invalid=False),
    "T-Spot Deep C3-C4, -2 dev, BE 1R": fb.Params(entry="tspot_deep", max_stage=4, tp=2.0, exit_invalid=False, be_r=1.0),
}
for cname, p in CONFIGS.items():
    print(f"\n== {cname}")
    for fname, (tk, mk) in FEES.items():
        fb.FEE_TAKER, fb.FEE_MAKER = tk, mk
        r = [(s.create_bar, fb.sim_setup(s, p, *A)) for s in S]
        t = pd.DataFrame([x[1] for x in r if x[1]])
        is_r, oos_r = t[t.entry_bar < cut].r.mean(), t[t.entry_bar >= cut].r.mean()
        by = t.groupby(yr[t.entry_bar]).r.sum().round(0).astype(int).to_dict()
        print(f"  {fname:28s} n={len(t)} ({len(t)/days:.1f}/dia) win={np.mean(t.r>0):.2f} avgR={t.r.mean():+.3f} "
              f"IS={is_r:+.3f} OOS={oos_r:+.3f} R/dia={t.r.sum()/days:+.2f} | {by}")
fb.FEE_TAKER, fb.FEE_MAKER = 0.0005, 0.0002
