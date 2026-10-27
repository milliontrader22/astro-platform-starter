"""Which setup features predict the outcome? Each setup simulated alone, checked IS vs OOS."""
import pickle, sys
import numpy as np, pandas as pd
import fm_backtest as fb
from fm_engine import load

sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
df = load(f"data/{sym}-5m.csv"); O, H, L, C, hrs, ema, dopen, wd = fb.arrays(df)
S = pickle.load(open("data/setups.pkl", "rb"))
cut = int((df["dt"] < pd.Timestamp("2025-01-01", tz="UTC")).sum())
P = fb.Params(entry="tspot_deep", max_stage=4, tp_mode="dev", tp=2.0, exit_invalid=False)
rows = []
for s in S:
    t = fb.sim_setup(s, P, O, H, L, C, hrs, ema, dopen, wd)
    if t is None:
        continue
    fb.FEE_TAKER, fb.FEE_MAKER = 0.0, 0.0
    gross = fb.sim_setup(s, P, O, H, L, C, hrs, ema, dopen, wd)["r"]
    fb.FEE_TAKER, fb.FEE_MAKER = 0.0005, 0.0002
    c1, c2 = s.c1, s.c2
    rng = c2["h"] - c2["l"]; d = 1 if s.is_bull else -1
    risk = t["stop_pct"] * t["entry"]
    e = t["entry_bar"]
    rows.append(dict(
        oos=s.create_bar >= cut, r=t["r"], gross=gross, stop_pct=t["stop_pct"] * 100,
        sweep=((c1["l"] - c2["l"]) if s.is_bull else (c2["h"] - c1["h"])) / rng,
        close_loc=((c2["c"] - c2["l"]) if s.is_bull else (c2["h"] - c2["c"])) / rng,
        c2_dir=np.sign((c2["c"] - c2["o"]) * d), c1_dir=np.sign((c1["c"] - c1["o"]) * d),
        c2_vs_c1=rng / max(1e-9, c1["h"] - c1["l"]), vol_ratio=c2["v"] / max(1e-9, c1["v"]),
        smt=s.smt, psp=s.psp, early_cisd=s.confirm_stage == 2, hour=hrs[e], weekday=wd[e],
        trend=np.sign((C[e - 1] - ema[e]) * d), vs_dopen=np.sign((t["entry"] - dopen[e]) * d)))
X = pd.DataFrame(rows)
print(f"{sym}: {len(X)} trades, gross {X.gross.mean():+.3f}R, net {X.r.mean():+.3f}R\n")
pd.set_option("display.width", 200)
for f in ["stop_pct", "sweep", "close_loc", "c2_vs_c1", "vol_ratio", "c2_dir", "c1_dir", "smt", "psp", "early_cisd",
          "trend", "vs_dopen", "weekday", "hour"]:
    col = X[f]
    key = pd.qcut(col, 5, duplicates="drop") if col.nunique() > 6 and f != "hour" else (col // 4 * 4 if f == "hour" else col)
    g = X.groupby([key, "oos"], observed=True)["gross"].agg(["mean", "size"]).unstack("oos")
    g.columns = ["IS_gross", "OOS_gross", "IS_n", "OOS_n"]
    g["net_all"] = X.groupby(key, observed=True)["r"].mean()
    print(f"--- {f}\n{g.round(3).to_string()}\n")
X.to_csv(f"features_{sym}.csv", index=False)
