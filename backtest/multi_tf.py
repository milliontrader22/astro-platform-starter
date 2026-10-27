"""Other fractals on the same BTC data: resample 5m -> 15m (15m-4H) and 1H (1H-1D)."""
import itertools, pickle, os
import pandas as pd
import fm_backtest as fb
from fm_engine import generate, load
from run_sweep import params


def resample(df, minutes):
    g = df["time"] // (minutes * 60_000)
    out = df.groupby(g).agg(time=("time", "first"), open=("open", "first"), high=("high", "max"),
                            low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).reset_index(drop=True)
    out["dt"] = pd.to_datetime(out["time"], unit="ms", utc=True)
    return out


btc, eth = load("data/BTCUSDT-5m.csv"), load("data/ETHUSDT-5m.csv")
for name, ltf, htf in [("15m-4H", 15, 240), ("1H-1D", 60, 1440)]:
    b, e = resample(btc, ltf), resample(eth, ltf)
    cache = f"data/setups_{name}.pkl"
    S = pickle.load(open(cache, "rb")) if os.path.exists(cache) else generate(b, htf, 6, e)
    pickle.dump(S, open(cache, "wb"))
    A = fb.arrays(b)
    fb.MAX_HOLD = int(htf / ltf * 6)
    days = len(b) * ltf / 1440
    cut = int((b["dt"] < pd.Timestamp("2025-01-01", tz="UTC")).sum())
    IS = [s for s in S if s.create_bar < cut]; OOS = [s for s in S if s.create_bar >= cut]
    print(f"\n{name}: {len(S)} setups ({len(S)/days:.1f}/day)")
    rows = []
    for entry, stage, tp, xi, conf, ms in itertools.product(["cisd", "tspot_deep", "tspot_edge"], [3, 4], [2.0, 3.0],
                                                           [True, False], ["none", "smt"], [0.0, 0.005, 0.01]):
        kw = dict(entry=entry, max_stage=stage, tp_mode="dev", tp=tp, exit_invalid=xi, min_stop=ms, bias="none",
                  conf=conf, side="both", sess="all")
        _, a = fb.run(IS, params(kw), A); _, o = fb.run(OOS, params(kw), A)
        if a and o:
            rows.append({**kw, "per_day": (a["trades"] + o["trades"]) / days, "is_r": a["avg_r"], "is_n": a["trades"],
                         "oos_r": o["avg_r"], "oos_n": o["trades"], "oos_pf": o["pf"]})
    r = pd.DataFrame(rows)
    r["both"] = r[["is_r", "oos_r"]].min(axis=1)
    print(r.sort_values("both", ascending=False).drop(columns=["tp_mode", "bias", "side", "sess"]).head(8).round(3).to_string())
