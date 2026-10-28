"""Detail + neighbourhood check of the best 1m daily candidate."""
import numpy as np, pandas as pd
import scalp as SC

SC.G.update(SC.setup_globals("1m"))
import itertools
keys = [("tspot_edge", 3, False), ("tspot_edge", 4, False), ("tspot_deep", 3, False), ("tspot_deep", 4, False), ("cisd", 3, False)]
SC.G["entries"] = dict(SC.pmap(SC.entries_for, keys))
BEST = dict(ctx="15m-4H", mode="htf", entry="tspot_edge", stage=3, smt=False, target=("r", 1.5), window="ny",
            news="blackout", min_stop=0.002, sess_exit=False)
tr = SC.run_variant(BEST, keep=True)
A, t = SC.G["A"], SC.G["t"]
tr["dt"] = pd.to_datetime(tr.t, unit="ms", utc=True).dt.tz_convert("America/New_York")
print(f"trades {len(tr)}, weekdays {SC.G['wdays']}, per weekday {len(tr)/SC.G['wdays']:.2f}, days with a trade {tr.day.nunique()/SC.G['wdays']:.0%}")
print("exit reasons", tr.why.value_counts().to_dict())
# hold time / stop size: recompute via walk is internal; approximate from consecutive data
print("\nBy year (1% risk, compounding):")
for y, g in list(tr.groupby(tr.dt.dt.year)) + [("TOTAL", tr)]:
    eq = np.cumprod(1 + 0.01 * g.r.to_numpy()); dd = (1 - eq / np.maximum.accumulate(eq)).max()
    print(f"  {y}: n={len(g):4d} win={np.mean(g.r>0):.0%} avgR={g.r.mean():+.3f} R={g.r.sum():+6.1f} ret={eq[-1]-1:+6.1%} DD={dd:.1%}")
m = tr.groupby(tr.dt.dt.to_period("M")).r.sum()
print(f"\nMonths positive: {(m>0).mean():.0%} of {len(m)}; worst month {m.min():+.1f}R; best {m.max():+.1f}R")
print("By ET hour of entry:", tr.groupby(tr.dt.dt.hour).r.agg(['size','mean']).round(3).to_dict('index'))
print("\nNeighbourhood (one parameter changed at a time):")
for k, vals in dict(target=[("r", 1.0), ("r", 2.0), ("dev", 2.0)], window=["ny_am", "ny_ext", "lon_ny"], stage=[4],
                    entry=["tspot_deep", "cisd"], news=["none", "strict"], sess_exit=[True]).items():
    for v in vals:
        r = SC.run_variant({**BEST, k: v})
        if r:
            print(f"  {k}={v}: n={r['n']} /wday={r['per_wday']:.2f} avgR={r['avg_r']:+.3f} IS={r['is_r']:+.3f} OOS={r['oos_r']:+.3f} years={r['years']}")
tr.to_csv("scalp_best_trades.csv", index=False)
