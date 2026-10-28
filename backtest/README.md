# FM PO3 backtest — BTCUSDT 5m-1H

Python port of `pine/FM_MTF_HTF.pine` (C2 → CISD → T-Spot C3-C6) plus a trade simulator.

```
python fetch_data.py BTCUSDT 5m 2022-01 2026-09   # data.binance.vision archives
python fetch_data.py ETHUSDT 5m 2022-01 2026-09   # SMT / PSP pair
python run_sweep.py data/BTCUSDT-5m.csv data/ETHUSDT-5m.csv   # IS 2022-24, OOS 2025-26
python family.py   # family-level OOS check
python robust.py   # year-by-year / sensitivity / fee stress of the chosen variant
```

Fills are conservative: market entries at next open, a bar touching SL and TP counts as SL,
fees 0.02% maker (limit entries, targets) / 0.05% taker (market, stops) — Binance USDT-M futures.

## Findings (data 2022-01 → 2026-10, 20,249 setups)

* Raw model (CISD entry, -2 target, stop at C2 extreme): ~0.00R gross, **-0.39R net** per trade.
  Median stop is 0.42%, so fees cost ~0.24R/trade. Stop size is the main lever.
* Robust in both IS and OOS: **SMT required on C2**, **T-Spot Deep limit entry**, **London+NY**,
  **stop ≥ 0.5%**, **C3 only**, **no XC4 exit**, wide target (-3 dev). CISD market entry, PSP and
  trading all hours lose. Long/short preference flips between periods, so both sides are kept.
* Chosen variant (Pine strategy defaults): 166 trades, 42% win, **+0.15R/trade, PF 1.24, DD 14%**
  at 1% risk. By year (R): 2022 +22.2, 2023 +4.0, 2024 -6.6, 2025 +2.4, 2026 +3.3.
  With the daily-open filter: 93 trades, +0.33R, PF 1.52.
* Caveats: few trades (~35/yr), most profit from 2022, a cliff between 0.3% and 0.4% min stop,
  and **spot fees (0.1%) turn it negative** (-0.06R). Treat as a candidate for forward testing.

## Round 2: frequency, features, other coins (`fees.py`, `features.py`, `multi_asset.py`, `weekdays.py`)

* **No gross edge at high frequency.** Trading every setup (7-8/day, concurrent positions) gives
  +0.007R/trade with **zero fees**, -0.33R with standard futures fees. Entry type, C1/C2 targets
  and break-even at 1R do not change this.
* Features (zero-fee R, IS vs OOS): only trend alignment (4H EMA, +0.04/+0.07R) and SMT
  (+0.005/+0.11R) help in both periods. Sweep depth, C2 close location, C2/C1 range, volume,
  hour, weekday: no consistent effect.
* Monday-Friday only: lowers the edge of the selective variant; only trims losses on frequent ones.
* **Other coins (same rules, no per-coin tuning):** the BTC variant loses on ETH (-0.17R),
  SOL (-0.21R), AVAX, LINK, XRP; ~0 on BNB, DOGE, ADA. 9-coin portfolio: 2.5 trades/day,
  -0.08R/trade; C3-C4 variant 4.6/day, -0.06R. The BTC result is most likely noise.

## Round 3: nested fractals (`nested.py`, `nested_multi.py`)

HTF setup (15m-4H or 1H-1D, CISD confirmed and still alive) gives direction; LTF setup (5m-1H or
1m-15m) in the same direction gives the entry. Stop/target from the LTF, the HTF, or mixed.

* **5m execution (2080 variants):** everything at >= 1 trade/day loses (~-0.33R). A cluster at
  ~0.1/day is positive on BTC in both periods (15m-4H context, 5m T-Spot Deep + SMT, HTF stop and
  target; best n=271, +0.11R, every year positive). **On the other 8 coins it loses** (only ADA
  positive); 9-coin average -0.03 to -0.05R. Nesting helps a little vs. no nesting (-0.08R), not enough.
* **1m execution (2020 variants, 75,623 1m-15m setups):** 1.3 trades/day loses -0.06R (HTF stop
  keeps fees small, but there is no gross edge); best low-frequency variants ~+0.02R. 1.2% of
  variants positive in both IS and OOS.

## Round 4: daily scalping, BTC only (`scalp.py`, `scalp_detail.py`)

Weekdays only, New York-time windows, news filter (no entries 08:25-09:00 and 13:55-14:30 ET;
"strict" also skips FOMC days and NFP Fridays), optional flat-at-window-end, optional HTF context.

* **5m execution:** 689 variants trade >= 0.7 per weekday; **none** is positive in both 2022-24 and 2025-26.
* **1m execution:** 1545 daily variants, 8 positive in both periods. Best (picked on 2022-24 only):
  15m-4H context alive, 1m-15m T-Spot Edge in C3, stop at the 4H C2 extreme, target 1.5R,
  NY 09:30-16:00 ET, news blackout. 893 trades, 0.71/weekday (62% of weekdays), +0.049R/trade,
  IS +0.051 / OOS +0.047; by year R: 2022 +28.1, 2023 -7.8, 2024 +8.1, 2025 +2.5, 2026 +12.6;
  +47% at 1% risk, max DD 19.7%.
* Caveats: narrow peak — changing entry, stage, window, target 1R or forcing flat at session end
  takes it to ~0; 36% of trades exit on the 24h time stop (not really scalping). Not statistically
  significant; forward-test only.
