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
