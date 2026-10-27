"""Trade simulation + parameter sweep on top of fm_engine setups.

Conservative fills: market entries fill at the next bar's open, a bar that touches
both SL and TP is counted as a loss, stops gap-fill at the open, fees on both sides.
"""
import itertools
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fm_engine import generate, load


@dataclass(frozen=True)
class Params:
    entry: str = "cisd"          # cisd | tspot_edge (shallow box edge) | tspot_deep (prior-candle 50% / EQ side)
    c2_cisd: bool = True         # cisd entry: also enter when CISD already confirmed inside C2
    tspot_need_cisd: bool = True
    max_stage: int = 3           # last candle (C3..C6) an entry may trigger in
    sl_buf: float = 0.0          # fraction beyond the C2 extreme
    tp_mode: str = "dev"         # dev | r
    tp: float = 2.0              # |std dev| or R multiple
    min_rr: float = 0.0
    exit_invalid: bool = True    # close on XC4/5/6
    exit_finish: bool = False    # close when the setup's last candle ends
    side: str = "both"           # both | long | short
    sess: tuple = ()             # allowed entry hours UTC (empty = all)
    need_smt: bool = False
    need_psp: bool = False
    min_stop: float = 0.0        # skip trades whose stop is tighter than this fraction of price
    bias: str = "none"           # none | ema (4H EMA50 side) | do (longs below / shorts above the daily open)


FEE_TAKER = 0.0005  # Binance USDT-M futures; spot is 0.001 (0.00075 with BNB)
FEE_MAKER = 0.0002
MAX_HOLD = 12 * 24  # bars: hard time stop (1 day on 5m)


def sim_setup(s, p: Params, O, H, L, C, hours, ema, dopen):
    if p.side == "long" and not s.is_bull or p.side == "short" and s.is_bull:
        return None
    if p.need_smt and not s.smt or p.need_psp and not s.psp:
        return None
    if np.isnan(s.cisd_level):
        return None
    d = 1 if s.is_bull else -1
    end = s.end_bar
    n = len(O)

    def stage_at(bar):
        st = 3
        for k, (b, _, _) in s.stages.items():
            if b <= bar:
                st = max(st, k)
        return st

    entry_bar = entry_px = None
    if p.entry == "cisd":
        if s.confirm_stage == 2 and p.c2_cisd:
            sig = s.create_bar
        elif s.confirm_stage >= 3 and s.confirm_bar >= s.create_bar:
            sig = s.confirm_bar
        else:
            return None
        st = stage_at(sig)
        if st > p.max_stage or sig >= end or any(b <= sig for k, b in s.invalid.items() if k == st):
            return None
        if sig + 1 >= n:
            return None
        entry_bar, entry_px = sig + 1, O[sig + 1]
    else:
        start = s.create_bar
        if p.tspot_need_cisd:
            if s.confirm_bar < 0:
                return None
            start = max(start, s.confirm_bar)
        last = min(end, n - 1)
        for st in range(3, p.max_stage + 1):
            if st not in s.stages:
                break
            b0, top, bot = s.stages[st]
            b1 = s.stages[st + 1][0] if st + 1 in s.stages else last + 1
            inv = s.invalid.get(st, 10**12)
            shallow = p.entry == "tspot_edge"
            px = (top if shallow else bot) if s.is_bull else (bot if shallow else top)
            for j in range(max(b0, start + 1), min(b1, last + 1, inv)):
                if (L[j] <= px) if s.is_bull else (H[j] >= px):
                    entry_bar = j
                    entry_px = min(px, O[j]) if s.is_bull else max(px, O[j])
                    break
            if entry_bar is not None:
                break
        if entry_bar is None:
            return None

    if p.sess and hours[entry_bar] not in p.sess:
        return None
    if p.bias == "ema" and ((C[entry_bar - 1] - ema[entry_bar]) * d <= 0 or np.isnan(ema[entry_bar])):
        return None
    if p.bias == "do" and (entry_px - dopen[entry_bar]) * d >= 0:
        return None
    sl = s.extreme * (1 - d * p.sl_buf)
    risk = (entry_px - sl) * d
    if risk <= 0 or risk / entry_px < p.min_stop:
        return None
    if p.tp_mode == "dev":
        rng = (s.cisd_level - s.extreme) * d
        tp = s.cisd_level + d * rng * p.tp
    else:
        tp = entry_px + d * risk * p.tp
    if (tp - entry_px) * d <= 0 or (tp - entry_px) * d / risk < p.min_rr:
        return None

    exit_bar = exit_px = None
    reason = "time"
    inv_bars = sorted(b for k, b in s.invalid.items())
    for j in range(entry_bar, min(n, entry_bar + MAX_HOLD)):
        o = O[j]
        if j > entry_bar and ((o <= sl) if s.is_bull else (o >= sl)):
            exit_bar, exit_px, reason = j, o, "sl"; break
        if (L[j] <= sl) if s.is_bull else (H[j] >= sl):
            exit_bar, exit_px, reason = j, sl, "sl"; break
        if (H[j] >= tp) if s.is_bull else (L[j] <= tp):
            exit_bar, exit_px, reason = j, max(tp, o) if (s.is_bull and j > entry_bar) else (min(tp, o) if j > entry_bar else tp), "tp"; break
        if p.exit_invalid and any(b == j for b in inv_bars if b >= entry_bar) and j + 1 < n:
            exit_bar, exit_px, reason = j + 1, O[j + 1], "xc"; break
        if p.exit_finish and s.finish_bar == j and j + 1 < n:
            exit_bar, exit_px, reason = j + 1, O[j + 1], "end"; break
    if exit_bar is None:
        j = min(n - 1, entry_bar + MAX_HOLD - 1)
        exit_bar, exit_px = j, C[j]
    gross_r = (exit_px - entry_px) * d / risk
    fee_in = FEE_TAKER if p.entry == "cisd" else FEE_MAKER
    fee_out = FEE_MAKER if reason == "tp" else FEE_TAKER
    fee_r = (fee_in * entry_px + fee_out * exit_px) / risk
    return dict(bull=s.is_bull, entry_bar=entry_bar, exit_bar=exit_bar, entry=entry_px, exit=exit_px,
                sl=sl, tp=tp, r=gross_r - fee_r, stop_pct=risk / entry_px, reason=reason)


def run(setups, p: Params, arrs, risk=0.01, max_lev=10.0):
    O, H, L, C, hours, ema, dopen = arrs
    trades = []
    busy_until = -1
    for s in sorted(setups, key=lambda s: s.create_bar):
        t = sim_setup(s, p, O, H, L, C, hours, ema, dopen)
        if t is None or t["entry_bar"] <= busy_until:
            continue
        trades.append(t)
        busy_until = t["exit_bar"]
    if not trades:
        return pd.DataFrame(), {}
    tr = pd.DataFrame(trades).sort_values("entry_bar")
    # risk-based sizing with leverage cap: notional = risk/stop_pct, capped at max_lev
    scale = np.minimum(1.0, max_lev * tr["stop_pct"] / risk)
    tr["r_eff"] = tr["r"] * scale
    eq = np.cumprod(1 + risk * tr["r_eff"].to_numpy())
    peak = np.maximum.accumulate(eq)
    wins = tr["r"] > 0
    stats = dict(trades=len(tr), winrate=wins.mean(), avg_r=tr["r"].mean(), total_r=tr["r"].sum(),
                 pf=tr.loc[wins, "r"].sum() / max(1e-9, -tr.loc[~wins, "r"].sum()),
                 max_dd=(1 - eq / peak).max(), ret=eq[-1] - 1)
    return tr, stats


def sweep(setups, arrs, grid: dict, base=Params(), min_trades=30):
    keys = list(grid)
    rows = []
    for vals in itertools.product(*grid.values()):
        p = Params(**{**base.__dict__, **dict(zip(keys, vals))})
        _, st = run(setups, p, arrs)
        if st and st["trades"] >= min_trades:
            rows.append({**dict(zip(keys, vals)), **st})
    return pd.DataFrame(rows)


def arrays(df):
    t = df["time"].to_numpy(dtype="int64")
    c = df["close"]
    # 4H EMA50 from closed 4H candles only, forward-filled onto 5m bars (no lookahead)
    h4 = t // (240 * 60 * 1000)
    h4_close = c.groupby(h4).last()
    h4_ema = h4_close.ewm(span=50, adjust=False).mean().shift(1)
    ema = h4_ema.reindex(h4).to_numpy()
    day = t // (24 * 3600 * 1000)
    dopen = df["open"].groupby(day).transform("first").to_numpy()
    return (df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), c.to_numpy(),
            df["dt"].dt.hour.to_numpy(), ema, dopen)


if __name__ == "__main__":
    df = load(sys.argv[1])
    pair = load(sys.argv[2]) if len(sys.argv) > 2 else None
    setups = generate(df, 60, 6, pair)
    print(len(setups), "setups")
    tr, st = run(setups, Params(), arrays(df))
    print(st)
