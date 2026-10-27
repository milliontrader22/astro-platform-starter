"""Python port of the FM MTF HTF (PO3 fractal model) setup logic, for backtesting.

Mirrors pine/FM_MTF_HTF.pine bar by bar on the LTF series:
  - early CISD level / reclaim (bull/bearLevelFixed, bull/bearReclaimed)
  - HTF candles, C2 trigger (sweep of C1 extreme + close back inside)
  - setup lifecycle: C3 box -> C4..C6 boxes (T-Spot), XC2 death, XC4-6 invalidation, CISD confirm
Visual-only behaviour (history eviction, labels) is intentionally left out, so a new
sell setup never kills a live buy setup as the 'History = 1' input does on the chart.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Setup:
    is_bull: bool
    create_bar: int            # first LTF bar of C3
    extreme: float             # C2 low (bull) / high (bear)
    c2_high: float
    c2_low: float
    cisd_level: float
    cisd_index: int
    confirm_bar: int = -1      # bar where CISD confirmed (may be < create_bar: inside C2)
    confirm_stage: int = 0     # 2 = confirmed inside C2, 3..6 = during that candle
    stage: int = 3
    phase: int = 1
    dead: bool = False
    c4_invalid: bool = False
    stages: dict = field(default_factory=dict)    # stage -> (start_bar, top, bot)
    invalid: dict = field(default_factory=dict)   # stage -> bar of XC invalidation
    dead_bar: int = -1
    finish_bar: int = -1
    replaced_bar: int = -1
    smt: bool = False
    psp: bool = False

    @property
    def end_bar(self):
        ends = [b for b in (self.dead_bar, self.finish_bar, self.replaced_bar) if b >= 0]
        return min(ends) if ends else 10**12


def load(path):
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def htf_id(df: pd.DataFrame, htf_minutes: int) -> np.ndarray:
    return (df["time"].to_numpy(dtype="int64") // (htf_minutes * 60 * 1000))


def generate(df: pd.DataFrame, htf_minutes=60, max_candle=6, pair: pd.DataFrame | None = None):
    O = df["open"].to_numpy(); H = df["high"].to_numpy(); L = df["low"].to_numpy(); C = df["close"].to_numpy()
    n = len(df)
    hid = htf_id(df, htf_minutes)
    up = C > O; dw = C < O; eq = C == O

    # pair (SMT / PSP) HTF candles keyed by htf id
    pair_htf = None
    if pair is not None:
        p = pair.copy()
        p["hid"] = htf_id(p, htf_minutes)
        pair_htf = p.groupby("hid").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))

    o0 = h0 = l0 = o1 = h1 = l1 = np.nan
    bull_lvl = bear_lvl = np.nan
    bull_idx = bear_idx = -1
    bull_rec = bear_rec = False
    bull_rec_bar = bear_rec_bar = -1
    prev = dict(bull_lvl=np.nan, bull_idx=-1, bull_rec=False, bull_rec_bar=-1,
                bear_lvl=np.nan, bear_idx=-1, bear_rec=False, bear_rec_bar=-1)

    candles = []   # completed HTF candles: dict(o,h,l,c,hid)
    cur = None
    buy = sell = None
    setups = []

    for i in range(n):
        pc = i == 0 or hid[i] != hid[i - 1]
        if pc:
            o1, h1, l1 = o0, h0, l0
            o0, h0, l0 = O[i], H[i], L[i]
        h0 = max(h0, H[i]); l0 = min(l0, L[i])

        # ---- early CISD levels (lines 1619-1659 of the Pine) ----
        new_bull_idx = bull_idx
        if L[i] == l0 and L[i] < l1:
            if (dw[i] or eq[i]) and i >= 1 and (up[i - 1] or eq[i - 1]) and not (eq[i] and eq[i - 1]):
                bull_lvl, new_bull_idx = O[i], i
            else:
                for k in range(2, 11):
                    if i - k < 0 or L[i - k] < L[i]:
                        break
                    if (up[i - k] or eq[i - k]) and dw[i - k + 1]:
                        b = k - 1
                        lvl, idx = O[i - b], i - b
                        for j in range(b, -1, -1):
                            if O[i - j] > lvl and dw[i - j]:
                                lvl, idx = O[i - j], i - j
                        if lvl < O[i]:
                            lvl, idx = (H[i] if C[i] > O[i] else O[i]), i
                        bull_lvl, new_bull_idx = lvl, idx
                        break
        new_bear_idx = bear_idx
        if H[i] == h0 and H[i] > h1:
            if (up[i] or eq[i]) and i >= 1 and (dw[i - 1] or eq[i - 1]) and not (eq[i] and eq[i - 1]):
                bear_lvl, new_bear_idx = O[i], i
            else:
                for k in range(2, 11):
                    if i - k < 0 or H[i - k] > H[i]:
                        break
                    if (dw[i - k] or eq[i - k]) and up[i - k + 1]:
                        b = k - 1
                        lvl, idx = O[i - b], i - b
                        for j in range(b, -1, -1):
                            if O[i - j] < lvl and up[i - j]:
                                lvl, idx = O[i - j], i - j
                        if lvl > O[i]:
                            lvl, idx = (L[i] if C[i] < O[i] else O[i]), i
                        bear_lvl, new_bear_idx = lvl, idx
                        break
        if new_bull_idx != bull_idx and new_bull_idx >= 0:
            bull_rec, bull_rec_bar = False, -1
        bull_idx = new_bull_idx
        if not np.isnan(bull_lvl) and not bull_rec and C[i] > bull_lvl:
            bull_rec, bull_rec_bar = True, i
        if new_bear_idx != bear_idx and new_bear_idx >= 0:
            bear_rec, bear_rec_bar = False, -1
        bear_idx = new_bear_idx
        if not np.isnan(bear_lvl) and not bear_rec and C[i] < bear_lvl:
            bear_rec, bear_rec_bar = True, i

        # ---- HTF candles ----
        if pc:
            if cur is not None:
                candles.append(cur)
            cur = dict(o=O[i], h=H[i], l=L[i], c=C[i], hid=hid[i])
        else:
            cur["h"] = max(cur["h"], H[i]); cur["l"] = min(cur["l"], L[i]); cur["c"] = C[i]

        trig_buy = trig_sell = False
        if pc and len(candles) >= 2:
            c2, c1 = candles[-1], candles[-2]
            trig_buy = c2["l"] < c1["l"] and c2["c"] > c1["l"]
            trig_sell = c2["h"] > c1["h"] and c2["c"] < c1["h"]

        def box(prev_c):
            mid = (prev_c["h"] + prev_c["l"]) / 2
            return max(O[i], mid), min(O[i], mid)

        def process(trig, is_bull, existing):
            if trig:
                c2, c1 = candles[-1], candles[-2]
                p = "bull" if is_bull else "bear"
                s = Setup(is_bull, i, c2["l"] if is_bull else c2["h"], c2["h"], c2["l"],
                          prev[p + "_lvl"], prev[p + "_idx"])
                top, bot = box(c2)
                s.stages[3] = (i, top, bot)
                if prev[p + "_rec"] and not np.isnan(prev[p + "_lvl"]) and prev[p + "_idx"] >= 0 and prev[p + "_rec_bar"] >= 0:
                    s.confirm_bar, s.confirm_stage = prev[p + "_rec_bar"], 2
                if pair_htf is not None and c2["hid"] in pair_htf.index and c1["hid"] in pair_htf.index:
                    pc2, pc1 = pair_htf.loc[c2["hid"]], pair_htf.loc[c1["hid"]]
                    s.smt = (pc2.l >= pc1.l) if is_bull else (pc2.h <= pc1.h)
                    s.psp = (c2["c"] - c2["o"]) * (pc2.c - pc2.o) < 0
                if existing is not None and existing.end_bar > i:
                    existing.replaced_bar = i
                setups.append(s)
                return s
            if pc and existing is not None and not existing.dead:
                s = existing
                if s.phase == 1:
                    top, bot = box(candles[-1])
                    s.phase, s.stage, s.c4_invalid = 2, 4, False
                    s.stages[4] = (i, top, bot)
                elif s.phase == 2:
                    if not s.c4_invalid and s.stage < max_candle:
                        top, bot = box(candles[-1])
                        s.stage += 1
                        s.c4_invalid = False
                        s.stages[s.stage] = (i, top, bot)
                    else:
                        s.phase = 0
                        s.finish_bar = i
            return existing

        buy = process(trig_buy, True, buy)
        sell = process(trig_sell, False, sell)

        for s in (buy, sell):
            if s is None or s.dead or s.phase == 0:
                continue
            # extendAndCheck: XC2
            if (L[i] < s.extreme) if s.is_bull else (H[i] > s.extreme):
                s.dead, s.phase, s.dead_bar = True, 0, i
                continue
            # checkC4Invalid uses close[1]
            if s.phase == 2 and not s.c4_invalid:
                _, top, bot = s.stages[s.stage]
                if (C[i - 1] < bot) if s.is_bull else (C[i - 1] > top):
                    s.c4_invalid = True
                    s.invalid[s.stage] = i
            # checkCisdConfirm
            if s.confirm_bar < 0 and not np.isnan(s.cisd_level) and s.cisd_index >= 0:
                if (C[i] > s.cisd_level) if s.is_bull else (C[i] < s.cisd_level):
                    s.confirm_bar, s.confirm_stage = i, s.stage

        prev.update(bull_lvl=bull_lvl, bull_idx=bull_idx, bull_rec=bull_rec, bull_rec_bar=bull_rec_bar,
                    bear_lvl=bear_lvl, bear_idx=bear_idx, bear_rec=bear_rec, bear_rec_bar=bear_rec_bar)

    return setups
