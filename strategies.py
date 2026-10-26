#!/usr/bin/env python3
"""
Backtest v3: new trading ideas for the 20 stocks (rules fixed BEFORE looking at results).

  ORB          Opening range breakout. Range = 9:15-9:45 high/low. First 15m candle that CLOSES
               above the high (buy) or below the low (sell) between 9:45 and 13:00; enter at the
               next candle's open. One trade per stock per day.
  ORB + volume Same, but the breakout candle's volume must be at least 1.5x the average volume of
               that same time of day over the previous 10 days. Otherwise no trade that day.
  ORB + RS     Same as ORB, but buy only if the stock is beating Nifty since the open AND Nifty is
               up since the open (sell: lagging Nifty and Nifty down).
  ORB exits    "close"  = hold to the day's close
               "stop"   = exit if price comes back to the middle of the opening range, else close
  RS top/bottom At 10:15, rank the 20 stocks by (stock change since open - Nifty change since open).
               Buy the 3 strongest, sell the 3 weakest, hold to close.
  RS + Nifty   Same, but buy the strongest only on days Nifty is up at 10:15, and sell the
               weakest only on days Nifty is down.
  Gap & go     Opens >= 1% above (below) yesterday's close and the first 15m candle closes in the
               same direction: buy (sell) at 9:30, hold to close.
  Gap fade     Same gap, but the first candle closes AGAINST the gap: trade against the gap.
  Random buy   Buy every stock every day at 9:45, hold to close (the comparison line).

Data is split into A (older two-thirds) and B (most recent third). A rule only counts as working
if it makes money AFTER costs in BOTH periods.

Usage: python strategies.py --months 24 --cost 0.08     |   python strategies.py --demo
"""
import argparse
import sys

import numpy as np
import pandas as pd

import backtest as bt
import indicator_bot as bot

OR_END = "09:45"
LAST_BREAKOUT_START = "12:45"     # breakout candle must start by 12:45, so entry is by 13:00
GAP = 0.01
RS_TIME = "10:00"                 # candle 10:00-10:15 closes at 10:15
TOP_N = 3
RVOL_MIN, RVOL_DAYS = 1.5, 10


def to_grid(df: pd.DataFrame) -> dict:
    """15m candles -> one table per field: rows = dates, columns = candle start time."""
    d = df.copy()
    d["date"] = d.index.date
    d["tod"] = d.index.strftime("%H:%M")
    return {f: d.pivot_table(index="date", columns="tod", values=f, aggfunc="last").sort_index()
            for f in ("Open", "High", "Low", "Close", "Volume")}


def day_close(g):
    return g["Close"].ffill(axis=1).iloc[:, -1]


def trade(rows, sym, date, side, entry, exit_px, name):
    if entry and exit_px and not np.isnan(entry) and not np.isnan(exit_px):
        rows.append({"strategy": name, "symbol": sym, "date": date, "side": side,
                     "gross": side * (exit_px / entry - 1)})


def orb_trades(sym, g, nifty_g, rows):
    O, H, L, C, V = g["Open"], g["High"], g["Low"], g["Close"], g["Volume"]
    cols = list(C.columns)
    if "09:15" not in cols or "09:30" not in cols:
        return
    close_px = day_close(g)
    or_hi = H[["09:15", "09:30"]].max(axis=1)
    or_lo = L[["09:15", "09:30"]].min(axis=1)
    rvol_base = {t: V[t].shift(1).rolling(RVOL_DAYS, min_periods=5).mean() for t in cols}
    n_open = nifty_g["Open"]["09:15"] if nifty_g is not None else None
    scan = [t for t in cols if OR_END <= t <= LAST_BREAKOUT_START]
    for date in C.index:
        if np.isnan(or_hi[date]) or np.isnan(or_lo[date]):
            continue
        for i, t in enumerate(scan):
            cpx = C.at[date, t]
            if np.isnan(cpx):
                continue
            side = 1 if cpx > or_hi[date] else -1 if cpx < or_lo[date] else 0
            if side == 0:
                continue
            nxt = cols[cols.index(t) + 1]
            entry = O.at[date, nxt]
            mid = (or_hi[date] + or_lo[date]) / 2
            # stop exit: first candle from entry onward that touches the range middle
            stop_px = close_px[date]
            for u in cols[cols.index(nxt):]:
                lo, hi, op = L.at[date, u], H.at[date, u], O.at[date, u]
                if np.isnan(lo):
                    continue
                if side == 1 and lo <= mid:
                    stop_px = min(mid, op)
                    break
                if side == -1 and hi >= mid:
                    stop_px = max(mid, op)
                    break
            for ex_name, ex_px in (("close", close_px[date]), ("stop", stop_px)):
                trade(rows, sym, date, side, entry, ex_px, f"ORB · {ex_name}")
                base = rvol_base[t].get(date, np.nan)
                if base and not np.isnan(base) and V.at[date, t] >= RVOL_MIN * base:
                    trade(rows, sym, date, side, entry, ex_px, f"ORB + volume · {ex_name}")
                if n_open is not None and date in n_open.index:
                    n_now = nifty_g["Close"].at[date, t] if t in nifty_g["Close"].columns else np.nan
                    n_ret = n_now / n_open[date] - 1
                    s_ret = cpx / O.at[date, "09:15"] - 1
                    if side * n_ret > 0 and side * (s_ret - n_ret) > 0:
                        trade(rows, sym, date, side, entry, ex_px, f"ORB + RS · {ex_name}")
            break                      # only the first breakout of the day


def gap_trades(sym, g, rows):
    O, C = g["Open"], g["Close"]
    if "09:15" not in O.columns or "09:30" not in O.columns:
        return
    close_px = day_close(g)
    prev = close_px.shift(1)
    for date in C.index:
        o1, c1, pc = O.at[date, "09:15"], C.at[date, "09:15"], prev[date]
        if np.isnan(pc) or np.isnan(o1) or np.isnan(c1):
            continue
        gap = o1 / pc - 1
        if abs(gap) < GAP:
            continue
        gdir = 1 if gap > 0 else -1
        cdir = 1 if c1 > o1 else -1 if c1 < o1 else 0
        if cdir == 0:
            continue
        entry = O.at[date, "09:30"]
        name = "Gap & go" if cdir == gdir else "Gap fade"
        trade(rows, sym, date, cdir, entry, close_px[date], name)


def rs_trades(grids, nifty_g, rows):
    if RS_TIME not in nifty_g["Close"].columns:
        return
    n_ret = nifty_g["Close"][RS_TIME] / nifty_g["Open"]["09:15"] - 1
    rs, entry, exit_ = {}, {}, {}
    for sym, g in grids.items():
        if RS_TIME not in g["Close"].columns or "10:15" not in g["Open"].columns:
            continue
        rs[sym] = g["Close"][RS_TIME] / g["Open"]["09:15"] - 1 - n_ret
        entry[sym] = g["Open"]["10:15"]
        exit_[sym] = day_close(g)
    rs = pd.DataFrame(rs)
    for date, row in rs.iterrows():
        row = row.dropna()
        if len(row) < 10 or date not in n_ret.index or np.isnan(n_ret[date]):
            continue
        ranked = row.sort_values()
        for side, picks in ((1, ranked.index[-TOP_N:]), (-1, ranked.index[:TOP_N])):
            for sym in picks:
                e, x = entry[sym].get(date, np.nan), exit_[sym].get(date, np.nan)
                trade(rows, sym, date, side, e, x, "RS top/bottom 3")
                if side * n_ret[date] > 0:
                    trade(rows, sym, date, side, e, x, "RS + Nifty direction")


def random_trades(sym, g, rows):
    if OR_END not in g["Open"].columns:
        return
    close_px = day_close(g)
    for date in g["Open"].index:
        trade(rows, sym, date, 1, g["Open"].at[date, OR_END], close_px[date], "Random buy (comparison)")


ORDER = ["ORB · close", "ORB · stop", "ORB + volume · close", "ORB + volume · stop",
         "ORB + RS · close", "ORB + RS · stop", "RS top/bottom 3", "RS + Nifty direction",
         "Gap & go", "Gap fade", "Random buy (comparison)"]


def report(tr: pd.DataFrame, cost: float, split, first, last, n_stocks, errors) -> str:
    c = cost * 100
    lines = [f"🧪 Backtest v3 · new ideas · {n_stocks} stocks",
             f"{first:%d-%b-%Y} to {last:%d-%b-%Y} · A = to {split - pd.Timedelta(days=1):%d-%b-%Y} · B = after",
             f"Avg profit per trade AFTER {c:.2f}% costs, A | B. ✅ = profitable in BOTH.", ""]
    winners = []
    for name in ORDER:
        s = tr[tr["strategy"] == name]
        if s.empty:
            lines.append(f"  {name}: no trades")
            continue
        net = s["gross"] - cost
        a, b = net[s["date"] < split], net[s["date"] >= split]
        ok = len(a) > 30 and len(b) > 30 and a.mean() > 0 and b.mean() > 0 and "Random" not in name
        if ok:
            winners.append((name, a.mean(), b.mean(), len(s)))
        lines.append(f"{'✅' if ok else '  '}{name}: {a.mean() * 100:+.3f} | {b.mean() * 100:+.3f} "
                     f"(n {len(a)}|{len(b)}) · won {(net > 0).mean() * 100:.0f}%")
    lines.append("")
    if winners:
        lines.append("RESULT: profitable in both periods:")
        for n, a, b, k in sorted(winners, key=lambda w: -min(w[1], w[2])):
            lines.append(f"  {n}: {a * 100:+.3f}% | {b * 100:+.3f}% per trade, {k} trades")
        lines.append("Next: confirm on a fresh period before trusting it.")
    else:
        lines.append("RESULT: no idea made money after costs in both periods.")
    lines.append("Per trade, +0.10% = ₹100 on ₹1 lakh. Full trade list in the run's Artifacts.")
    if errors:
        lines.append(f"⚠️ Skipped: {', '.join(errors)}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=24)
    p.add_argument("--cost", type=float, default=0.08)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    cost = args.cost / 100

    if args.demo:
        nifty, keys = bt.demo_history(999, days=480, index=True), {}
    else:
        keys = bot.load_instrument_keys()
        nifty = bt.fetch_history(bt.nifty_key(), args.months)
    nifty_g = to_grid(nifty)
    print(f"Nifty: {len(nifty)} candles")

    rows, grids, errors, first, last = [], {}, {}, None, None
    for i, sym in enumerate(bot.SYMBOLS):
        try:
            df = bt.demo_history(i, days=480) if args.demo else bt.fetch_history(keys[sym], args.months)
            g = to_grid(df)
            grids[sym] = g
            orb_trades(sym, g, nifty_g, rows)
            gap_trades(sym, g, rows)
            random_trades(sym, g, rows)
            first = df.index[0] if first is None else min(first, df.index[0])
            last = df.index[-1] if last is None else max(last, df.index[-1])
            print(f"{sym}: {len(df)} candles")
        except Exception as e:
            errors[sym] = str(e)[:80]
            print(f"[error] {sym}: {e}", file=sys.stderr)
    rs_trades(grids, nifty_g, rows)

    tr = pd.DataFrame(rows)
    if tr.empty:
        text = "🧪 Backtest v3: no trades produced. " + "; ".join(f"{k}: {v}" for k, v in errors.items())
    else:
        tr["date"] = pd.to_datetime(tr["date"])
        span = last - first
        split = (first + span * 2 / 3).normalize().tz_localize(None)
        tr.to_csv("strategy_trades.csv", index=False)
        text = report(tr, cost, split, first, last, len(bot.SYMBOLS) - len(errors), errors)
    print("\n" + text)
    if not args.demo:
        bot.send_telegram(text)


if __name__ == "__main__":
    main()
    if bot.SEND_FAILURES:
        sys.exit(1)
