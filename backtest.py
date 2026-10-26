#!/usr/bin/env python3
"""
Backtest v2: does any filter make the bot's colour signal profitable?

Base signal = the live bot's rules (EMA 5/13/26 15m, MACD 12/26 1h, Stoch 14/4/3 15m, Volume vs
20 SMA 15m; 🟢 = all 4 green, 🔴 = all 4 red), checked at the bot's update times.
A trade starts when a stock NEWLY turns 🟢 (buy) or 🔴 (sell short); entry = open of the next 15m
candle. Intraday only.

Filters tested (each alone, then combined):
  Nifty trend   buy only if Nifty 50 15m EMA13 > EMA26, sell only if below
  Not choppy    skip when Nifty 50 ADX(14, 15m) < 20 (sideways market)
  Daily trend   buy only if the stock is above its 20-day EMA (as of yesterday), sell only if below
  First signal  only the first signal of the day for each stock
Exits: 60 min, when the colour ends, and at the day's close.

To avoid fooling ourselves, results are shown separately for
  A = all but the last 4 months, B = the last 4 months.
A filter is only worth using if it helps in BOTH periods.

Numbers shown are average profit per trade BEFORE costs. A trade needs more than the cost
(default 0.08%) on average to make money.

Usage:
  python backtest.py --months 12 --cost 0.08
  python backtest.py --demo
"""
import argparse
import datetime as dt
import gzip
import json
import sys
import time
import urllib.parse

import numpy as np
import pandas as pd

import indicator_bot as bot

IST = bot.IST
G, R, W = bot.G, bot.R, bot.W
UPDATE_SLOTS = [s for s in bot.SLOTS if s != "15:30"]
NIFTY_KEY = "NSE_INDEX|Nifty 50"
B_MONTHS = 4


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def fetch_history(key: str, months: int) -> pd.DataFrame:
    """15m candles for the last `months` months (Upstox allows 1 month per request for 15m)."""
    k = urllib.parse.quote(key, safe="")
    end = dt.date.today()
    start = end - dt.timedelta(days=int(months * 30.5) + 10)
    frames, to = [], end
    while to > start:
        frm = max(start, to - dt.timedelta(days=27))
        candles = bot.http_get(f"{bot.API}/{k}/minutes/15/{to}/{frm}")["data"]["candles"]
        frames.append(bot.candles_to_df(candles))
        to = frm - dt.timedelta(days=1)
        time.sleep(0.3)
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.between_time("09:15", "15:29")


def nifty_key() -> str:
    """Upstox key for the Nifty 50 index (checked against the instrument list when possible)."""
    try:
        rows = json.loads(gzip.decompress(bot.http_get(bot.INSTRUMENTS_URL, raw=True)))
        for r in rows:
            if r.get("segment") == "NSE_INDEX" and str(r.get("name", "")).strip().lower() == "nifty 50":
                return r["instrument_key"]
    except Exception:
        pass
    return NIFTY_KEY


def demo_history(seed: int, days: int = 260, index: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range(end="2026-10-23", periods=days):
        start = pd.Timestamp(d.date().isoformat() + " 09:15", tz=IST)
        idx += list(pd.date_range(start, periods=25, freq="15min"))
    n = len(idx)
    drift = np.repeat(rng.normal(0, 0.001, n // 12 + 1), 12)[:n]
    ret = drift + rng.normal(0, 0.003, n)
    close = 1000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.0015, n)) * close
    vol = np.zeros(n) if index else rng.lognormal(11, 0.5, n) * (1 + np.abs(ret) / 0.003)
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + wick,
                         "Low": np.minimum(open_, close) - wick, "Close": close, "Volume": vol},
                        index=pd.DatetimeIndex(idx, name="time"))


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def colours_at_slots(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised version of indicator_bot.evaluate (same rules), one row per update time."""
    c = df["Close"]
    e5, e13, e26 = bot.ema(c, 5), bot.ema(c, 13), bot.ema(c, 26)
    ema_s = np.where((e5 > e13) & (e13 > e26), G, np.where((e5 < e13) & (e13 < e26), R, W))

    ll, hh = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    k = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(4).mean()
    d = k.rolling(3).mean()
    sto_s = np.where(k > d, G, np.where(k < d, R, W))

    sma = df["Volume"].rolling(20).mean()
    hv = df["Volume"] > sma
    vol_s = np.where(hv & (c > df["Open"]), G, np.where(hv & (c < df["Open"]), R, W))

    s = pd.DataFrame({"EMA": ema_s, "STO": sto_s, "VOL": vol_s, "price": c}, index=df.index)
    s["end"] = df.index + pd.Timedelta(minutes=15)

    h = bot.hourly(df)
    macd = bot.ema(h["Close"], 12) - bot.ema(h["Close"], 26)
    hm = pd.DataFrame({"end": pd.DatetimeIndex(list(h["end"])).tz_convert(IST),
                       "MACD": np.where(macd > 0, G, np.where(macd < 0, R, W))}).sort_values("end")

    # daily trend: price vs the 20-day EMA of daily closes, as of the previous day
    daily = c.groupby(df.index.date).last()
    d_ema = bot.ema(daily, 20).shift(1)
    s["d_ema"] = [d_ema.get(t.date(), np.nan) for t in df.index]

    s = s.iloc[60:]
    s = s[s["end"].dt.strftime("%H:%M").isin(UPDATE_SLOTS)].reset_index()
    s = pd.merge_asof(s.sort_values("end"), hm, on="end", direction="backward")
    parts = s[["EMA", "MACD", "STO", "VOL"]]
    s["colour"] = np.where((parts == G).all(axis=1), G, np.where((parts == R).all(axis=1), R, "🟠"))
    s["daily_trend"] = np.where(s["price"] > s["d_ema"], 1, np.where(s["price"] < s["d_ema"], -1, 0))
    return s.set_index("end")


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ADX."""
    h, l, c = df["High"], df["Low"], df["Close"]
    up, down = h.diff(), -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = 1 / n
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=a, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def market_context(nifty: pd.DataFrame) -> pd.DataFrame:
    """Nifty trend (+1/-1) and ADX, known at each update time (completed candles only)."""
    c = nifty["Close"]
    trend = np.where(bot.ema(c, 13) > bot.ema(c, 26), 1, -1)
    m = pd.DataFrame({"nifty_trend": trend, "nifty_adx": adx(nifty).values}, index=nifty.index)
    m.index = m.index + pd.Timedelta(minutes=15)          # value known when the candle closes
    return m


# ----------------------------------------------------------------------------
# Trades
# ----------------------------------------------------------------------------
def simulate(sym: str, df: pd.DataFrame, slots: pd.DataFrame, ctx: pd.DataFrame, cost: float):
    o, c = df["Open"], df["Close"]
    day_close = c.groupby(df.index.date).last()
    trades, randoms = [], []
    prev_col, seen_today, last_day = None, False, None
    slot_list = list(slots.itertuples())
    for i, row in enumerate(slot_list):
        t, col = row.Index, row.colour
        day = t.date()
        if day != last_day:
            seen_today, last_day = False, day
        if t not in o.index:
            prev_col = col
            continue
        entry = o[t]

        def close_after(minutes):
            k = t + pd.Timedelta(minutes=minutes - 15)
            return c[k] if (k in c.index and k.date() == day) else day_close[day]

        for name, px in (("60 min", close_after(60)), ("day close", day_close[day])):
            randoms.append({"time": t, "exit_rule": name, "gross": px / entry - 1})

        if col in (G, R) and col != prev_col:
            side = 1 if col == G else -1
            exit_px = day_close[day]
            for later in slot_list[i + 1:]:
                if later.Index.date() != day:
                    break
                if later.colour != col:
                    if later.Index in o.index:
                        exit_px = o[later.Index]
                    break
            cx = ctx.loc[t] if t in ctx.index else None
            base = {"symbol": sym, "time": t, "side": side,
                    "nifty_trend": int(cx["nifty_trend"]) if cx is not None else 0,
                    "nifty_adx": float(cx["nifty_adx"]) if cx is not None else np.nan,
                    "daily_trend": int(row.daily_trend), "first_today": not seen_today}
            for name, px in (("60 min", close_after(60)), ("colour ends", exit_px),
                             ("day close", day_close[day])):
                g = side * (px / entry - 1)
                trades.append({**base, "exit_rule": name, "gross": g, "net": g - cost})
            seen_today = True
        prev_col = col
    return trades, randoms


VARIANTS = [
    ("Base (all signals)", lambda t: t["side"] == t["side"]),
    ("+ Nifty trend", lambda t: t["side"] == t["nifty_trend"]),
    ("+ Not choppy", lambda t: t["nifty_adx"] >= 20),
    ("+ Daily trend", lambda t: t["side"] == t["daily_trend"]),
    ("+ First signal/day", lambda t: t["first_today"]),
    ("Nifty trend + not choppy", lambda t: (t["side"] == t["nifty_trend"]) & (t["nifty_adx"] >= 20)),
    ("Nifty + daily trend", lambda t: (t["side"] == t["nifty_trend"]) & (t["side"] == t["daily_trend"])),
    ("All three trends", lambda t: (t["side"] == t["nifty_trend"]) & (t["side"] == t["daily_trend"])
                                   & (t["nifty_adx"] >= 20)),
]


def summarise(tr: pd.DataFrame, rnd: pd.DataFrame, split: pd.Timestamp, cost: float) -> pd.DataFrame:
    rows = []
    for rule in ("60 min", "colour ends", "day close"):
        t = tr[tr["exit_rule"] == rule]
        for name, f in VARIANTS:
            sub = t[f(t)]
            a, b = sub[sub["time"] < split], sub[sub["time"] >= split]
            rows.append({"exit": rule, "variant": name, "n_A": len(a), "n_B": len(b),
                         "gross_A%": a["gross"].mean() * 100 if len(a) else np.nan,
                         "gross_B%": b["gross"].mean() * 100 if len(b) else np.nan,
                         "net_win%": (sub["net"] > 0).mean() * 100 if len(sub) else np.nan,
                         "net_total%": sub["net"].sum() * 100})
        if rule in ("60 min", "day close"):
            r = rnd[rnd["exit_rule"] == rule]
            a, b = r[r["time"] < split], r[r["time"] >= split]
            rows.append({"exit": rule, "variant": "Random buy (no signal)", "n_A": len(a), "n_B": len(b),
                         "gross_A%": a["gross"].mean() * 100, "gross_B%": b["gross"].mean() * 100,
                         "net_win%": ((r["gross"] - cost) > 0).mean() * 100, "net_total%": np.nan})
    return pd.DataFrame(rows)


def report(s: pd.DataFrame, cost: float, first, last, split, n_stocks, errors) -> str:
    c = cost * 100
    lines = [f"🧪 Backtest v2 · {n_stocks} stocks · {first:%d-%b-%Y} to {last:%d-%b-%Y}",
             f"A = to {split - pd.Timedelta(days=1):%d-%b-%Y} · B = {split:%d-%b-%Y} onward (last {B_MONTHS} months)",
             f"Avg profit per trade BEFORE costs, A | B. Needs above +{c:.2f} in BOTH to make money.",
             "✅ = beats costs in both periods", ""]
    for rule in ("60 min", "colour ends", "day close"):
        lines.append(f"Exit: {rule}")
        for _, r in s[s["exit"] == rule].iterrows():
            ok = r["gross_A%"] > c and r["gross_B%"] > c
            mark = "✅" if ok else "  "
            ga = "  n/a" if np.isnan(r["gross_A%"]) else f"{r['gross_A%']:+.3f}"
            gb = "  n/a" if np.isnan(r["gross_B%"]) else f"{r['gross_B%']:+.3f}"
            win = "" if np.isnan(r["net_win%"]) else f" · won {r['net_win%']:.0f}%"
            lines.append(f"{mark}{r['variant']}: {ga} | {gb} (n {r['n_A']}|{r['n_B']}){win}")
        lines.append("")
    best = s[(s["gross_A%"] > c) & (s["gross_B%"] > c) & (s["variant"] != "Random buy (no signal)")]
    if best.empty:
        lines.append("RESULT: no version beat costs in both periods.")
    else:
        b = best.sort_values("gross_B%", ascending=False).iloc[0]
        lines.append(f"RESULT: best = {b['variant']}, exit {b['exit']} "
                     f"({b['gross_A%']:+.3f} | {b['gross_B%']:+.3f}, {b['n_A'] + b['n_B']} trades).")
    lines.append("'won' = trades profitable after costs. Full table in the run's Artifacts.")
    if errors:
        lines.append(f"⚠️ Skipped: {', '.join(errors)}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=12)
    p.add_argument("--cost", type=float, default=0.08, help="round-trip cost in %% (default 0.08)")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    cost = args.cost / 100

    if args.demo:
        nifty = demo_history(999, index=True)
        keys = {}
    else:
        keys = bot.load_instrument_keys()
        nifty = fetch_history(nifty_key(), args.months)
    ctx = market_context(nifty)
    print(f"Nifty: {len(nifty)} candles")

    all_tr, all_rnd, errors, first, last = [], [], {}, None, None
    for i, sym in enumerate(bot.SYMBOLS):
        try:
            df = demo_history(i) if args.demo else fetch_history(keys[sym], args.months)
            tr, rnd = simulate(sym, df, colours_at_slots(df), ctx, cost)
            all_tr += tr
            all_rnd += rnd
            first = df.index[0] if first is None else min(first, df.index[0])
            last = df.index[-1] if last is None else max(last, df.index[-1])
            print(f"{sym}: {len(df)} candles, {sum(1 for t in tr if t['exit_rule'] == '60 min')} signals")
        except Exception as e:
            errors[sym] = str(e)[:80]
            print(f"[error] {sym}: {e}", file=sys.stderr)

    if not all_tr:
        text = "🧪 Backtest v2: no data or no signals. " + "; ".join(f"{k}: {v}" for k, v in errors.items())
        print(text)
        if not args.demo:
            bot.send_telegram(text)
        return
    tr, rnd = pd.DataFrame(all_tr), pd.DataFrame(all_rnd)
    split = (last - pd.DateOffset(months=B_MONTHS)).normalize()
    s = summarise(tr, rnd, split, cost)
    tr.to_csv("backtest_trades.csv", index=False)
    s.to_csv("backtest_summary.csv", index=False)
    text = report(s, cost, first, last, split, len(bot.SYMBOLS) - len(errors), errors)
    print("\n" + text)
    if not args.demo:
        bot.send_telegram(text)


if __name__ == "__main__":
    main()
    if bot.SEND_FAILURES:
        sys.exit(1)
