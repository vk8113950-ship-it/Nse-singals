#!/usr/bin/env python3
"""
Backtest of the NSE indicator bot's colour signal on real Upstox data.

Uses exactly the bot's rules (EMA 5/13/26 15m, MACD 12/26 1h, Stoch 14/4/3 15m, Volume vs 20 SMA 15m;
🟢 = all 4 green, 🔴 = all 4 red), checked at the bot's update times (9:45, 10:15 ... 15:15).

A "trade" starts when a stock NEWLY turns 🟢 (buy) or 🔴 (sell short) at an update:
  entry  = open of the next 15m candle (the price right after the message would arrive)
Three ways to exit are tested side by side:
  30 min      close 30 minutes after entry
  60 min      close 60 minutes after entry
  colour ends exit at the first later update where the stock is no longer that colour
All trades close by the end of the day (intraday only). Costs are subtracted from every trade.

For comparison, "random" = buying at EVERY update time for the same holding period, so you can
see whether 🟢/🔴 does better than just being in the market.

Usage:
  python backtest.py --months 6 --cost 0.08
  python backtest.py --demo        (synthetic data, offline test)
"""
import argparse
import datetime as dt
import os
import sys
import time
import urllib.parse

import numpy as np
import pandas as pd

import indicator_bot as bot

IST = bot.IST
G, R, W = bot.G, bot.R, bot.W
UPDATE_SLOTS = [s for s in bot.SLOTS if s != "15:30"]      # 15:30 is the close; no trade can start then


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


def demo_history(seed: int, days: int = 130) -> pd.DataFrame:
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
    vol = rng.lognormal(11, 0.5, n) * (1 + np.abs(ret) / 0.003)
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + wick,
                         "Low": np.minimum(open_, close) - wick, "Close": close, "Volume": vol},
                        index=pd.DatetimeIndex(idx, name="time"))


# ----------------------------------------------------------------------------
# Signals (vectorised version of indicator_bot.evaluate, same rules)
# ----------------------------------------------------------------------------
def colours_at_slots(df: pd.DataFrame) -> pd.DataFrame:
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

    s = pd.DataFrame({"EMA": ema_s, "STO": sto_s, "VOL": vol_s}, index=df.index)
    s["end"] = df.index + pd.Timedelta(minutes=15)

    h = bot.hourly(df)
    macd = bot.ema(h["Close"], 12) - bot.ema(h["Close"], 26)
    hm = pd.DataFrame({"end": pd.DatetimeIndex(list(h["end"])).tz_convert(IST),
                       "MACD": np.where(macd > 0, G, np.where(macd < 0, R, W))})
    hm = hm.sort_values("end")

    s = s.iloc[60:]                                   # same warm-up as the bot (60 candles)
    s = s[s["end"].dt.strftime("%H:%M").isin(UPDATE_SLOTS)].reset_index()
    s = pd.merge_asof(s.sort_values("end"), hm, on="end", direction="backward")
    parts = s[["EMA", "MACD", "STO", "VOL"]]
    s["colour"] = np.where((parts == G).all(axis=1), G, np.where((parts == R).all(axis=1), R, "🟠"))
    return s.set_index("end")


# ----------------------------------------------------------------------------
# Trades
# ----------------------------------------------------------------------------
def simulate(sym: str, df: pd.DataFrame, slots: pd.DataFrame, cost: float):
    o = df["Open"]
    c = df["Close"]
    day_close = c.groupby(df.index.date).last()
    day_last_t = pd.Series(df.index, index=df.index).groupby(df.index.date).last()
    trades, randoms = [], []
    prev_col = None
    slot_list = list(slots.itertuples())
    for i, row in enumerate(slot_list):
        t, col = row.Index, row.colour
        day = t.date()
        entry_t = t                                          # candle starting at the update time
        if entry_t not in o.index:
            prev_col = col
            continue
        entry = o[entry_t]

        def close_after(minutes):
            exit_candle = t + pd.Timedelta(minutes=minutes - 15)    # candle ending at t+minutes
            if exit_candle in c.index and exit_candle.date() == day:
                return c[exit_candle]
            return day_close[day]

        # random baseline: long at every update
        for m in (30, 60):
            randoms.append({"hold": f"{m} min", "ret": close_after(m) / entry - 1 - cost})

        if col in (G, R) and col != prev_col:
            side = 1 if col == G else -1
            # colour-ends exit
            exit_px, exit_t = day_close[day], day_last_t[day] + pd.Timedelta(minutes=15)
            for later in slot_list[i + 1:]:
                if later.Index.date() != day:
                    break
                if later.colour != col:
                    if later.Index in o.index:
                        exit_px, exit_t = o[later.Index], later.Index
                    break
            for name, px in (("30 min", close_after(30)), ("60 min", close_after(60)), ("colour ends", exit_px)):
                gross = side * (px / entry - 1)
                trades.append({"symbol": sym, "signal_time": t, "side": "BUY" if side == 1 else "SELL",
                               "exit_rule": name, "entry": round(entry, 2), "exit": round(px, 2),
                               "gross_pct": round(gross * 100, 3), "net_pct": round((gross - cost) * 100, 3)})
        prev_col = col
    return trades, randoms


def stats(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0}
    net = sub["net_pct"]
    wins, losses = net[net > 0], net[net <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    return {"n": n, "wins": len(wins), "win%": 100 * len(wins) / n, "avg": net.mean(),
            "total": net.sum(), "pf": pf, "gross_win%": 100 * (sub["gross_pct"] > 0).mean()}


def report(tr: pd.DataFrame, rnd: pd.DataFrame, months: int, cost: float, first: str, last: str,
           n_stocks: int, errors: dict) -> str:
    lines = [f"🧪 Backtest: indicator colours, {n_stocks} stocks",
             f"Data {first} to {last} (~{months} months, 15m candles)",
             f"Cost {cost * 100:.2f}% per trade subtracted. Entry right after each update.", ""]
    if tr.empty:
        lines.append("No 🟢/🔴 signals in this period.")
        return "\n".join(lines)
    n_sig = tr[tr["exit_rule"] == "30 min"].shape[0]
    buys = tr[(tr["exit_rule"] == "30 min") & (tr["side"] == "BUY")].shape[0]
    lines.append(f"Signals: {n_sig} ({buys} 🟢 buy, {n_sig - buys} 🔴 sell)")
    lines.append("")
    for rule in ("30 min", "60 min", "colour ends"):
        s = stats(tr[tr["exit_rule"] == rule])
        lines.append(f"Exit after {rule}:" if rule != "colour ends" else "Exit when colour ends:")
        lines.append(f"  Profitable: {s['wins']} of {s['n']} ({s['win%']:.0f}%) "
                     f"[before costs {s['gross_win%']:.0f}%]")
        lines.append(f"  Avg per trade {s['avg']:+.3f}%  ·  total {s['total']:+.1f}%  ·  PF {s['pf']:.2f}")
        for side, icon in (("BUY", G), ("SELL", R)):
            ss = stats(tr[(tr["exit_rule"] == rule) & (tr["side"] == side)])
            if ss["n"]:
                lines.append(f"  {icon} {ss['wins']}/{ss['n']} won ({ss['win%']:.0f}%), avg {ss['avg']:+.3f}%")
        lines.append("")
    lines.append("Compare: buying at EVERY update (no signal)")
    for hold in ("30 min", "60 min"):
        r = rnd[rnd["hold"] == hold]["ret"] * 100
        lines.append(f"  {hold}: {100 * (r > 0).mean():.0f}% profitable, avg {r.mean():+.3f}%")
    lines += ["", "PF = profit factor (total won ÷ total lost; above 1 = net profit).",
              "Past results only. Small edges can vanish with slippage."]
    if errors:
        lines.append(f"⚠️ Skipped: {', '.join(errors)}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=6)
    p.add_argument("--cost", type=float, default=0.08, help="round-trip cost in %% (default 0.08)")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--csv", default="backtest_trades.csv")
    args = p.parse_args()
    cost = args.cost / 100

    keys = {} if args.demo else bot.load_instrument_keys()
    all_tr, all_rnd, errors, first, last = [], [], {}, None, None
    for i, sym in enumerate(bot.SYMBOLS):
        try:
            df = demo_history(i) if args.demo else fetch_history(keys[sym], args.months)
            slots = colours_at_slots(df)
            tr, rnd = simulate(sym, df, slots, cost)
            all_tr += tr
            all_rnd += rnd
            first = min(first or df.index[0], df.index[0])
            last = max(last or df.index[-1], df.index[-1])
            print(f"{sym}: {len(df)} candles, {sum(1 for t in tr if t['exit_rule'] == '30 min')} signals")
        except Exception as e:
            errors[sym] = str(e)[:80]
            print(f"[error] {sym}: {e}", file=sys.stderr)

    tr = pd.DataFrame(all_tr)
    rnd = pd.DataFrame(all_rnd)
    if not tr.empty:
        tr.to_csv(args.csv, index=False)
    text = report(tr, rnd, args.months, cost,
                  first.strftime("%d-%b-%Y") if first is not None else "-",
                  last.strftime("%d-%b-%Y") if last is not None else "-",
                  len(bot.SYMBOLS) - len(errors), errors)
    print("\n" + text)
    if not args.demo:
        bot.send_telegram(text)


if __name__ == "__main__":
    main()
    if bot.SEND_FAILURES:
        sys.exit(1)
