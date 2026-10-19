#!/usr/bin/env python3
"""
NSE intraday signal scanner + backtester
-----------------------------------------
Indicators (your settings):
  EMA 5 / 13 / 26
  MACD 12 / 26 (signal 9)
  Stochastic 14 / 4 / 3   (%K length 14, %K smoothing 4, %D 3)
  Volume vs 20-period average

Timeframes:
  15-minute candles -> trend filter (BULL / BEAR / NEUTRAL)
  5-minute candles  -> entry signal, only taken in the direction of the 15m trend

Strategies:
  --strategy 1  Momentum: EMAs stacked, fresh MACD crossover, Stochastic agrees,
                volume >= 1.5x its 20-bar average.
  --strategy 2  Pullback: price dips to EMA 13 inside the trend, Stochastic turns
                up from oversold (down from overbought for shorts), MACD histogram
                improving, volume >= its 20-bar average.

Usage:
  pip install yfinance pandas numpy
  python nse_signal_agent.py                                  # scan default stocks, strategy 1
  python nse_signal_agent.py --symbols RELIANCE SBIN --strategy 2
  python nse_signal_agent.py --backtest --strategy 1          # test on last ~60 days
  python nse_signal_agent.py --backtest --hold 6 --cost 0.05  # 30-min hold, 0.05% costs
  python nse_signal_agent.py --demo --backtest                # offline test on fake data

Telegram alerts (used by the GitHub Actions workflow):
  set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables, then
  python nse_signal_agent.py --test-alert                     # send a test message
  python nse_signal_agent.py --alert --strategy 1             # send only NEW signals
  python nse_signal_agent.py --snapshot                       # send current status of every stock
  python nse_signal_agent.py --summary                        # send today's signal summary
  --alert only runs during market hours (Mon-Fri 09:20-16:00 IST) unless --force is given,
  and remembers what it already sent in sent_signals.json so you don't get repeats.

Notes:
  * Yahoo only provides ~60 days of 5m/15m history, so backtests are short.
  * Index symbols (^NSEI, ^NSEBANK) have no volume on Yahoo; the volume rule
    is skipped for them automatically.
  * This is a research tool, not investment advice. Signals are probabilities,
    not predictions. Paper-trade before using real money.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"

CFG = dict(
    ema_fast=5, ema_mid=13, ema_slow=26,
    macd_fast=12, macd_slow=26, macd_signal=9,
    stoch_k=14, stoch_k_smooth=4, stoch_d=3,
    vol_len=20,
)

DEFAULT_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "SBIN", "AXISBANK", "KOTAKBANK", "LT", "BHARTIARTL",
]

# No new entries before / after these times (first 15 min are noisy,
# last 30 min leave no room for the trade).
ENTRY_START = "09:30"
ENTRY_END = "15:00"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def to_yahoo(sym: str) -> str:
    sym = sym.strip().upper()
    if sym.startswith("^") or "." in sym:
        return sym
    return sym + ".NS"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    df.index.name = "time"
    df = df.between_time("09:15", "15:29")
    return df.dropna(subset=["Close"])


def drop_forming_candle(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Remove the candle that hasn't closed yet, so signals never use unfinished data."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz=IST)
    if df.index[-1] + pd.Timedelta(minutes=minutes) > now:
        return df.iloc[:-1]
    return df


def fetch(symbol: str, interval: str, period: str = "59d") -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return clean(df)


def resample_15m(df5: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df5.resample("15min").agg(agg).dropna(subset=["Close"])


def demo_data(seed: int):
    """Synthetic 5m + 15m candles with trending regimes, for offline testing."""
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range("2026-08-03", periods=40):
        start = pd.Timestamp(d).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
        idx += list(pd.date_range(start, periods=75, freq="5min"))
    idx = pd.DatetimeIndex(idx, name="time")
    n = len(idx)
    drift = np.repeat(rng.normal(0, 0.0005, n // 30 + 1), 30)[:n]
    ret = drift + rng.normal(0, 0.0015, n)
    close = 1000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.0008, n)) * close
    df5 = pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) + wick,
        "Low": np.minimum(open_, close) - wick,
        "Close": close,
        "Volume": rng.lognormal(10, 0.4, n) * (1 + np.abs(ret) / 0.0015),
    }, index=idx)
    return df5, resample_15m(df5)


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame, c=CFG) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["ema_f"] = close.ewm(span=c["ema_fast"], adjust=False).mean()
    df["ema_m"] = close.ewm(span=c["ema_mid"], adjust=False).mean()
    df["ema_s"] = close.ewm(span=c["ema_slow"], adjust=False).mean()

    macd_fast = close.ewm(span=c["macd_fast"], adjust=False).mean()
    macd_slow = close.ewm(span=c["macd_slow"], adjust=False).mean()
    df["macd"] = macd_fast - macd_slow
    df["macd_sig"] = df["macd"].ewm(span=c["macd_signal"], adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    ll = df["Low"].rolling(c["stoch_k"]).min()
    hh = df["High"].rolling(c["stoch_k"]).max()
    raw_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    df["stoch_k"] = raw_k.rolling(c["stoch_k_smooth"]).mean()
    df["stoch_d"] = df["stoch_k"].rolling(c["stoch_d"]).mean()

    df["vol_avg"] = df["Volume"].rolling(c["vol_len"]).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_avg"].replace(0, np.nan)

    warmup = max(c["ema_slow"], c["macd_slow"] + c["macd_signal"],
                 c["stoch_k"] + c["stoch_k_smooth"] + c["stoch_d"], c["vol_len"])
    return df.iloc[warmup:]


def add_trend(df15: pd.DataFrame) -> pd.DataFrame:
    df15 = df15.copy()
    bull = (df15["ema_f"] > df15["ema_m"]) & (df15["ema_m"] > df15["ema_s"]) & (df15["macd"] > df15["macd_sig"])
    bear = (df15["ema_f"] < df15["ema_m"]) & (df15["ema_m"] < df15["ema_s"]) & (df15["macd"] < df15["macd_sig"])
    df15["trend"] = np.select([bull, bear], ["BULL", "BEAR"], "NEUTRAL")
    return df15


def attach_15m_trend(df5: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    """Give each 5m candle the trend of the last *completed* 15m candle (no look-ahead)."""
    hi = pd.DataFrame({"trend15": df15["trend"].values,
                       "known_at": df15.index + pd.Timedelta(minutes=15)})
    lo = df5.reset_index()
    lo["known_at"] = lo["time"] + pd.Timedelta(minutes=5)
    m = pd.merge_asof(lo.sort_values("known_at"), hi.sort_values("known_at"),
                      on="known_at", direction="backward")
    m["trend15"] = m["trend15"].fillna("NEUTRAL")
    return m.drop(columns="known_at").set_index("time")


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
def crossed_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossed_down(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def within(cond: pd.Series, bars: int) -> pd.Series:
    """True if cond happened on this bar or any of the previous (bars-1) bars."""
    return cond.astype(int).rolling(bars, min_periods=1).max().astype(bool)


def volume_ok(df: pd.DataFrame, threshold: float) -> pd.Series:
    if df["Volume"].sum() == 0:          # indices on Yahoo have no volume
        return pd.Series(True, index=df.index)
    return df["vol_ratio"] >= threshold


def strategy_1(df):
    """Momentum: stacked EMAs + fresh MACD cross + Stochastic agreement + volume spike."""
    up, dn = df["trend15"] == "BULL", df["trend15"] == "BEAR"
    k, d = df["stoch_k"], df["stoch_d"]
    vol = volume_ok(df, 1.5)

    long_ = (up
             & (df["ema_f"] > df["ema_m"]) & (df["ema_m"] > df["ema_s"])
             & within(crossed_up(df["macd"], df["macd_sig"]), 3)
             & (k > d) & (k < 80)
             & vol)
    short = (dn
             & (df["ema_f"] < df["ema_m"]) & (df["ema_m"] < df["ema_s"])
             & within(crossed_down(df["macd"], df["macd_sig"]), 3)
             & (k < d) & (k > 20)
             & vol)
    return long_, short


def strategy_2(df):
    """Pullback: dip to EMA 13 in the trend, Stochastic turns from extreme, MACD histogram improving."""
    up, dn = df["trend15"] == "BULL", df["trend15"] == "BEAR"
    k, d, hist = df["stoch_k"], df["stoch_d"], df["macd_hist"]
    vol = volume_ok(df, 1.0)

    long_ = (up
             & (df["Close"] > df["ema_s"])
             & within(df["Low"] <= df["ema_m"], 3) & (df["Close"] > df["ema_m"])
             & within(crossed_up(k, d) & (k.shift(1) < 30), 2)
             & (hist > hist.shift(1))
             & vol)
    short = (dn
             & (df["Close"] < df["ema_s"])
             & within(df["High"] >= df["ema_m"], 3) & (df["Close"] < df["ema_m"])
             & within(crossed_down(k, d) & (k.shift(1) > 70), 2)
             & (hist < hist.shift(1))
             & vol)
    return long_, short


STRATEGIES = {1: strategy_1, 2: strategy_2}


def add_signals(df: pd.DataFrame, strategy: int) -> pd.DataFrame:
    df = df.copy()
    long_, short = STRATEGIES[strategy](df)
    t = df.index.strftime("%H:%M")
    in_window = (t >= ENTRY_START) & (t <= ENTRY_END)
    df["signal"] = np.where(long_ & in_window, 1, np.where(short & in_window, -1, 0))
    return df


def build(df5_raw, df15_raw, strategy):
    df15 = add_trend(add_indicators(df15_raw))
    df5 = add_indicators(df5_raw)
    df5 = attach_15m_trend(df5, df15)
    return add_signals(df5, strategy), df15


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def backtest(df: pd.DataFrame, hold: int, cost_pct: float) -> pd.DataFrame:
    """Enter at next candle's open, exit at close after `hold` candles or end of day.
    One trade at a time, no overnight positions. cost_pct = round-trip cost in %."""
    sig = df["signal"].values
    o, c = df["Open"].values, df["Close"].values
    dates = df.index.date
    n, i, trades = len(df), 0, []
    cost = cost_pct / 100

    while i < n - 1:
        s = sig[i]
        if s == 0 or dates[i + 1] != dates[i]:
            i += 1
            continue
        entry = i + 1
        exit_ = min(entry + hold - 1, n - 1)
        while dates[exit_] != dates[entry]:
            exit_ -= 1
        ret = s * (c[exit_] / o[entry] - 1) - cost
        trades.append({
            "entry_time": df.index[entry], "exit_time": df.index[exit_],
            "side": "LONG" if s > 0 else "SHORT",
            "entry": round(o[entry], 2), "exit": round(c[exit_], 2),
            "ret_pct": round(ret * 100, 3),
        })
        i = exit_ + 1
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["ret_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    equity = r.cumsum()
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    return {
        "trades": len(r),
        "win_rate_%": round(100 * len(wins) / len(r), 1),
        "avg_win_%": round(wins.mean(), 3) if len(wins) else 0,
        "avg_loss_%": round(losses.mean(), 3) if len(losses) else 0,
        "profit_factor": round(pf, 2),
        "total_%": round(r.sum(), 2),
        "max_drawdown_%": round((equity - equity.cummax()).min(), 2),
    }


# --------------------------------------------------------------------------
# Telegram alerts
# --------------------------------------------------------------------------
def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("[alert] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set. Message would have been:\n" + text)
        return False
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=15)
        return True
    except Exception as e:
        print(f"[alert] Telegram send failed: {e}", file=sys.stderr)
        return False


def market_hours_now() -> bool:
    now = pd.Timestamp.now(tz=IST)
    return now.weekday() < 5 and "09:20" <= now.strftime("%H:%M") <= "16:00"


def load_state(path: str) -> set:
    try:
        with open(path) as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_state(path: str, keys: set) -> None:
    today = pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d")
    keep = sorted(k for k in keys if today in k)   # forget previous days
    with open(path, "w") as f:
        json.dump(keep, f)


def alert_text(sym, ts, row, strategy) -> str:
    side = "BUY" if row["signal"] > 0 else "SELL"
    icon = "🟢" if side == "BUY" else "🔴"
    name = {1: "Momentum", 2: "Pullback"}[strategy]
    vol = "n/a" if pd.isna(row["vol_ratio"]) else f"{row['vol_ratio']:.1f}x avg"
    return (
        f"{icon} {side} {sym}  (Strategy {strategy}: {name})\n"
        f"5m candle: {ts.strftime('%d-%b %H:%M')}  Close: {row['Close']:.2f}\n"
        f"15m trend: {row['trend15']}\n"
        f"EMA 5/13/26: {row['ema_f']:.1f} / {row['ema_m']:.1f} / {row['ema_s']:.1f}\n"
        f"MACD hist: {row['macd_hist']:.2f}   Stoch K/D: {row['stoch_k']:.0f}/{row['stoch_d']:.0f}\n"
        f"Volume: {vol}\n"
        f"Signal only, not advice. Check the chart before trading."
    )


def new_signals(sym, df5, strategy, sent: set, lookback_min: int = 90):
    """Signals on candles that closed in the last `lookback_min` minutes and weren't sent yet."""
    now = pd.Timestamp.now(tz=IST)
    recent = df5[(df5["signal"] != 0) &
                 (df5.index + pd.Timedelta(minutes=5) >= now - pd.Timedelta(minutes=lookback_min))]
    out = []
    for ts, row in recent.iterrows():
        key = f"{sym}|{strategy}|{ts.strftime('%Y-%m-%d %H:%M')}|{int(row['signal'])}"
        if key not in sent:
            out.append((key, alert_text(sym, ts, row, strategy)))
    return out


def snapshot_text(rows, strategy, n_requested) -> str:
    now = pd.Timestamp.now(tz=IST).strftime("%d-%b %H:%M")
    name = {1: "Momentum", 2: "Pullback"}[strategy]
    lines = [f"📊 Snapshot {now} IST  (Strategy {strategy}: {name})", ""]
    if not rows:
        lines.append("⚠️ No market data could be fetched. Check the GitHub Actions log.")
        return "\n".join(lines)
    trend_icon = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "⚪"}
    for r in rows:
        sig = "" if r["signal_5m"] == "-" else f"  ➜ {r['signal_5m']} NOW"
        recent = "" if r["last_30min"] == "-" else f"  (last 30m: {r['last_30min']})"
        lines.append(
            f"{trend_icon.get(r['trend_15m'], '')} {r['symbol']}  {r['close']}  "
            f"15m {r['trend_15m']}  Stoch {r['stoch_k/d']}  Vol {r['vol_x_avg']}x{sig}{recent}"
        )
    if len(rows) < n_requested:
        lines.append(f"\n({n_requested - len(rows)} symbol(s) had no data)")
    lines.append(f"\nCandle used: {rows[0]['candle']} (last completed 5m)")
    return "\n".join(lines)


def signals_today(sym, df5):
    today = pd.Timestamp.now(tz=IST).date()
    hits = df5[(df5["signal"] != 0) & (df5.index.date == today)]
    return [(sym, ts, "BUY" if r["signal"] > 0 else "SELL", r["Close"]) for ts, r in hits.iterrows()]


def summary_text(sigs, strategy, n_ok, n_requested) -> str:
    day = pd.Timestamp.now(tz=IST).strftime("%a %d-%b")
    name = {1: "Momentum", 2: "Pullback"}[strategy]
    lines = [f"📋 Day summary {day}  (Strategy {strategy}: {name})", ""]
    if n_ok == 0:
        lines.append("⚠️ No market data could be fetched today. Check the GitHub Actions log.")
    elif not sigs:
        lines.append(f"No signals today. Checked {n_ok} stocks; none met all the conditions.")
    else:
        lines.append(f"{len(sigs)} signal(s) today:")
        for sym, ts, side, close in sorted(sigs, key=lambda x: x[1]):
            icon = "🟢" if side == "BUY" else "🔴"
            lines.append(f"{icon} {ts.strftime('%H:%M')}  {side} {sym} @ {close:.2f}")
    if 0 < n_ok < n_requested:
        lines.append(f"\n({n_requested - n_ok} symbol(s) had no data)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def describe_last(sym, df5):
    last = df5.iloc[-1]
    sig = {1: "BUY", -1: "SELL", 0: "-"}[int(last["signal"])]
    recent = df5["signal"].iloc[-6:]
    recent_sig = recent[recent != 0]
    recent_txt = ""
    if not recent_sig.empty:
        t = recent_sig.index[-1].strftime("%H:%M")
        recent_txt = f"{'BUY' if recent_sig.iloc[-1] > 0 else 'SELL'} @ {t}"
    return {
        "symbol": sym,
        "candle": df5.index[-1].strftime("%d-%b %H:%M"),
        "close": round(last["Close"], 2),
        "trend_15m": last["trend15"],
        "signal_5m": sig,
        "last_30min": recent_txt or "-",
        "ema5/13/26": f"{last['ema_f']:.1f}/{last['ema_m']:.1f}/{last['ema_s']:.1f}",
        "macd_hist": round(last["macd_hist"], 2),
        "stoch_k/d": f"{last['stoch_k']:.0f}/{last['stoch_d']:.0f}",
        "vol_x_avg": "n/a" if pd.isna(last["vol_ratio"]) else round(last["vol_ratio"], 2),
    }


def main():
    p = argparse.ArgumentParser(description="NSE 5m/15m signal scanner and backtester")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                   help="NSE symbols, e.g. RELIANCE SBIN ^NSEI (default: 10 large caps)")
    p.add_argument("--strategy", type=int, choices=[1, 2], default=1,
                   help="1 = momentum crossover, 2 = pullback in trend")
    p.add_argument("--backtest", action="store_true", help="run a backtest instead of a live scan")
    p.add_argument("--hold", type=int, default=6, help="candles to hold a trade (6 x 5m = 30 min)")
    p.add_argument("--cost", type=float, default=0.05,
                   help="round-trip cost in %% incl. brokerage, STT, slippage (default 0.05)")
    p.add_argument("--csv", help="save backtest trades to this CSV file")
    p.add_argument("--demo", action="store_true", help="use synthetic data (no internet needed)")
    p.add_argument("--alert", action="store_true", help="send new signals to Telegram")
    p.add_argument("--test-alert", action="store_true", help="send one test message to Telegram and exit")
    p.add_argument("--summary", action="store_true",
                   help="send a summary of every signal that fired today")
    p.add_argument("--snapshot", action="store_true",
                   help="send the current status of every symbol to Telegram (manual check)")
    p.add_argument("--force", action="store_true", help="with --alert, run even outside market hours")
    p.add_argument("--state", default="sent_signals.json", help="file that remembers sent alerts")
    args = p.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    if args.test_alert:
        now = pd.Timestamp.now(tz=IST).strftime("%d-%b %H:%M")
        ok = send_telegram(f"✅ NSE signal agent is connected ({now} IST).\n"
                           f"Watching: {', '.join(args.symbols)}")
        print("Test message sent." if ok else "Test message NOT sent.")
        sys.exit(0 if ok else 1)

    sent = set()
    if args.alert:
        sent = load_state(args.state)
        if not args.force and not market_hours_now():
            print("Outside market hours (Mon-Fri 09:20-16:00 IST). Nothing to do.")
            save_state(args.state, sent)
            return

    rows, all_trades, alerts, today_sigs = [], [], [], []
    for n_sym, sym in enumerate(args.symbols):
        ysym = to_yahoo(sym)
        try:
            if args.demo:
                df5_raw, df15_raw = demo_data(seed=n_sym)
            else:
                period = "10d" if args.alert else "59d"   # alerts only need recent history
                df5_raw = fetch(ysym, "5m", period)
                df15_raw = fetch(ysym, "15m", period)
                if not args.backtest:
                    df5_raw = drop_forming_candle(df5_raw, 5)
                    df15_raw = drop_forming_candle(df15_raw, 15)
            if df5_raw.empty or df15_raw.empty:
                print(f"[skip] {ysym}: no data", file=sys.stderr)
                continue
            df5, _ = build(df5_raw, df15_raw, args.strategy)
        except Exception as e:  # keep scanning other symbols
            print(f"[error] {ysym}: {e}", file=sys.stderr)
            continue

        if args.backtest:
            tr = backtest(df5, args.hold, args.cost)
            if not tr.empty:
                tr.insert(0, "symbol", sym)
                all_trades.append(tr)
            rows.append({"symbol": sym, **summarize(tr)})
        else:
            rows.append(describe_last(sym, df5))
            if args.alert:
                alerts += new_signals(sym, df5, args.strategy, sent)
            if args.summary:
                today_sigs += signals_today(sym, df5)

    if args.alert:
        today = pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d")
        if not rows and f"datafail|{today}" not in sent:
            # warn once a day if every symbol failed (e.g. Yahoo blocking the server)
            if send_telegram("⚠️ NSE signal agent could not fetch market data today. Check the GitHub Actions log."):
                sent.add(f"datafail|{today}")
        for key, text in alerts:
            if send_telegram(text):
                sent.add(key)
        save_state(args.state, sent)
        print(f"{len(alerts)} new signal(s) found.")

    if args.summary:
        send_telegram(summary_text(today_sigs, args.strategy, len(rows), len(args.symbols)))

    if args.snapshot:
        send_telegram(snapshot_text(rows, args.strategy, len(args.symbols)))

    if not rows:
        print("No results. Check your internet connection or symbols.")
        return

    name = {1: "Momentum crossover", 2: "Pullback in trend"}[args.strategy]
    print(f"\nStrategy {args.strategy}: {name}" + ("  [DEMO DATA]" if args.demo else ""))

    if args.backtest:
        print(f"Hold {args.hold} x 5m candles, cost {args.cost}% per round trip\n")
        print(pd.DataFrame(rows).fillna("").to_string(index=False))
        if all_trades:
            trades = pd.concat(all_trades, ignore_index=True)
            print("\nAll symbols combined:")
            for k, v in summarize(trades).items():
                print(f"  {k:<15} {v}")
            if args.csv:
                trades.to_csv(args.csv, index=False)
                print(f"\nTrades saved to {args.csv}")
        print("\nReminder: ~60 days of data is a small sample. Paper-trade before going live.")
    else:
        print("15m = trend filter, 5m = entry. Signal shown for the last completed candle.\n")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
