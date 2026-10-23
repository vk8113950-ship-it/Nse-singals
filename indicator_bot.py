#!/usr/bin/env python3
"""
NSE indicator bot: every 30 minutes during market hours, sends a Telegram update
showing the indicator status of 20 large NSE stocks. No buy/sell calls.

Data: Upstox candle API (no login needed). Runs on GitHub Actions, started on time by
cron-job.org.

INDICATOR RULES (agreed with Benjamin, 23-Oct-2026)
  EMA 5/13/26    15m  green 5>13>26 | red 5<13<26 | white otherwise
  MACD 12/26     1h   green MACD line > 0 | red < 0   (last COMPLETED hour, NSE hours start 9:15)
  Stoch 14/4/3   15m  green %K > %D | red %K < %D
  Volume 20 SMA  15m  green: green candle & volume > SMA | red: red candle & volume > SMA | white: volume <= SMA
  VWAP           15m  green close > VWAP | red close < VWAP   (VWAP resets daily)
  RSI 13         15m  SHOWN, NOT COUNTED. green from a cross back UP through 30 until RSI reaches 50,
                      red from a cross back DOWN through 70 until RSI reaches 50, white otherwise
  Trendline      15m  info only (not counted): ⬆️ close breaks above the line through the last
                      two swing highs, ⬇️ close breaks below the line through the last two swing lows,
                      within the last 30 minutes
  Stock colour   🟢 all 5 counted (EMA, MACD, STO, VOL, VWAP) green | 🔴 all 5 red | 🟠 anything else

MODES
  --mode scheduled          cron-job.org, every :16/:46 + 15:31. Sends only if a 30-min slot just closed
  --mode scheduled-summary  cron-job.org, 16:10. Day summary on trading days only
  --mode update-now         manual: latest update right now
  --mode summary-now        manual: day summary for today (or the last trading day)
  --mode test               manual: Telegram connection test
  --demo                    use synthetic data (offline testing)
"""

import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"

SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "SBIN", "AXISBANK", "KOTAKBANK", "LT", "BHARTIARTL",
    "ITC", "HINDUNILVR", "BAJFINANCE", "MARUTI", "M&M",
    "SUNPHARMA", "HCLTECH", "NTPC", "TITAN", "ULTRACEMCO",
]

# NSE equity trading holidays (weekdays only). Update every December for the next year.
HOLIDAYS = {
    2026: [
        "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31",
        "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
        "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24",
        "2026-12-25",
    ],
}

# Update times = closes of 30-minute candles counted from the 9:15 open, plus the 15:30 close.
SLOTS = ["09:45", "10:15", "10:45", "11:15", "11:45", "12:15", "12:45",
         "13:15", "13:45", "14:15", "14:45", "15:15", "15:30"]
SLOT_WINDOW_MIN = 12          # a scheduled run sends only if a slot closed in the last 12 min
SUMMARY_WINDOW = ("15:45", "17:30")

INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
API = "https://api.upstox.com/v3/historical-candle"

G, R, W, O = "🟢", "🔴", "⚪", "🟠"
SEND_FAILURES = []


# ----------------------------------------------------------------------------
# Calendar
# ----------------------------------------------------------------------------
def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


def is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in HOLIDAYS.get(d.year, [])


def last_trading_day(d: dt.date) -> dt.date:
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def at(d: dt.date, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{d.isoformat()} {hhmm}", tz=IST)


def scheduled_slot(now: pd.Timestamp):
    """The slot that closed within the last SLOT_WINDOW_MIN minutes, else None."""
    for s in SLOTS:
        t = at(now.date(), s)
        if t <= now < t + pd.Timedelta(minutes=SLOT_WINDOW_MIN):
            return t
    return None


def latest_slot(now: pd.Timestamp) -> pd.Timestamp:
    """Most recent completed slot: today if the market has passed 9:45, else last trading day's close."""
    d = now.date()
    if is_trading_day(d):
        done = [at(d, s) for s in SLOTS if at(d, s) <= now]
        if done:
            return done[-1]
    prev = last_trading_day(d - dt.timedelta(days=1))
    return at(prev, "15:30")


def holiday_list_missing(year: int) -> bool:
    return year not in HOLIDAYS


# ----------------------------------------------------------------------------
# Data (Upstox)
# ----------------------------------------------------------------------------
def http_get(url: str, raw: bool = False, tries: int = 3):
    last_err = None
    for i in range(tries):
        req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": "nse-indicator-bot"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                return body if raw else json.loads(body)
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:150]}"
            if e.code in (400, 401, 403, 404):
                break
        except Exception as e:
            last_err = str(e)[:150]
        time.sleep(2 * (i + 1))
    raise RuntimeError(last_err)


def load_instrument_keys() -> dict:
    rows = json.loads(gzip.decompress(http_get(INSTRUMENTS_URL, raw=True)))
    return {r["trading_symbol"]: r["instrument_key"] for r in rows
            if r.get("segment") == "NSE_EQ" and r.get("instrument_type") == "EQ"
            and r.get("trading_symbol") in SYMBOLS}


def candles_to_df(candles) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame([c[:6] for c in candles], columns=["time", "Open", "High", "Low", "Close", "Volume"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(IST)
    return df.set_index("time").astype(float)


def fetch_15m(key: str, today: dt.date) -> pd.DataFrame:
    """~4 weeks of 15m candles (history + today), oldest first, duplicates removed."""
    k = urllib.parse.quote(key, safe="")
    frm = today - dt.timedelta(days=28)
    hist = http_get(f"{API}/{k}/minutes/15/{today}/{frm}")["data"]["candles"]
    intra = http_get(f"{API}/intraday/{k}/minutes/15")["data"]["candles"]
    df = pd.concat([candles_to_df(hist), candles_to_df(intra)])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.between_time("09:15", "15:29")


def demo_15m(seed: int, end_day: dt.date) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = [d for d in pd.bdate_range(end=end_day, periods=20) if is_trading_day(d.date())]
    idx = []
    for d in days:
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
# Indicators
# ----------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=13):
    """Wilder's RSI, seeded with a simple average (same method as TradingView)."""
    c = close.values.astype(float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return pd.Series(out, index=close.index)
    d = np.diff(c)
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[:n].mean(), l[:n].mean()
    to_rsi = lambda ag, al: 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    out[n] = to_rsi(ag, al)
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + l[i]) / n
        out[i + 1] = to_rsi(ag, al)
    return pd.Series(out, index=close.index)


def rsi_state(r: pd.Series) -> int:
    """Option A: red after crossing down through 70 until RSI reaches 50 (or goes back above 70);
    green after crossing up through 30 until RSI reaches 50 (or goes back below 30)."""
    state, prev = 0, np.nan
    for v in r.values:
        if np.isnan(v):
            prev = v
            continue
        if state == -1 and (v <= 50 or v > 70):
            state = 0
        elif state == 1 and (v >= 50 or v < 30):
            state = 0
        if not np.isnan(prev):
            if prev > 70 and v <= 70:
                state = -1
            elif prev < 30 and v >= 30:
                state = 1
        prev = v
    return state


def hourly(df15: pd.DataFrame) -> pd.DataFrame:
    """NSE hourly candles starting 9:15 (9:15-10:15 ... 14:15-15:15, 15:15-15:30)."""
    t = df15.index
    day_open = t.normalize() + pd.Timedelta(hours=9, minutes=15)
    start = day_open + ((t - day_open) // pd.Timedelta(hours=1)) * pd.Timedelta(hours=1)
    g = df15.groupby(start)
    h = g.agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
              Close=("Close", "last"), Volume=("Volume", "sum"))
    h["end"] = [min(s + pd.Timedelta(hours=1), s.normalize() + pd.Timedelta(hours=15, minutes=30))
                for s in h.index]
    return h


def pivots(values: np.ndarray, k: int, kind: str):
    out = []
    for i in range(k, len(values) - k):
        win = values[i - k:i + k + 1]
        if (kind == "high" and values[i] == win.max()) or (kind == "low" and values[i] == win.min()):
            if not out or i - out[-1] > k:
                out.append(i)
    return out


def trendline_break(df: pd.DataFrame, k: int = 3, lookback: int = 75, recent: int = 2) -> str:
    """⬆️ / ⬇️ if the close crossed the swing-high / swing-low line in the last `recent` candles."""
    d = df.iloc[-lookback:]
    hi, lo, c = d["High"].values, d["Low"].values, d["Close"].values
    n = len(d)
    marks = []
    for kind, vals, sign in (("high", hi, 1), ("low", lo, -1)):
        p = [i for i in pivots(vals, k, kind) if i < n - recent - 1]
        if len(p) < 2:
            continue
        i1, i2 = p[-2], p[-1]
        slope = (vals[i2] - vals[i1]) / (i2 - i1)
        line = lambda x: vals[i1] + slope * (x - i1)
        for j in range(n - recent, n):
            if sign == 1 and c[j - 1] <= line(j - 1) and c[j] > line(j):
                marks.append("⬆️")
                break
            if sign == -1 and c[j - 1] >= line(j - 1) and c[j] < line(j):
                marks.append("⬇️")
                break
    return "".join(marks)


def colour(cond_green, cond_red):
    if cond_green:
        return G
    if cond_red:
        return R
    return W


def evaluate(df15_all: pd.DataFrame, upto: pd.Timestamp) -> dict:
    """Indicator statuses using only candles completed by `upto`."""
    df = df15_all[df15_all.index + pd.Timedelta(minutes=15) <= upto]
    if len(df) < 60:
        raise RuntimeError(f"not enough candles ({len(df)})")
    last = df.iloc[-1]
    c = df["Close"]

    e5, e13, e26 = ema(c, 5).iloc[-1], ema(c, 13).iloc[-1], ema(c, 26).iloc[-1]
    ema_s = colour(e5 > e13 > e26, e5 < e13 < e26)

    data_end = df.index[-1] + pd.Timedelta(minutes=15)
    h = hourly(df)
    h = h[h["end"] <= min(upto, data_end)]
    macd_line = ema(h["Close"], 12) - ema(h["Close"], 26)
    m = macd_line.iloc[-1]
    macd_s = colour(m > 0, m < 0)

    ll, hh = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    k = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(4).mean()
    d = k.rolling(3).mean()
    kv, dv = k.iloc[-1], d.iloc[-1]
    sto_s = W if np.isnan(kv) or np.isnan(dv) else colour(kv > dv, kv < dv)

    sma = df["Volume"].rolling(20).mean().iloc[-1]
    high_vol = last["Volume"] > sma
    vol_s = colour(high_vol and last["Close"] > last["Open"], high_vol and last["Close"] < last["Open"])

    today = df[df.index.date == df.index[-1].date()]
    tp = (today["High"] + today["Low"] + today["Close"]) / 3
    vwap = (tp * today["Volume"]).cumsum().iloc[-1] / today["Volume"].cumsum().iloc[-1]
    vwap_s = colour(last["Close"] > vwap, last["Close"] < vwap)

    r = rsi(c, 13)
    st = rsi_state(r)
    rsi_s = G if st == 1 else R if st == -1 else W

    parts = {"EMA": ema_s, "MACD": macd_s, "STO": sto_s, "VOL": vol_s, "VWAP": vwap_s}
    vals = list(parts.values())          # RSI is shown but not counted (decided 23-Oct-2026)
    stock = G if all(v == G for v in vals) else R if all(v == R for v in vals) else O
    return {
        "parts": parts, "rsi": rsi_s, "stock": stock, "close": last["Close"], "tl": trendline_break(df),
        "candle": df.index[-1], "hour": h.index[-1], "hour_end": h["end"].iloc[-1],
        "values": {"ema": (e5, e13, e26), "macd": m, "k": kv, "d": dv, "vol": last["Volume"],
                   "vol_sma": sma, "vwap": vwap, "rsi": r.iloc[-1]},
    }


# ----------------------------------------------------------------------------
# Telegram
# ----------------------------------------------------------------------------
def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        print("[telegram] secrets not set. Message would have been:\n" + text)
        SEND_FAILURES.append(1)
        return False
    ok = True
    for chunk in split_message(text):
        data = urllib.parse.urlencode({"chat_id": chat, "text": chunk}).encode()
        try:
            urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=15)
            print("[telegram] sent:\n" + chunk)
        except urllib.error.HTTPError as e:
            print(f"[telegram] FAILED ({e.code}): {e.read().decode(errors='replace')}", file=sys.stderr)
            SEND_FAILURES.append(1)
            ok = False
        except Exception as e:
            print(f"[telegram] FAILED: {e}", file=sys.stderr)
            SEND_FAILURES.append(1)
            ok = False
    return ok


def split_message(text: str, limit: int = 3500):
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


# ----------------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------------
ORDER = {G: 0, R: 1, O: 2}


def fmt_price(p):
    return f"{p:,.1f}"


def update_text(slot: pd.Timestamp, results: dict, errors: dict) -> str:
    any_r = next(iter(results.values()))
    c0 = any_r["candle"]
    lines = [
        f"📊 {slot.strftime('%H:%M')} update · {slot.strftime('%a %d-%b')}",
        f"15m candle {c0.strftime('%H:%M')}–{(c0 + pd.Timedelta(minutes=15)).strftime('%H:%M')} · "
        f"MACD 1h {any_r['hour'].strftime('%H:%M')}–{any_r['hour_end'].strftime('%H:%M')}"
        + ("" if any_r["hour"].date() == slot.date() else f" ({any_r['hour'].strftime('%d-%b')})"),
        "",
    ]
    for sym, r in sorted(results.items(), key=lambda kv: (ORDER[kv[1]["stock"]], SYMBOLS.index(kv[0]))):
        p = r["parts"]
        lines.append(f"{r['stock']} {sym} {fmt_price(r['close'])}  "
                     + " ".join(f"{k}{v}" for k, v in p.items())
                     + f" · RSI{r['rsi']}"
                     + (f" {r['tl']}" if r["tl"] else ""))
    counts = {c: sum(1 for r in results.values() if r["stock"] == c) for c in (G, R, O)}
    lines += ["", f"Summary: {G} {counts[G]}  {R} {counts[R]}  {O} {counts[O]}",
              "Colour = EMA·MACD·STO·VOL·VWAP. RSI, ⬆️ breakout, ⬇️ breakdown shown only."]
    if errors:
        lines.append(f"⚠️ No data for: {', '.join(errors)}")
    if holiday_list_missing(slot.year):
        lines.append(f"📅 NSE {slot.year} holiday list missing in the bot. Ask Claude to add it.")
    return "\n".join(lines)


def summary_text(day: dt.date, data: dict, errors: dict, now: pd.Timestamp) -> str:
    data_end = max(df.index[-1] for df in data.values()) + pd.Timedelta(minutes=15)
    slots = [at(day, s) for s in SLOTS if at(day, s) <= min(now, data_end)]
    head = "close" if len(slots) == len(SLOTS) else f"{slots[-1].strftime('%H:%M')}" if slots else "-"
    lines = [f"📋 Day summary · {day.strftime('%a %d-%b')}",
             f"Colour at {head} · how many of today's {len(slots)} updates were 🟢/🔴/🟠", ""]
    rows = []
    for sym, df in data.items():
        cols = []
        for s in slots:
            try:
                cols.append(evaluate(df, s)["stock"])
            except Exception:
                pass
        if cols:
            rows.append((sym, cols[-1], cols.count(G), cols.count(R), cols.count(O)))
    for sym, close_c, g, r, o in sorted(rows, key=lambda x: (ORDER[x[1]], SYMBOLS.index(x[0]))):
        lines.append(f"{close_c} {sym}  {G}{g} {R}{r} {O}{o}")
    if errors:
        lines.append(f"\n⚠️ No data for: {', '.join(errors)}")
    nxt = day.year + 1
    if day.month == 12 and holiday_list_missing(nxt):
        lines.append(f"\n📅 Reminder: add NSE {nxt} holidays to the bot before 1 Jan. "
                     f"Ask Claude to update the holiday list.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def load_all(day: dt.date, demo: bool):
    data, errors = {}, {}
    keys = {} if demo else load_instrument_keys()
    for i, sym in enumerate(SYMBOLS):
        try:
            if demo:
                df = demo_15m(i, day)
            else:
                if sym not in keys:
                    raise RuntimeError("instrument key not found")
                df = fetch_15m(keys[sym], day)
            if df.empty:
                raise RuntimeError("no candles")
            data[sym] = df
        except Exception as e:
            errors[sym] = str(e)
            print(f"[data] {sym}: {e}", file=sys.stderr)
    return data, errors


def run_update(slot: pd.Timestamp, demo: bool, silent_if_closed: bool):
    data, errors = load_all(slot.date(), demo)
    if not data:
        raise RuntimeError("could not fetch data for any stock: " + next(iter(errors.values()), "?"))
    has_today = any((df.index.date == slot.date()).any() for df in data.values())
    if not has_today and silent_if_closed:
        print(f"No candles for {slot.date()}: market closed (unlisted holiday?). Staying silent.")
        return
    results = {}
    for sym, df in data.items():
        try:
            results[sym] = evaluate(df, slot)
        except Exception as e:
            errors[sym] = str(e)
    if not results:
        raise RuntimeError("indicator calculation failed for every stock")
    send_telegram(update_text(slot, results, errors))


def run_summary(day: dt.date, demo: bool, silent_if_closed: bool, now: pd.Timestamp):
    data, errors = load_all(day, demo)
    if not data:
        raise RuntimeError("could not fetch data for any stock")
    if silent_if_closed and not any((df.index.date == day).any() for df in data.values()):
        print("No candles today; skipping summary.")
        return
    send_telegram(summary_text(day, data, errors, now))


def main():
    p = argparse.ArgumentParser(description="NSE indicator bot")
    p.add_argument("--mode", default="update-now",
                   choices=["scheduled", "scheduled-summary", "update-now", "summary-now", "test"])
    p.add_argument("--demo", action="store_true", help="synthetic data, no internet")
    p.add_argument("--now", help="pretend current IST time, e.g. '2026-10-23 10:16' (testing)")
    args = p.parse_args()

    now = pd.Timestamp(args.now, tz=IST) if args.now else now_ist()
    today = now.date()
    label = args.mode

    try:
        if args.mode == "test":
            send_telegram(f"✅ NSE indicator bot connected ({now.strftime('%d-%b %H:%M')} IST).\n"
                          f"Watching {len(SYMBOLS)} stocks.")
        elif args.mode == "scheduled":
            if not is_trading_day(today):
                print(f"{today} is not a trading day. Nothing to do.")
                return
            slot = scheduled_slot(now)
            if slot is None:
                print(f"{now.strftime('%H:%M')}: no 30-min candle just closed. Nothing to do.")
                return
            label = f"{slot.strftime('%H:%M')} update"
            run_update(slot, args.demo, silent_if_closed=True)
        elif args.mode == "update-now":
            slot = latest_slot(now)
            label = f"{slot.strftime('%H:%M')} update"
            run_update(slot, args.demo, silent_if_closed=False)
        elif args.mode == "scheduled-summary":
            hhmm = now.strftime("%H:%M")
            if not is_trading_day(today) or not (SUMMARY_WINDOW[0] <= hhmm <= SUMMARY_WINDOW[1]):
                print("Not a trading day or outside the summary window. Nothing to do.")
                return
            label = "day summary"
            run_summary(today, args.demo, silent_if_closed=True, now=now)
        elif args.mode == "summary-now":
            day = today if is_trading_day(today) and now >= at(today, "09:45") \
                else last_trading_day(today - dt.timedelta(days=1))
            label = "day summary"
            run_summary(day, args.demo, silent_if_closed=False, now=now)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        send_telegram(f"⚠️ NSE bot: {label} failed ({now.strftime('%d-%b %H:%M')} IST)\n"
                      f"Reason: {str(e)[:300]}\nIt will try again at the next update.")
        sys.exit(1)


if __name__ == "__main__":
    main()
    if SEND_FAILURES:
        print(f"\n{len(SEND_FAILURES)} Telegram message(s) failed to send.", file=sys.stderr)
        sys.exit(1)
