#!/usr/bin/env python3
"""
One-off test: can GitHub's servers get 30-minute candles from Upstox for our 20 stocks?

Checks, for every stock:
  1. its Upstox instrument key (from Upstox's public instrument list)
  2. today's 30-min candles      (intraday API)
  3. the last ~2 weeks of 30-min candles (historical API, needed to warm up indicators)
First WITHOUT any login. If Upstox refuses (401/403) and an UPSTOX_ACCESS_TOKEN secret
exists, it retries with the token. Results are printed and sent to Telegram.
No extra packages needed.
"""
import datetime as dt
import gzip
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "SBIN", "AXISBANK", "KOTAKBANK", "LT", "BHARTIARTL",
    "ITC", "HINDUNILVR", "BAJFINANCE", "MARUTI", "M&M",
    "SUNPHARMA", "HCLTECH", "NTPC", "TITAN", "ULTRACEMCO",
]
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
API = "https://api.upstox.com/v3/historical-candle"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def http_get(url, token=None, raw=False):
    headers = {"Accept": "application/json", "User-Agent": "nse-signal-agent-test"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
            return r.status, (body if raw else json.loads(body))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:200]
    except Exception as e:
        return 0, str(e)[:200]


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        print("(Telegram secrets not set, message not sent)")
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=15)
        print("Telegram message sent.")
    except Exception as e:
        print(f"Telegram send failed: {e}")


def load_keys():
    status, body = http_get(INSTRUMENTS_URL, raw=True)
    if status != 200:
        return None, f"instrument list download failed ({status}): {body}"
    rows = json.loads(gzip.decompress(body))
    keys = {}
    for r in rows:
        if r.get("segment") == "NSE_EQ" and r.get("instrument_type") == "EQ" \
                and r.get("trading_symbol") in SYMBOLS:
            keys[r["trading_symbol"]] = r["instrument_key"]
    return keys, None


def candles(url, token):
    status, body = http_get(url, token)
    if status == 200 and isinstance(body, dict):
        c = body.get("data", {}).get("candles", [])
        last = max((x[0] for x in c), default="-")
        return status, len(c), last
    return status, 0, str(body)[:80]


def run(keys, token):
    today = dt.datetime.now(IST).date()
    frm = today - dt.timedelta(days=15)
    results = []
    for sym in SYMBOLS:
        key = keys.get(sym)
        if not key:
            results.append((sym, "no key", "", ""))
            continue
        k = urllib.parse.quote(key, safe="")
        s1, n1, last1 = candles(f"{API}/intraday/{k}/minutes/30", token)
        s2, n2, _ = candles(f"{API}/{k}/minutes/30/{today}/{frm}", token)
        results.append((sym, s1, f"{n1} today (last {last1[11:16] if s1 == 200 else last1})",
                        f"{n2} past 2 wks" if s2 == 200 else f"history HTTP {s2}"))
    return results


def main():
    now = dt.datetime.now(IST).strftime("%a %d-%b %H:%M")
    keys, err = load_keys()
    if err:
        msg = f"🧪 Upstox test {now} IST\n❌ {err}"
        print(msg); send_telegram(msg); sys.exit(1)

    mode = "no login"
    results = run(keys, None)
    refused = all(r[1] in (401, 403) for r in results if r[1] != "no key")
    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if refused and token:
        mode = "a login token"
        results = run(keys, token)

    ok = sum(1 for r in results if r[1] == 200)
    lines = [f"🧪 Upstox test {now} IST ({mode})",
             f"Instrument keys found: {len(keys)}/{len(SYMBOLS)}",
             f"Stocks with data: {ok}/{len(SYMBOLS)}", ""]
    for sym, status, today_txt, hist_txt in results:
        icon = "✅" if status == 200 else "❌"
        detail = f"{today_txt}, {hist_txt}" if status == 200 else (status if status == "no key" else f"HTTP {status}")
        lines.append(f"{icon} {sym}: {detail}")
    if ok == len(SYMBOLS):
        lines.append("\nRESULT: Upstox works from GitHub (" + mode + "). Ready to build.")
    elif refused and not token:
        lines.append("\nRESULT: Upstox now requires login. We'll use the daily-login option.")
    else:
        lines.append("\nRESULT: Partly working. Send this message to Claude.")
    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
