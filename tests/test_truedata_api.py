"""
TrueData API Integration Tests
───────────────────────────────
Systematic tests of all API endpoints we need.
Run: python tests/test_truedata_api.py

Tests:
  1. REST Authentication (bearer token)
  2. Historical 1-min bars for NIFTY-I (index continuous futures)
  3. Historical ticks for NIFTY-I
  4. Symbol master (getAllSymbols for F&O)
  5. Expiry list for NIFTY
  6. Option chain symbols for NIFTY
  7. Historical bars for a specific option symbol (NIFTY YYMMDD strike CE)
  8. Last N bars / Last N ticks quick fetch
  9. TCP/WebSocket streaming (brief connect + subscribe test)
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta, date

from config.settings import (
    TRUEDATA_USER, TRUEDATA_PASSWORD,
    TD_AUTH_URL, TD_HISTORY_URL, TD_SYMBOL_MASTER_URL,
    TD_TCP_HOST, TD_TCP_PORT,
)

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"

_token = None


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_result(name, success, detail=""):
    status = PASS if success else FAIL
    print(f"  {status}  {name}")
    if detail:
        # Print first 500 chars of detail
        for line in detail[:500].split("\n")[:8]:
            print(f"         {line}")
        if len(detail) > 500:
            print(f"         ... ({len(detail)} chars total)")


# ═══════════════════════════════════════════════════════════════
# TEST 1: Authentication
# ═══════════════════════════════════════════════════════════════

def test_auth():
    global _token
    section("TEST 1: REST Authentication")

    print(f"  User: {TRUEDATA_USER}")
    print(f"  URL:  {TD_AUTH_URL}")

    try:
        resp = requests.post(TD_AUTH_URL, data={
            "username": TRUEDATA_USER,
            "password": TRUEDATA_PASSWORD,
            "grant_type": "password",
        }, timeout=15)

        test_result("HTTP status", resp.status_code == 200,
                     f"Status: {resp.status_code}")

        data = resp.json()
        _token = data.get("access_token")

        test_result("Got access_token", _token is not None,
                     f"Token: {_token[:50]}..." if _token else "No token")

        expires_in = data.get("expires_in")
        test_result("Token expiry info", expires_in is not None,
                     f"Expires in: {expires_in}s")

        print(f"\n  Full response keys: {list(data.keys())}")
        return True

    except Exception as e:
        test_result("Authentication", False, str(e))
        return False


# ═══════════════════════════════════════════════════════════════
# TEST 2: Historical 1-min bars for NIFTY-I
# ═══════════════════════════════════════════════════════════════

def _auth_header():
    return {"Authorization": f"Bearer {_token}"}


def _fmt_date(dt):
    """Format datetime to TrueData REST format: YYMMDDTHH:MM:SS"""
    return dt.strftime("%y%m%dT%H:%M:%S")


def test_historical_bars():
    section("TEST 2: Historical 1-min Bars (NIFTY-I)")

    if not _token:
        test_result("Skipped (no token)", False)
        return

    # Fetch last 5 trading days of 1-min bars
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=7)

    url = f"{TD_HISTORY_URL}/getbars"
    params = {
        "symbol": "NIFTY-I",
        "from": _fmt_date(from_dt),
        "to": _fmt_date(to_dt),
        "interval": "1min",
        "response": "csv",
        "bidask": "0",
    }

    print(f"  URL: {url}")
    print(f"  Symbol: NIFTY-I")
    print(f"  Range: {_fmt_date(from_dt)} to {_fmt_date(to_dt)}")

    try:
        resp = requests.get(url, params=params, headers=_auth_header(), timeout=30)
        test_result("HTTP status", resp.status_code == 200,
                     f"Status: {resp.status_code}")

        if resp.status_code != 200:
            print(f"  Response body: {resp.text[:300]}")
            return

        df = pd.read_csv(StringIO(resp.text))
        test_result("Got DataFrame", not df.empty,
                     f"Shape: {df.shape}, Columns: {list(df.columns)}")

        if not df.empty:
            print(f"\n  First 3 rows:")
            print(df.head(3).to_string(index=False))
            print(f"\n  Last 3 rows:")
            print(df.tail(3).to_string(index=False))
            print(f"\n  Total rows: {len(df)}")

    except Exception as e:
        test_result("Historical bars", False, str(e))


# ═══════════════════════════════════════════════════════════════
# TEST 3: Historical ticks for NIFTY-I
# ═══════════════════════════════════════════════════════════════

def test_historical_ticks():
    section("TEST 3: Historical Ticks (NIFTY-I)")

    if not _token:
        test_result("Skipped (no token)", False)
        return

    # Fetch ticks for a recent trading day (try yesterday, today)
    # Use a known market hour
    to_dt = datetime.now().replace(hour=15, minute=30, second=0)
    from_dt = to_dt.replace(hour=9, minute=15, second=0)

    # If it's before market hours or weekend, go back
    if to_dt.weekday() >= 5:  # Weekend
        days_back = to_dt.weekday() - 4
        to_dt -= timedelta(days=days_back)
        from_dt -= timedelta(days=days_back)

    url = f"{TD_HISTORY_URL}/getticks"
    params = {
        "symbol": "NIFTY-I",
        "from": _fmt_date(from_dt),
        "to": _fmt_date(to_dt),
        "response": "csv",
        "bidask": "1",
    }

    print(f"  Range: {_fmt_date(from_dt)} to {_fmt_date(to_dt)}")

    try:
        resp = requests.get(url, params=params, headers=_auth_header(), timeout=30)
        test_result("HTTP status", resp.status_code == 200,
                     f"Status: {resp.status_code}")

        if resp.status_code != 200:
            print(f"  Response: {resp.text[:300]}")
            return

        df = pd.read_csv(StringIO(resp.text))
        test_result("Got tick data", not df.empty,
                     f"Shape: {df.shape}, Columns: {list(df.columns)}")

        if not df.empty:
            print(f"\n  First 3 rows:")
            print(df.head(3).to_string(index=False))
            print(f"\n  Total ticks: {len(df)}")

    except Exception as e:
        test_result("Historical ticks", False, str(e))


# ═══════════════════════════════════════════════════════════════
# TEST 4: Symbol Master (getAllSymbols)
# ═══════════════════════════════════════════════════════════════

def test_symbol_master():
    section("TEST 4: Symbol Master (F&O)")

    url = f"{TD_SYMBOL_MASTER_URL}/getAllSymbols"
    params = {
        "segment": "fo",
        "user": TRUEDATA_USER,
        "password": TRUEDATA_PASSWORD,
        "csv": "true",
        "csvHeader": "true",
        "token": "true",
        "ticksize": "true",
        "underlying": "true",
        "search": "NIFTY",
        "limit": "100",
    }

    print(f"  URL: {url}")
    print(f"  Segment: fo, search: NIFTY")

    try:
        resp = requests.get(url, params=params, timeout=30)
        test_result("HTTP status", resp.status_code == 200,
                     f"Status: {resp.status_code}")

        if resp.status_code != 200:
            print(f"  Response: {resp.text[:500]}")
            return

        df = pd.read_csv(StringIO(resp.text))
        test_result("Got symbol master", not df.empty,
                     f"Shape: {df.shape}, Columns: {list(df.columns)}")

        if not df.empty:
            print(f"\n  First 5 rows:")
            print(df.head(5).to_string(index=False))
            print(f"\n  Total NIFTY F&O symbols: {len(df)}")

            # Check column names
            cols = [c.lower().strip() for c in df.columns]
            print(f"\n  Normalized columns: {cols}")

    except Exception as e:
        test_result("Symbol master", False, str(e))


# ═══════════════════════════════════════════════════════════════
# TEST 5: Expiry List for NIFTY
# ═══════════════════════════════════════════════════════════════

def test_expiry_list():
    section("TEST 5: NIFTY Expiry List")

    # Try both endpoints
    for host_name, base_url in [("api.truedata.in", TD_SYMBOL_MASTER_URL),
                                  ("history.truedata.in", TD_HISTORY_URL)]:
        print(f"\n  --- Via {host_name} ---")

        url = f"{base_url}/getSymbolExpiryList"

        if "api.truedata.in" in base_url:
            params = {
                "user": TRUEDATA_USER,
                "password": TRUEDATA_PASSWORD,
                "symbol": "NIFTY",
                "response": "csv",
            }
            headers = {}
        else:
            params = {
                "symbol": "NIFTY",
                "response": "csv",
            }
            headers = _auth_header()

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            test_result(f"HTTP status ({host_name})", resp.status_code == 200,
                         f"Status: {resp.status_code}")

            if resp.status_code == 200 and resp.text.strip():
                df = pd.read_csv(StringIO(resp.text))
                print(f"    Expiries found: {len(df)}")
                if not df.empty:
                    print(f"    Columns: {list(df.columns)}")
                    print(f"    First 5:\n{df.head(5).to_string(index=False)}")
            else:
                print(f"    Response: {resp.text[:200]}")

        except Exception as e:
            test_result(f"Expiry list ({host_name})", False, str(e))


# ═══════════════════════════════════════════════════════════════
# TEST 6: Option Chain Symbols
# ═══════════════════════════════════════════════════════════════

def _get_next_expiry():
    """Fetch actual next expiry from API instead of guessing."""
    url = f"{TD_SYMBOL_MASTER_URL}/getSymbolExpiryList"
    params = {
        "user": TRUEDATA_USER, "password": TRUEDATA_PASSWORD,
        "symbol": "NIFTY", "response": "csv",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.text))
            expiries = pd.to_datetime(df.iloc[:, 0]).dt.date.tolist()
            today = date.today()
            future = [e for e in expiries if e >= today]
            if future:
                return future[0]
    except Exception:
        pass
    return None


def test_option_chain():
    section("TEST 6: NIFTY Option Chain")

    next_expiry = _get_next_expiry()
    if next_expiry is None:
        test_result("Could not fetch expiry list", False)
        return
    expiry_str = next_expiry.strftime("%y%m%d")

    print(f"  Trying expiry: {next_expiry} (YYMMDD: {expiry_str}) [from API]")

    url = f"{TD_HISTORY_URL}/getSymbolOptionChain"
    params = {
        "symbol": "NIFTY",
        "expiry": expiry_str,
        "response": "csv",
    }

    try:
        resp = requests.get(url, params=params, headers=_auth_header(), timeout=15)
        test_result("HTTP status (history)", resp.status_code == 200,
                     f"Status: {resp.status_code}")

        if resp.status_code == 200 and resp.text.strip():
            df = pd.read_csv(StringIO(resp.text))
            test_result("Got option chain", not df.empty,
                         f"Shape: {df.shape}, Columns: {list(df.columns)}")
            if not df.empty:
                print(f"\n  First 5 rows:")
                print(df.head(5).to_string(index=False))
                print(f"\n  Total strikes: {len(df)}")
        else:
            print(f"  Response: {resp.text[:300]}")

    except Exception as e:
        test_result("Option chain", False, str(e))

    # Also try api.truedata.in with YYYYMMDD format
    print(f"\n  --- Also trying api.truedata.in (YYYYMMDD: {next_expiry.strftime('%Y%m%d')}) ---")
    url2 = f"{TD_SYMBOL_MASTER_URL}/getoptionchain"
    params2 = {
        "segment": "fo",
        "user": TRUEDATA_USER,
        "password": TRUEDATA_PASSWORD,
        "symbol": "NIFTY",
        "expiry": next_expiry.strftime("%Y%m%d"),
        "csv": "true",
    }

    try:
        resp2 = requests.get(url2, params=params2, timeout=15)
        test_result("HTTP status (api)", resp2.status_code == 200,
                     f"Status: {resp2.status_code}")
        if resp2.status_code == 200 and resp2.text.strip():
            df2 = pd.read_csv(StringIO(resp2.text))
            if not df2.empty:
                print(f"    Shape: {df2.shape}, Columns: {list(df2.columns)}")
                print(f"    First 3:\n{df2.head(3).to_string(index=False)}")
        else:
            print(f"    Response: {resp2.text[:200]}")
    except Exception as e:
        test_result("Option chain (api)", False, str(e))


# ═══════════════════════════════════════════════════════════════
# TEST 7: Historical bars for a specific option symbol
# ═══════════════════════════════════════════════════════════════

def test_option_symbol_bars():
    section("TEST 7: Option Symbol Historical Bars")

    if not _token:
        test_result("Skipped (no token)", False)
        return

    next_expiry = _get_next_expiry()
    if next_expiry is None:
        test_result("Could not fetch expiry", False)
        return

    # Get current NIFTY price from last tick to compute ATM
    try:
        resp = requests.get(f"{TD_HISTORY_URL}/getlastnticks",
            params={"symbol": "NIFTY-I", "nticks": "1", "interval": "tick",
                    "response": "csv", "bidask": "0"},
            headers=_auth_header(), timeout=10)
        ltp = float(pd.read_csv(StringIO(resp.text)).iloc[0]["ltp"])
        test_strike = round(ltp / 50) * 50  # Round to nearest 50 (NIFTY gap)
        print(f"  Current NIFTY LTP: {ltp}, ATM strike: {test_strike}")
    except Exception:
        test_strike = 23750
        print(f"  Using fallback strike: {test_strike}")
    yy = next_expiry.strftime("%y")
    mm = next_expiry.strftime("%m")
    dd = next_expiry.strftime("%d")
    opt_symbol = f"NIFTY{yy}{mm}{dd}{test_strike}CE"

    print(f"  Option symbol: {opt_symbol}")
    print(f"  (expiry: {next_expiry}, strike: {test_strike})")

    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=3)

    url = f"{TD_HISTORY_URL}/getbars"
    params = {
        "symbol": opt_symbol,
        "from": _fmt_date(from_dt),
        "to": _fmt_date(to_dt),
        "interval": "1min",
        "response": "csv",
    }

    try:
        resp = requests.get(url, params=params, headers=_auth_header(), timeout=30)
        test_result("HTTP status", resp.status_code == 200,
                     f"Status: {resp.status_code}")

        if resp.status_code == 200 and resp.text.strip():
            df = pd.read_csv(StringIO(resp.text))
            test_result("Got option bars", not df.empty,
                         f"Shape: {df.shape}")
            if not df.empty:
                print(f"    Columns: {list(df.columns)}")
                print(f"    First 3:\n{df.head(3).to_string(index=False)}")
                print(f"    Total: {len(df)} bars")
            else:
                print(f"    Empty result — symbol may not exist or no trading data")
                print(f"    Try different strike. Response: {resp.text[:200]}")
        else:
            print(f"  Response: {resp.text[:300]}")

    except Exception as e:
        test_result("Option bars", False, str(e))


# ═══════════════════════════════════════════════════════════════
# TEST 8: Last N Bars / Last N Ticks
# ═══════════════════════════════════════════════════════════════

def test_last_n():
    section("TEST 8: Last N Bars / Last N Ticks")

    if not _token:
        test_result("Skipped", False)
        return

    # Last N bars — try getlastnbars (lowercase) as in Postman
    for endpoint in ["getlastnbars", "getLastNBars"]:
        url = f"{TD_HISTORY_URL}/{endpoint}"
        params = {
            "symbol": "NIFTY-I",
            "interval": "1min",
            "response": "csv",
            "nbars": "10",
            "comp": "false",
            "bidask": "0",
        }

        try:
            resp = requests.get(url, params=params, headers=_auth_header(), timeout=15)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                test_result(f"Last 10 bars via /{endpoint}", not df.empty,
                             f"Rows: {len(df)}, Cols: {list(df.columns)}")
                if not df.empty:
                    print(f"    Latest: {df.iloc[-1].to_dict()}")
                break
            else:
                test_result(f"/{endpoint}", False, f"Status: {resp.status_code}")
        except Exception as e:
            test_result(f"/{endpoint}", False, str(e))

    # Last N ticks
    url2 = f"{TD_HISTORY_URL}/getlastnticks"
    params2 = {
        "symbol": "NIFTY-I",
        "bidask": "1",
        "response": "csv",
        "nticks": "5",
        "interval": "tick",
    }

    try:
        resp2 = requests.get(url2, params=params2, headers=_auth_header(), timeout=15)
        if resp2.status_code == 200:
            df2 = pd.read_csv(StringIO(resp2.text))
            test_result("Last 5 ticks (NIFTY-I)", not df2.empty,
                         f"Rows: {len(df2)}, Cols: {list(df2.columns)}")
            if not df2.empty:
                print(f"    Latest: {df2.iloc[-1].to_dict()}")
        else:
            test_result("Last N ticks", False,
                         f"Status: {resp2.status_code}, Body: {resp2.text[:200]}")
    except Exception as e:
        test_result("Last N ticks", False, str(e))


# ═══════════════════════════════════════════════════════════════
# TEST 9: TCP Streaming (brief connect test)
# ═══════════════════════════════════════════════════════════════

def test_wss_streaming():
    section("TEST 9: WebSocket Streaming (WSS)")

    print(f"  URL: wss://{TD_TCP_HOST}:{TD_TCP_PORT}")

    try:
        from data.truedata_adapter import TrueDataAdapter

        td = TrueDataAdapter()
        ok = td.ws_connect()
        test_result("WSS connect + auth", ok,
                     f"Metadata: {td._ws_metadata}" if ok else "Failed")

        if not ok:
            return

        # Subscribe to NIFTY 50 and capture snapshot
        snapshots = []
        td._callbacks.append(lambda t: snapshots.append(t))
        td.ws_subscribe(["NIFTY 50"])

        test_result("Subscribe + snapshot", len(snapshots) > 0,
                     f"Got {len(snapshots)} snapshot(s)")

        if snapshots:
            t = snapshots[0]
            print(f"    Symbol: {t['symbol']}")
            print(f"    LTP: {t['price']}")
            print(f"    Open: {t['open']}, High: {t['high']}, Low: {t['low']}")
            print(f"    Prev Close: {t['prev_close']}")
            print(f"    SymbolID: {t['symbol_id']}")

        td.ws_disconnect()
        test_result("WSS disconnect", True)

    except Exception as e:
        test_result("WSS streaming", False, str(e))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TrueData API Integration Tests")
    print("=" * 60)
    print(f"  User:     {TRUEDATA_USER}")
    print(f"  TCP Port: {TD_TCP_PORT}")
    print(f"  Time:     {datetime.now()}")

    # Run all tests sequentially
    auth_ok = test_auth()

    if auth_ok:
        test_historical_bars()
        test_historical_ticks()
        test_symbol_master()
        test_expiry_list()
        test_option_chain()
        test_option_symbol_bars()
        test_last_n()
        test_wss_streaming()
    else:
        print("\n  ⚠ Authentication failed. Skipping remaining tests.")

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60 + "\n")
