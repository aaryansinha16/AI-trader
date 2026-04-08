"""
TrueData Adapter
─────────────────
Provides two modes of data access based on the official TrueData API docs:

1. Historical data (REST)  – via history.truedata.in
   - getbars:       OHLCV bars (1m, 5m, eod)
   - getticks:      tick-level data with bid/ask/OI
   - getlastnbars:  last N bars (max 200)
   - getlastnticks: last N ticks (max 200)

2. Real-time streaming (TCP/WebSocket) – via tcp.truedata.in:7070
   - Touchline:  LTP, volume, OI, bid/ask per tick
   - 1-min bars: OHLCV streamed every minute

Authentication:
  - REST: POST https://auth.truedata.in/token → bearer token
  - TCP:  LOGIN username password

Rate limits:
  - REST: 1 request/second
  - TCP:  No explicit rate limit

Date format for REST: YYMMDDTHH:MM:SS (e.g., 250317T09:15:00)

TrueData Symbol Naming (confirmed):
  Index spot:        NIFTY 50, NIFTY BANK
  Continuous futures: NIFTY-I, BANKNIFTY-I
  Contract futures:  NIFTY26APRFUT
  Options:           SYMBOL+YYMMDD+STRIKE+CE/PE  e.g. NIFTY26032424500CE
"""

import json
import time
import threading
from datetime import datetime, timedelta
from io import StringIO
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests

from config.settings import (
    TRUEDATA_USER,
    TRUEDATA_PASSWORD,
    SYMBOLS,
    TD_AUTH_URL,
    TD_HISTORY_URL,
    TD_TCP_HOST,
    TD_TCP_PORT,
    TD_RATE_LIMIT_RPS,
    TD_INDEX_SYMBOLS,
    TD_INDEX_SPOT_SYMBOLS,
    TD_INDEX_FUTURES_SYMBOLS,
)
from utils.logger import get_logger

logger = get_logger("truedata")


class TrueDataAdapter:
    """
    TrueData API adapter using REST + WebSocket.

    REST (history.truedata.in):
      - Authentication via bearer token
      - Historical bars: timestamp,open,high,low,close,volume,oi
      - Historical ticks: timestamp,ltp,volume,oi,bid,bidqty,ask,askqty

    WebSocket (wss://push.truedata.in:8084):
      - Auth via URL params: ?user=X&password=Y
      - Subscribe: {"method":"addsymbol","symbols":["NIFTY 50"]}
      - Tick data: JSON arrays [symbol,symbolID,ts,LTP,tickvol,ATP,totalvol,O,H,L,prevclose,OI,prevOI,turnover,bid,bidqty,ask,askqty]
    """

    def __init__(self):
        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._ws = None  # websocket.WebSocketApp instance
        self._ws_connected: bool = False
        self._callbacks: List[Callable] = []
        self._streaming: bool = False
        self._stream_thread: Optional[threading.Thread] = None
        self._last_request_time: float = 0.0
        self._ws_metadata: Dict = {}  # segments, maxsymbols, validity, etc.
        self._subscribed_symbols: List[str] = []  # tracked for auto-reconnect
        self._symbol_id_map: Dict[str, str] = {}  # symbolID (str) -> symbol name

    # ── Authentication (REST) ───────────────────────────────────────────────

    def authenticate(self) -> bool:
        """
        Get bearer token from TrueData auth service.
        POST https://auth.truedata.in/token
        Body: grant_type=password&username=X&password=Y
        """
        if not TRUEDATA_USER or not TRUEDATA_PASSWORD:
            logger.error("TrueData credentials not configured.")
            return False

        # Reuse valid token
        if self._token and self._token_expires and datetime.now() < self._token_expires:
            return True

        try:
            resp = requests.post(
                TD_AUTH_URL,
                data={
                    "grant_type": "password",
                    "username": TRUEDATA_USER,
                    "password": TRUEDATA_PASSWORD,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            self._token = data.get("access_token")
            expires_in = int(data.get("expires_in", 86400))
            self._token_expires = datetime.now() + timedelta(seconds=expires_in - 300)

            logger.info(f"TrueData authenticated. Token valid until {self._token_expires}")
            return True

        except requests.RequestException as e:
            logger.error(f"TrueData authentication failed: {e}")
            return False

    @property
    def is_authenticated(self) -> bool:
        return (
            self._token is not None
            and self._token_expires is not None
            and datetime.now() < self._token_expires
        )

    def _auth_header(self) -> dict:
        """Build Authorization header for REST requests."""
        return {"Authorization": f"Bearer {self._token}"}

    def _rate_limit(self):
        """Enforce 1 request/second rate limit."""
        elapsed = time.time() - self._last_request_time
        wait = (1.0 / TD_RATE_LIMIT_RPS) - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    # ── REST: Date Format Helper ────────────────────────────────────────────

    @staticmethod
    def _fmt_date(dt: datetime) -> str:
        """
        Format datetime to TrueData REST format: YYMMDDTHH:MM:SS
        e.g., 2025-03-17 09:15:00 → 250317T09:15:00
        """
        return dt.strftime("%y%m%dT%H:%M:%S")

    # ── REST: Historical Bars ───────────────────────────────────────────────

    def fetch_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1min",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars via REST.

        GET https://history.truedata.in/getbars
          ?symbol=X&from=YYMMDDTHH:MM:SS&to=YYMMDDTHH:MM:SS
          &response=csv&interval=1min

        Response: timestamp,open,high,low,close,volume,oi

        Args:
            symbol: TrueData symbol (e.g., "NIFTY-I", "NIFTY24500CE")
            start: start datetime
            end: end datetime
            interval: 1min/2min/3min/5min/15min/30min/60min/eod
        """
        if not self.authenticate():
            return pd.DataFrame()

        self._rate_limit()

        url = f"{TD_HISTORY_URL}/getbars"
        params = {
            "symbol": symbol,
            "from": self._fmt_date(start),
            "to": self._fmt_date(end),
            "response": "csv",
            "interval": interval,
        }

        try:
            resp = requests.get(url, params=params, headers=self._auth_header(), timeout=30)
            resp.raise_for_status()

            df = pd.read_csv(StringIO(resp.text))
            if df.empty:
                return pd.DataFrame()

            df.columns = [c.strip().lower() for c in df.columns]

            rename_map = {
                "time": "timestamp",
                "openinterest": "oi",
            }
            df.rename(columns=rename_map, inplace=True)
            df["symbol"] = symbol
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            logger.info(f"Fetched {len(df)} {interval} bars for {symbol}")
            return df

        except requests.RequestException as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical_minute_bars(
        self,
        symbol: str,
        days: int = 180,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical 1-minute bars (convenience wrapper).
        Powers the Macro ML Model training.

        Automatically chunks requests into ~30-day windows because the
        TrueData REST API returns at most ~8000 rows per request.
        """
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(
            f"Fetching 1m bars for {symbol}: "
            f"{start_date.date()} → {end_date.date()} (chunked)"
        )

        chunks: list[pd.DataFrame] = []
        chunk_start = start_date
        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=30), end_date)
            df = self.fetch_historical_bars(symbol, chunk_start, chunk_end, "1min")
            if not df.empty:
                chunks.append(df)
            chunk_start = chunk_end

        if not chunks:
            return pd.DataFrame()

        combined = pd.concat(chunks, ignore_index=True)
        combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        combined.sort_values("timestamp", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        logger.info(
            f"Combined {len(chunks)} chunks → {len(combined)} bars for {symbol}"
        )
        return combined

    # ── REST: Historical Ticks ──────────────────────────────────────────────

    def fetch_historical_ticks(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: int = 5,
        bidask: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical tick data via REST.

        GET https://history.truedata.in/getticks
          ?symbol=X&bidask=1&from=YYMMDDTHH:MM:SS&to=YYMMDDTHH:MM:SS&response=csv

        Response: timestamp,ltp,volume,oi,bid,bidqty,ask,askqty

        Args:
            symbol: TrueData symbol
            start: start datetime (default: days ago)
            end: end datetime (default: now)
            days: fallback days if start not provided
            bidask: include bid/ask data
        """
        if not self.authenticate():
            return pd.DataFrame()

        end = end or datetime.now()
        start = start or (end - timedelta(days=days))

        self._rate_limit()

        url = f"{TD_HISTORY_URL}/getticks"
        params = {
            "symbol": symbol,
            "from": self._fmt_date(start),
            "to": self._fmt_date(end),
            "response": "csv",
            "bidask": "1" if bidask else "0",
        }

        try:
            resp = requests.get(url, params=params, headers=self._auth_header(), timeout=60)
            resp.raise_for_status()

            df = pd.read_csv(StringIO(resp.text))
            if df.empty:
                return pd.DataFrame()

            df.columns = [c.strip().lower() for c in df.columns]

            rename_map = {
                "ltp": "price",
                "bid": "bid_price",
                "ask": "ask_price",
                "bidqty": "bid_qty",
                "askqty": "ask_qty",
                "openinterest": "oi",
                "time": "timestamp",
            }
            df.rename(columns=rename_map, inplace=True)
            df["symbol"] = symbol
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            logger.info(f"Fetched {len(df)} ticks for {symbol}")
            return df

        except requests.RequestException as e:
            logger.error(f"Error fetching ticks for {symbol}: {e}")
            return pd.DataFrame()

    # ── REST: Last N Bars / Ticks ───────────────────────────────────────────

    def fetch_last_n_bars(
        self, symbol: str, n: int = 200, interval: str = "1min"
    ) -> pd.DataFrame:
        """
        Fetch last N bars (max 200).
        GET https://history.truedata.in/getlastnbars
          ?symbol=X&nbars=200&interval=1min&response=csv&bidask=0
        """
        if not self.authenticate():
            return pd.DataFrame()

        self._rate_limit()

        url = f"{TD_HISTORY_URL}/getlastnbars"
        params = {
            "symbol": symbol,
            "nbars": min(n, 200),
            "interval": interval,
            "response": "csv",
            "bidask": "0",
        }

        try:
            resp = requests.get(url, params=params, headers=self._auth_header(), timeout=15)
            resp.raise_for_status()

            df = pd.read_csv(StringIO(resp.text))
            if df.empty:
                return pd.DataFrame()

            df.columns = [c.strip().lower() for c in df.columns]
            df.rename(columns={"time": "timestamp", "openinterest": "oi"}, inplace=True)
            df["symbol"] = symbol
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            logger.info(f"Fetched last {len(df)} {interval} bars for {symbol}")
            return df

        except requests.RequestException as e:
            logger.error(f"Error fetching last N bars for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_last_n_ticks(
        self, symbol: str, n: int = 200, bidask: bool = True
    ) -> pd.DataFrame:
        """
        Fetch last N ticks (max 200).
        GET https://history.truedata.in/getlastnticks
          ?symbol=X&nticks=200&bidask=1&response=csv&interval=tick
        """
        if not self.authenticate():
            return pd.DataFrame()

        self._rate_limit()

        url = f"{TD_HISTORY_URL}/getlastnticks"
        params = {
            "symbol": symbol,
            "nticks": min(n, 200),
            "response": "csv",
            "interval": "tick",
            "bidask": "1" if bidask else "0",
        }

        try:
            resp = requests.get(url, params=params, headers=self._auth_header(), timeout=15)
            resp.raise_for_status()

            df = pd.read_csv(StringIO(resp.text))
            if df.empty:
                return pd.DataFrame()

            df.columns = [c.strip().lower() for c in df.columns]
            rename_map = {
                "ltp": "price",
                "bid": "bid_price",
                "ask": "ask_price",
                "bidqty": "bid_qty",
                "askqty": "ask_qty",
                "openinterest": "oi",
                "time": "timestamp",
            }
            df.rename(columns=rename_map, inplace=True)
            df["symbol"] = symbol
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            logger.info(f"Fetched last {len(df)} ticks for {symbol}")
            return df

        except requests.RequestException as e:
            logger.error(f"Error fetching last N ticks for {symbol}: {e}")
            return pd.DataFrame()

    # ── REST: Bhavcopy (EOD snapshot) ───────────────────────────────────────

    def fetch_bhavcopy(
        self, segment: str = "FO", date_str: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch EOD bhavcopy for a segment.
        GET https://history.truedata.in/getbhavcopy
          ?segment=FO&date=YYYY-MM-DD&response=csv
        """
        if not self.authenticate():
            return pd.DataFrame()

        self._rate_limit()
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")

        url = f"{TD_HISTORY_URL}/getbhavcopy"
        params = {
            "segment": segment,
            "date": date_str,
            "response": "csv",
        }

        try:
            resp = requests.get(url, params=params, headers=self._auth_header(), timeout=15)
            resp.raise_for_status()

            df = pd.read_csv(StringIO(resp.text))
            df.columns = [c.strip().lower() for c in df.columns]
            logger.info(f"Fetched bhavcopy for {segment} on {date_str}: {len(df)} rows")
            return df

        except requests.RequestException as e:
            logger.error(f"Error fetching bhavcopy: {e}")
            return pd.DataFrame()

    # ── WebSocket: Real-time Streaming ───────────────────────────────────────

    def ws_connect(self) -> bool:
        """
        Connect to TrueData WebSocket streaming service.
        URL: wss://push.truedata.in:8084?user=X&password=Y
        Auth is via URL query params (no separate login step).

        On success, receives JSON:
          {"success":true,"message":"TrueData Real Time Data Service",
           "segments":["FO","IND",""],"maxsymbols":50,
           "subscription":"tick","validity":"2026-04-18T00:00:00"}
        """
        if not TRUEDATA_USER or not TRUEDATA_PASSWORD:
            logger.error("TrueData credentials not configured.")
            return False

        try:
            import websocket

            ws_url = (
                f"wss://{TD_TCP_HOST}:{TD_TCP_PORT}"
                f"?user={TRUEDATA_USER}&password={TRUEDATA_PASSWORD}"
            )
            logger.info(f"Connecting to WebSocket: wss://{TD_TCP_HOST}:{TD_TCP_PORT}")

            self._ws = websocket.create_connection(ws_url, timeout=10)

            # Read initial auth response
            raw = self._ws.recv()
            msg = json.loads(raw)

            if msg.get("success"):
                self._ws_connected = True
                self._ws_metadata = msg
                logger.info(
                    f"WebSocket connected: segments={msg.get('segments')}, "
                    f"maxsymbols={msg.get('maxsymbols')}, "
                    f"subscription={msg.get('subscription')}, "
                    f"validity={msg.get('validity')}"
                )
                return True
            else:
                logger.error(f"WebSocket auth failed: {msg}")
                return False

        except ImportError:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
            return False
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._ws_connected = False
            return False

    def ws_subscribe(self, symbols: List[str]):
        """
        Subscribe to symbols on WebSocket stream.
        Send: {"method":"addsymbol","symbols":["NIFTY 50","NIFTY26032423750CE"]}

        Response includes current snapshot for each symbol:
          {"success":true,"message":"symbols added","symbolsadded":N,
           "symbollist":[[symbol,symbolID,ts,LTP,tickvol,ATP,totalvol,
                          O,H,L,prevclose,OI,prevOI,turnover,bid,bidqty,ask,askqty],...],
           "totalsymbolsubscribed":N}
        """
        if not self._ws_connected:
            logger.error("WebSocket not connected. Call ws_connect() first.")
            return

        # Track for reconnect — accumulate, don't replace, so NIFTY-I isn't lost
        # when dynamic ATM re-subscription calls ws_subscribe with only new options
        existing = set(self._subscribed_symbols)
        for s in symbols:
            if s not in existing:
                self._subscribed_symbols.append(s)
        msg = json.dumps({"method": "addsymbol", "symbols": symbols})
        self._ws.send(msg)
        logger.info(f"WebSocket subscribing to {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")

        # Read subscription confirmation
        try:
            raw = self._ws.recv()
            resp = json.loads(raw)
            if resp.get("success"):
                added = resp.get("symbolsadded", 0)
                total = resp.get("totalsymbolsubscribed", 0)
                logger.info(f"WebSocket subscribed: {added} added, {total} total")

                # Parse initial snapshots from symbollist and build symbolID map
                # symbollist format: [symbol, symbolID, ts, LTP, ...]
                for sym_data in resp.get("symbollist", []):
                    if isinstance(sym_data, (list, tuple)) and len(sym_data) >= 2:
                        sym_name = str(sym_data[0]).strip()
                        sym_id = str(sym_data[1]).strip()
                        if sym_name and sym_id:
                            self._symbol_id_map[sym_id] = sym_name
                    tick = self._parse_ws_tick(sym_data)
                    if tick:
                        for cb in self._callbacks:
                            try:
                                cb(tick)
                            except Exception as e:
                                logger.error(f"Callback error on snapshot: {e}")
                logger.info(f"Symbol ID map: {len(self._symbol_id_map)} entries")
            else:
                logger.warning(f"Subscribe response: {resp}")
        except Exception as e:
            logger.error(f"Error reading subscribe response: {e}")

    def ws_unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols."""
        if not self._ws_connected:
            return
        msg = json.dumps({"method": "removesymbol", "symbols": symbols})
        self._ws.send(msg)
        logger.info(f"WebSocket unsubscribed from {len(symbols)} symbols")

    def ws_start_streaming(self, callback: Callable):
        """
        Start receiving live ticks in a background thread.

        Callback receives a parsed tick dict:
          {symbol, symbol_id, timestamp, price, volume, atp, total_volume,
           open, high, low, prev_close, oi, prev_oi, turnover,
           bid_price, bid_qty, ask_price, ask_qty}

        The WebSocket sends:
          - Heartbeats: {"success":true,"message":"HeartBeat","timestamp":"..."}
          - Tick arrays: ["NIFTY 50","200000001","2026-03-18T16:41:01","23777.8",...]
          - JSON objects for status messages
        """
        if not self._ws_connected:
            logger.error("WebSocket not connected.")
            return

        # Deduplicate — don't add the same callback twice (happens on reconnect)
        if callback not in self._callbacks:
            self._callbacks.append(callback)

        # Stop any existing stream thread before starting a new one
        if self._streaming and self._stream_thread and self._stream_thread.is_alive():
            self._streaming = False
            self._stream_thread.join(timeout=3)

        self._streaming = True

        def _stream_loop():
            reconnect_delay = 5   # seconds between reconnect attempts
            max_reconnect_delay = 60
            subscribed_symbols = list(self._subscribed_symbols) if hasattr(self, '_subscribed_symbols') else []

            while self._streaming:
                try:
                    raw = self._ws.recv()
                    if not raw:
                        continue

                    msg = json.loads(raw)

                    # Heartbeat
                    if isinstance(msg, dict) and msg.get("message") == "HeartBeat":
                        logger.debug(f"Heartbeat: {msg.get('timestamp')}")
                        reconnect_delay = 5  # reset on successful heartbeat
                        continue

                    # Status/info messages
                    if isinstance(msg, dict) and "message" in msg:
                        logger.debug(f"WS message: {msg.get('message')}")
                        continue

                    # Live trade tick: {"trade": [symbolID, ts, LTP, LTQ, ATP, TTQ, O, H, L, prevclose, OI, prevOI, turnover, tag, bid_qty, bid, ask_qty, ask, ...]}
                    # Note: symbol name is NOT included — must look up from _symbol_id_map
                    if isinstance(msg, dict) and "trade" in msg:
                        trade = msg["trade"]
                        if isinstance(trade, (list, tuple)) and len(trade) >= 3:
                            sym_id = str(trade[0]).strip()
                            sym_name = self._symbol_id_map.get(sym_id, "")
                            if sym_name:
                                # Build a tick array in the standard format:
                                # [symbol, symbolID, ts, LTP, tickvol, ATP, totalvol, O, H, L, prevclose, OI, prevOI, turnover, bid, bidqty, ask, askqty]
                                tick_array = [
                                    sym_name,        # 0: symbol
                                    trade[0],        # 1: symbolID
                                    trade[1] if len(trade) > 1 else "",   # 2: timestamp
                                    trade[2] if len(trade) > 2 else 0,    # 3: LTP
                                    trade[3] if len(trade) > 3 else 0,    # 4: LTQ (tickvol)
                                    trade[4] if len(trade) > 4 else 0,    # 5: ATP
                                    trade[5] if len(trade) > 5 else 0,    # 6: TTQ (totalvol)
                                    trade[6] if len(trade) > 6 else 0,    # 7: open
                                    trade[7] if len(trade) > 7 else 0,    # 8: high
                                    trade[8] if len(trade) > 8 else 0,    # 9: low
                                    trade[9] if len(trade) > 9 else 0,    # 10: prevclose
                                    trade[10] if len(trade) > 10 else 0,  # 11: OI
                                    trade[11] if len(trade) > 11 else 0,  # 12: prevOI
                                    trade[12] if len(trade) > 12 else 0,  # 13: turnover
                                    trade[15] if len(trade) > 15 else 0,  # 14: bid_price
                                    trade[14] if len(trade) > 14 else 0,  # 15: bid_qty
                                    trade[17] if len(trade) > 17 else 0,  # 16: ask_price
                                    trade[16] if len(trade) > 16 else 0,  # 17: ask_qty
                                ]
                                tick = self._parse_ws_tick(tick_array)
                                if tick:
                                    reconnect_delay = 5
                                    for cb in self._callbacks:
                                        try:
                                            cb(tick)
                                        except Exception as e:
                                            logger.error(f"Callback error: {e}")
                            else:
                                logger.debug(f"Unknown symbolID in trade: {sym_id}")
                        continue

                    # Legacy format: plain JSON array tick (kept for backward compat)
                    if isinstance(msg, list) and len(msg) >= 4:
                        if isinstance(msg[0], list):
                            for item in msg:
                                tick = self._parse_ws_tick(item)
                                if tick:
                                    reconnect_delay = 5
                                    for cb in self._callbacks:
                                        try:
                                            cb(tick)
                                        except Exception as e:
                                            logger.error(f"Callback error: {e}")
                        else:
                            tick = self._parse_ws_tick(msg)
                            if tick:
                                reconnect_delay = 5
                                for cb in self._callbacks:
                                    try:
                                        cb(tick)
                                    except Exception as e:
                                        logger.error(f"Callback error: {e}")
                        continue

                    # Unknown message format — log for debugging
                    logger.debug(f"WS unknown msg type={type(msg).__name__}: {str(msg)[:100]}")

                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON message: {raw[:100]}")
                except Exception as e:
                    if not self._streaming:
                        break
                    logger.error(f"WebSocket stream error: {e}")
                    # ── Auto-reconnect ────────────────────────────────────
                    logger.info(f"Attempting WebSocket reconnect in {reconnect_delay}s...")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

                    try:
                        # Close old socket silently
                        try:
                            self._ws.close()
                        except Exception:
                            pass
                        self._ws_connected = False

                        # Reconnect
                        if self.ws_connect():
                            # Re-subscribe to same symbols
                            subscribed_symbols = list(self._subscribed_symbols) if hasattr(self, '_subscribed_symbols') else []
                            if subscribed_symbols:
                                self.ws_subscribe(subscribed_symbols)
                            logger.info(f"WebSocket reconnected, re-subscribed to {len(subscribed_symbols)} symbols.")
                            reconnect_delay = 5  # reset delay on success
                        else:
                            logger.error("WebSocket reconnect failed, will retry...")
                    except Exception as re:
                        logger.error(f"Reconnect attempt failed: {re}")

        self._stream_thread = threading.Thread(target=_stream_loop, daemon=True)
        self._stream_thread.start()
        logger.info("WebSocket streaming started in background thread.")

    def _parse_ws_tick(self, data) -> Optional[dict]:
        """
        Parse a tick from WebSocket data.

        Data arrives as a JSON array (from symbollist or live ticks):
          [symbol, symbolID, timestamp, LTP, tickvol, ATP, totalvol,
           open, high, low, prevclose, OI, prevOI, turnover,
           bid, bidqty, ask, askqty]

        Index: 0=symbol, 1=symbolID, 2=timestamp, 3=LTP, 4=tickvol,
               5=ATP, 6=totalvol, 7=open, 8=high, 9=low, 10=prevclose,
               11=OI, 12=prevOI, 13=turnover, 14=bid, 15=bidqty,
               16=ask, 17=askqty
        """
        if not isinstance(data, (list, tuple)) or len(data) < 4:
            return None

        def _float(val, default=0.0):
            try:
                return float(val) if val and str(val).strip() else default
            except (ValueError, TypeError):
                return default

        def _int(val, default=0):
            try:
                return int(float(val)) if val and str(val).strip() else default
            except (ValueError, TypeError):
                return default

        try:
            tick = {
                "symbol": str(data[0]).strip() if len(data) > 0 else "",
                "symbol_id": _int(data[1]) if len(data) > 1 else 0,
                "timestamp": pd.to_datetime(str(data[2]).strip()) if len(data) > 2 and data[2] else datetime.now(),
                "price": _float(data[3]) if len(data) > 3 else 0.0,
                "volume": _int(data[4]) if len(data) > 4 else 0,
                "atp": _float(data[5]) if len(data) > 5 else 0.0,
                "total_volume": _int(data[6]) if len(data) > 6 else 0,
                "open": _float(data[7]) if len(data) > 7 else 0.0,
                "high": _float(data[8]) if len(data) > 8 else 0.0,
                "low": _float(data[9]) if len(data) > 9 else 0.0,
                "prev_close": _float(data[10]) if len(data) > 10 else 0.0,
                "oi": _int(data[11]) if len(data) > 11 else 0,
                "prev_oi": _int(data[12]) if len(data) > 12 else 0,
                "turnover": _float(data[13]) if len(data) > 13 else 0.0,
                "bid_price": _float(data[14]) if len(data) > 14 else 0.0,
                "bid_qty": _int(data[15]) if len(data) > 15 else 0,
                "ask_price": _float(data[16]) if len(data) > 16 else 0.0,
                "ask_qty": _int(data[17]) if len(data) > 17 else 0,
            }
            return tick
        except Exception as e:
            logger.debug(f"Failed to parse WS tick: {e} | {str(data)[:100]}")
            return None

    def ws_stop_streaming(self):
        """Stop the streaming loop."""
        self._streaming = False
        if self._stream_thread:
            self._stream_thread.join(timeout=5)
        logger.info("WebSocket streaming stopped.")

    # ── WebSocket: Disconnect ─────────────────────────────────────────────

    def ws_disconnect(self):
        """Disconnect from WebSocket stream."""
        self._streaming = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._ws_connected = False
        self._ws = None
        logger.info("WebSocket disconnected.")

    @property
    def is_ws_connected(self) -> bool:
        return self._ws_connected

    # ── Backward-compat aliases ───────────────────────────────────────────
    # Old code may reference tcp_* methods. Redirect to ws_*.
    def tcp_connect(self) -> bool:
        return self.ws_connect()

    def tcp_subscribe(self, symbols: List[str]):
        return self.ws_subscribe(symbols)

    def tcp_start_streaming(self, callback: Callable):
        return self.ws_start_streaming(callback)

    def tcp_stop_streaming(self):
        return self.ws_stop_streaming()

    def tcp_disconnect(self):
        return self.ws_disconnect()

    @property
    def is_tcp_connected(self) -> bool:
        return self._ws_connected

    # ── Convenience: Fetch All Historical ───────────────────────────────────

    def fetch_all_historical(
        self,
        symbols: Optional[List[str]] = None,
        bar_days: int = 180,
        tick_days: int = 5,
    ) -> dict:
        """
        Fetch both minute bars and tick data for all given symbols.
        Returns {"minute_bars": DataFrame, "ticks": DataFrame}.
        """
        symbols = symbols or [TD_INDEX_SYMBOLS.get(s, s) for s in SYMBOLS]
        all_minutes = []
        all_ticks = []

        for symbol in symbols:
            minute_df = self.fetch_historical_minute_bars(symbol, days=bar_days)
            if not minute_df.empty:
                all_minutes.append(minute_df)

            tick_df = self.fetch_historical_ticks(symbol, days=tick_days)
            if not tick_df.empty:
                all_ticks.append(tick_df)

        return {
            "minute_bars": pd.concat(all_minutes, ignore_index=True) if all_minutes else pd.DataFrame(),
            "ticks": pd.concat(all_ticks, ignore_index=True) if all_ticks else pd.DataFrame(),
        }

    def disconnect(self):
        """Disconnect from all services."""
        self.ws_disconnect()
        self._token = None
        self._token_expires = None
        logger.info("TrueData fully disconnected.")
