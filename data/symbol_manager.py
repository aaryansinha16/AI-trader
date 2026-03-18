"""
Symbol Manager
──────────────
Handles dynamic option symbol resolution for TrueData:

  1. Fetches F&O symbol master from TrueData API
  2. Computes ATM strike from current spot price
  3. Generates ATM ±N strike symbols for CE + PE
  4. Tracks nearest expiry (weekly rolling)
  5. Maintains symbol ↔ symbolID mapping

TrueData symbol naming conventions (confirmed by TrueData team):
  Equity:                   RELIANCE, TCS, etc. (same as NSE)
  Index spot:               NIFTY 50, NIFTY BANK, INDIAVIX
  Continuous futures:       NIFTY-I, NIFTY-II, BANKNIFTY-I
  Contract futures:         NIFTY26APRFUT, BANKNIFTY26MAYFUT
  Options (weekly/monthly): SYMBOL + YY + MM + DD + STRIKE + CE/PE
                            e.g. NIFTY26032424500CE  (NIFTY, 2026-03-24 expiry, 24500 strike, CE)
                            e.g. BANKNIFTY26032752000PE

Key design:
  - Each option strike is a SEPARATE symbol
  - NIFTY strikes gap = 50, BANKNIFTY = 100
  - ATM ±3 × CE/PE = 14 symbols per expiry + 1 index = ~15 for NIFTY only
  - Fits well within 50-symbol plan
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import math

import pandas as pd
import requests

from config.settings import (
    SYMBOLS,
    TRUEDATA_USER,
    TRUEDATA_PASSWORD,
    TD_SYMBOL_MASTER_URL,
    TD_INDEX_SYMBOLS,
    TD_INDEX_SPOT_SYMBOLS,
    TD_INDEX_FUTURES_SYMBOLS,
    STRIKE_GAP,
    ATM_RANGE,
    MAX_SYMBOLS,
)
from utils.logger import get_logger

logger = get_logger("symbol_manager")


@dataclass
class OptionSymbol:
    """Represents a single option contract."""
    symbol: str             # e.g. "NIFTY26032424500CE" (with YYMMDD expiry)
    symbol_id: int = 0
    underlying: str = ""    # e.g. "NIFTY"
    expiry: date = None
    strike: float = 0.0
    option_type: str = ""   # CE / PE
    lot_size: int = 0
    tick_size: float = 0.0
    relative_strike: int = 0  # 0=ATM, +1=ATM+1, -1=ATM-1


class SymbolManager:
    """
    Manages the dynamic option symbol universe.

    Workflow:
      1. load_symbol_master()  → fetch all F&O symbols from TrueData
      2. get_nearest_expiry()  → find current weekly expiry
      3. compute_atm()         → round spot to nearest strike
      4. get_option_symbols()  → ATM ±N strikes for CE + PE
      5. get_subscription_list() → full list for TCP subscription
    """

    def __init__(self):
        self._master: pd.DataFrame = pd.DataFrame()
        self._symbol_id_map: Dict[str, int] = {}
        self._id_symbol_map: Dict[int, str] = {}
        self._expiry_cache: Dict[str, List[date]] = {}
        self._last_refresh: Optional[datetime] = None

    # ── Symbol Master ───────────────────────────────────────────────────────

    def load_symbol_master(self, segment: str = "fo") -> pd.DataFrame:
        """
        Fetch full F&O symbol master from TrueData.
        GET https://api.truedata.in/getAllSymbols?segment=fo&user=X&password=Y&csv=true&csvHeader=true

        Returns DataFrame with columns:
          symbol, symbolid, underlying, expiry, strike, option_type, lot_size, tick_size
        """
        if not TRUEDATA_USER or not TRUEDATA_PASSWORD:
            logger.warning("TrueData credentials not set. Cannot load symbol master.")
            return pd.DataFrame()

        url = f"{TD_SYMBOL_MASTER_URL}/getAllSymbols"
        params = {
            "segment": segment,
            "user": TRUEDATA_USER,
            "password": TRUEDATA_PASSWORD,
            "csv": "true",
            "csvHeader": "true",
            "token": "true",
            "ticksize": "true",
            "underlying": "true",
        }

        try:
            logger.info(f"Fetching symbol master for segment={segment}...")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            df.columns = [c.strip().lower() for c in df.columns]

            # Normalize column names (TrueData may vary)
            col_map = {}
            for col in df.columns:
                if "symbolid" in col or "symbol_id" in col:
                    col_map[col] = "symbol_id"
                elif "symbol" in col and "id" not in col:
                    col_map[col] = "symbol"
                elif "expiry" in col:
                    col_map[col] = "expiry"
                elif "strike" in col:
                    col_map[col] = "strike"
                elif "option" in col and "type" in col:
                    col_map[col] = "option_type"
                elif "lot" in col:
                    col_map[col] = "lot_size"
                elif "tick" in col:
                    col_map[col] = "tick_size"
                elif "underlying" in col:
                    col_map[col] = "underlying"
            df.rename(columns=col_map, inplace=True)

            # Filter to NIFTY / BANKNIFTY options only
            if "underlying" in df.columns:
                df = df[df["underlying"].str.upper().isin(
                    [s.upper() for s in SYMBOLS]
                )].copy()

            # Parse expiry dates
            if "expiry" in df.columns:
                df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date

            # Build symbol ↔ ID maps
            if "symbol_id" in df.columns and "symbol" in df.columns:
                for _, row in df.iterrows():
                    sym = str(row["symbol"]).strip()
                    sid = int(row.get("symbol_id", 0))
                    self._symbol_id_map[sym] = sid
                    self._id_symbol_map[sid] = sym

            self._master = df
            self._last_refresh = datetime.now()

            # Cache expiry lists per underlying
            if "underlying" in df.columns and "expiry" in df.columns:
                for underlying in SYMBOLS:
                    mask = df["underlying"].str.upper() == underlying.upper()
                    expiries = sorted(df.loc[mask, "expiry"].dropna().unique())
                    self._expiry_cache[underlying] = expiries

            logger.info(
                f"Symbol master loaded: {len(df)} symbols, "
                f"expiries: {', '.join(str(len(v)) + ' for ' + k for k, v in self._expiry_cache.items())}"
            )
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to fetch symbol master: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error parsing symbol master: {e}")
            return pd.DataFrame()

    # ── Expiry Resolution ───────────────────────────────────────────────────

    def get_nearest_expiry(
        self, underlying: str, ref_date: Optional[date] = None
    ) -> Optional[date]:
        """
        Get the nearest weekly expiry for an underlying on or after ref_date.
        If ref_date is None, uses today.
        """
        ref_date = ref_date or date.today()
        expiries = self._expiry_cache.get(underlying, [])

        if not expiries:
            logger.warning(f"No expiries cached for {underlying}. Load symbol master first.")
            return None

        # Find first expiry >= ref_date
        for exp in expiries:
            if exp >= ref_date:
                return exp

        logger.warning(f"No future expiry found for {underlying} after {ref_date}")
        return expiries[-1] if expiries else None

    def get_next_expiry(
        self, underlying: str, ref_date: Optional[date] = None
    ) -> Optional[date]:
        """
        Get the SECOND nearest expiry (next week's expiry).
        On expiry day, liquidity shifts to next week — we need both.
        """
        ref_date = ref_date or date.today()
        expiries = self._expiry_cache.get(underlying, [])

        if not expiries:
            return None

        future_expiries = [e for e in expiries if e >= ref_date]
        if len(future_expiries) >= 2:
            return future_expiries[1]
        return None

    def get_current_and_next_expiry(
        self, underlying: str, ref_date: Optional[date] = None
    ) -> Tuple[Optional[date], Optional[date]]:
        """
        Get both current and next expiry.
        Returns (current_expiry, next_expiry). Either can be None.
        """
        return (
            self.get_nearest_expiry(underlying, ref_date),
            self.get_next_expiry(underlying, ref_date),
        )

    def get_expiry_for_timestamp(
        self, underlying: str, timestamp: datetime
    ) -> Optional[date]:
        """
        Get the correct expiry that was active at a historical timestamp.
        This is the nearest expiry ON or AFTER the timestamp's date.
        Used for historical dataset construction.
        """
        return self.get_nearest_expiry(underlying, ref_date=timestamp.date())

    # ── ATM Computation ─────────────────────────────────────────────────────

    @staticmethod
    def compute_atm(spot_price: float, strike_gap: int) -> float:
        """
        Round spot price to nearest valid strike (ATM).
        e.g., spot=24567, gap=50 → ATM=24550
              spot=52340, gap=100 → ATM=52300
        """
        return round(spot_price / strike_gap) * strike_gap

    def get_atm_strikes(
        self, spot_price: float, underlying: str, n_strikes: int = None
    ) -> List[float]:
        """
        Generate ATM ± N strike prices.
        Returns sorted list: [ATM-3, ATM-2, ATM-1, ATM, ATM+1, ATM+2, ATM+3]
        """
        n = n_strikes or ATM_RANGE
        gap = STRIKE_GAP.get(underlying, 50)
        atm = self.compute_atm(spot_price, gap)

        strikes = []
        for i in range(-n, n + 1):
            strikes.append(atm + i * gap)

        return strikes

    # ── Option Symbol Name Construction ─────────────────────────────────────

    @staticmethod
    def build_option_symbol_name(
        underlying: str, expiry: date, strike: float, option_type: str
    ) -> str:
        """
        Build TrueData option symbol name.

        Format: SYMBOL + YY + MM + DD + STRIKE + CE/PE
        Example: NIFTY26032424500CE
                 = NIFTY + 26 + 03 + 24 + 24500 + CE
                 = NIFTY, expiry 2026-03-24, strike 24500, Call

        Args:
            underlying: "NIFTY" or "BANKNIFTY"
            expiry: expiry date
            strike: strike price (will be converted to int)
            option_type: "CE" or "PE"
        """
        yy = expiry.strftime("%y")   # e.g. "26"
        mm = expiry.strftime("%m")   # e.g. "03"
        dd = expiry.strftime("%d")   # e.g. "24"
        return f"{underlying}{yy}{mm}{dd}{int(strike)}{option_type}"

    # ── Option Symbol Resolution ────────────────────────────────────────────

    def get_option_symbols(
        self,
        underlying: str,
        spot_price: float,
        expiry: Optional[date] = None,
        n_strikes: int = None,
    ) -> List[OptionSymbol]:
        """
        Get all option symbols for ATM ±N strikes for both CE and PE.

        Args:
            underlying: "NIFTY" or "BANKNIFTY"
            spot_price: current index spot price
            expiry: target expiry date (None = nearest)
            n_strikes: override ATM_RANGE

        Returns list of OptionSymbol objects.
        """
        n = n_strikes or ATM_RANGE
        expiry = expiry or self.get_nearest_expiry(underlying)

        if expiry is None:
            logger.error(f"Cannot resolve expiry for {underlying}")
            return []

        gap = STRIKE_GAP.get(underlying, 50)
        atm = self.compute_atm(spot_price, gap)
        strikes = self.get_atm_strikes(spot_price, underlying, n)

        symbols = []
        for strike in strikes:
            relative = int((strike - atm) / gap)

            for opt_type in ["CE", "PE"]:
                # Try to find in master first
                matched = self._find_in_master(underlying, expiry, strike, opt_type)

                if matched is not None:
                    sym = OptionSymbol(
                        symbol=str(matched.get("symbol", "")),
                        symbol_id=int(matched.get("symbol_id", 0)),
                        underlying=underlying,
                        expiry=expiry,
                        strike=strike,
                        option_type=opt_type,
                        lot_size=int(matched.get("lot_size", 0)),
                        tick_size=float(matched.get("tick_size", 0)),
                        relative_strike=relative,
                    )
                else:
                    # Construct symbol name using confirmed TrueData format:
                    # SYMBOL + YY + MM + DD + STRIKE + CE/PE
                    sym_name = self.build_option_symbol_name(
                        underlying, expiry, strike, opt_type
                    )
                    sym = OptionSymbol(
                        symbol=sym_name,
                        underlying=underlying,
                        expiry=expiry,
                        strike=strike,
                        option_type=opt_type,
                        relative_strike=relative,
                    )

                symbols.append(sym)

        logger.info(
            f"{underlying}: ATM={atm}, expiry={expiry}, "
            f"strikes={len(strikes)}, symbols={len(symbols)}"
        )
        return symbols

    def _find_in_master(
        self, underlying: str, expiry: date, strike: float, option_type: str
    ) -> Optional[dict]:
        """Look up a specific contract in the symbol master."""
        if self._master.empty:
            return None

        mask = (
            (self._master["underlying"].str.upper() == underlying.upper())
            & (self._master["expiry"] == expiry)
            & (abs(self._master["strike"] - strike) < 0.01)
            & (self._master["option_type"].str.upper() == option_type.upper())
        )
        matches = self._master[mask]

        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    # ── Subscription List ───────────────────────────────────────────────────

    def get_subscription_list(
        self,
        spot_prices: Dict[str, float],
        include_index: bool = True,
        use_spot: bool = True,
    ) -> List[str]:
        """
        Build the full list of symbols to subscribe to via TCP/WebSocket.

        Args:
            spot_prices: {"NIFTY": 24500.0, "BANKNIFTY": 52000.0}
            include_index: whether to include index symbols
            use_spot: True = use spot index (NIFTY 50) for live stream,
                      False = use continuous futures (NIFTY-I) for historical

        Returns list of symbol names, respecting MAX_SYMBOLS limit.
        """
        all_symbols = []

        # Add index symbols first
        if include_index:
            sym_map = TD_INDEX_SPOT_SYMBOLS if use_spot else TD_INDEX_FUTURES_SYMBOLS
            for underlying in SYMBOLS:
                idx_sym = sym_map.get(underlying)
                if idx_sym:
                    all_symbols.append(idx_sym)

        # Add option symbols for each underlying
        for underlying in SYMBOLS:
            spot = spot_prices.get(underlying)
            if spot is None:
                logger.warning(f"No spot price for {underlying}, skipping options.")
                continue

            option_syms = self.get_option_symbols(underlying, spot)
            for opt in option_syms:
                all_symbols.append(opt.symbol)

        # Enforce plan limit
        if len(all_symbols) > MAX_SYMBOLS:
            logger.warning(
                f"Symbol count {len(all_symbols)} exceeds plan limit {MAX_SYMBOLS}. "
                f"Truncating to {MAX_SYMBOLS}."
            )
            all_symbols = all_symbols[:MAX_SYMBOLS]

        logger.info(f"Subscription list: {len(all_symbols)} symbols")
        return all_symbols

    # ── Symbol ID Mapping ───────────────────────────────────────────────────

    def get_symbol_by_id(self, symbol_id: int) -> Optional[str]:
        """Resolve a symbolID back to symbol name."""
        return self._id_symbol_map.get(symbol_id)

    def get_id_by_symbol(self, symbol: str) -> Optional[int]:
        """Get the symbolID for a symbol name."""
        return self._symbol_id_map.get(symbol)

    def register_symbol_id(self, symbol: str, symbol_id: int):
        """Register a symbol ↔ symbolID mapping (from live TCP stream)."""
        self._symbol_id_map[symbol] = symbol_id
        self._id_symbol_map[symbol_id] = symbol

    # ── Fetch Expiry List via API ───────────────────────────────────────────

    def fetch_expiry_list(self, underlying: str) -> List[date]:
        """
        Fetch upcoming expiry dates from TrueData API.
        GET https://api.truedata.in/getSymbolExpiryList?symbol=NIFTY&response=csv
        """
        if not TRUEDATA_USER:
            return []

        url = f"{TD_SYMBOL_MASTER_URL}/getSymbolExpiryList"
        params = {
            "symbol": underlying,
            "user": TRUEDATA_USER,
            "password": TRUEDATA_PASSWORD,
            "response": "csv",
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            expiries = pd.to_datetime(df.iloc[:, 0], errors="coerce").dt.date.tolist()
            expiries = sorted([e for e in expiries if e is not None])

            self._expiry_cache[underlying] = expiries
            logger.info(f"Fetched {len(expiries)} expiries for {underlying}")
            return expiries

        except Exception as e:
            logger.error(f"Failed to fetch expiry list for {underlying}: {e}")
            return []

    # ── Fetch Option Chain Symbols via API ──────────────────────────────────

    def fetch_option_chain_symbols(
        self, underlying: str, expiry: date
    ) -> pd.DataFrame:
        """
        Fetch option chain symbols for a specific expiry.
        GET https://api.truedata.in/getoptionchain?segment=fo&symbol=NIFTY&expiry=YYYYMMDD&csv=true
        """
        if not TRUEDATA_USER:
            return pd.DataFrame()

        expiry_str = expiry.strftime("%Y%m%d")
        url = f"{TD_SYMBOL_MASTER_URL}/getoptionchain"
        params = {
            "segment": "fo",
            "user": TRUEDATA_USER,
            "password": TRUEDATA_PASSWORD,
            "symbol": underlying,
            "expiry": expiry_str,
            "csv": "true",
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            df.columns = [c.strip().lower() for c in df.columns]
            logger.info(
                f"Fetched {len(df)} option chain symbols for "
                f"{underlying} expiry={expiry}"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to fetch option chain for {underlying}: {e}")
            return pd.DataFrame()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return not self._master.empty

    @property
    def symbol_count(self) -> int:
        return len(self._master)

    @property
    def last_refresh(self) -> Optional[datetime]:
        return self._last_refresh

    def summary(self) -> str:
        """Return a human-readable summary of loaded symbols."""
        if self._master.empty:
            return "Symbol master not loaded."

        lines = [f"Symbol Master: {len(self._master)} total symbols"]
        for underlying in SYMBOLS:
            expiries = self._expiry_cache.get(underlying, [])
            nearest = expiries[0] if expiries else "N/A"
            count = len(self._master[
                self._master.get("underlying", pd.Series()).str.upper() == underlying.upper()
            ]) if "underlying" in self._master.columns else 0
            lines.append(f"  {underlying}: {count} symbols, nearest expiry: {nearest}")

        return "\n".join(lines)
