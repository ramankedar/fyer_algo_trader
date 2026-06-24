#!/usr/bin/env python3
"""
run_live_theta.py — Production runner for ProductionThetaStrategy.

Runs TWO strategies simultaneously on NIFTY Thursday expiry:
  Strategy A: entry=13:15 / buffer=0.50 / cat=4.0   (research baseline)
  Strategy B: entry=13:15 / buffer=0.40 / cat=3.0   (tighter variant)

Modes:
  Paper (default): simulates fills using real Fyers quotes, no real orders.
  Live  (--live) : places real orders via Fyers API. START WITH PAPER.

Data:
  - NIFTY spot    : Fyers /data/quotes (real-time polling)
  - 1-min bars    : Fyers /data/history (last completed bar)
  - Option chain  : Fyers /data/quotes per-strike (real bid/ask)
  - Morning range : Fyers /data/history 9:15–12:59 on startup

Logs:
  logs/theta_YYYY-MM-DD.log       system log (INFO/WARNING/ERROR)
  trades/theta_YYYY-MM-DD.jsonl   one JSON event per trade action

Infrastructure note:
  This strategy places 1–3 orders per Thursday. It is NOT HFT.
  A ₹15/month Oracle E2 AMD instance is more than sufficient.
  Latency requirement: seconds (not milliseconds).

Usage:
  python3 run_live_theta.py           # paper mode (safe default)
  python3 run_live_theta.py --live    # REAL money — verify paper results first
  python3 run_live_theta.py --dry-run # print schedule and exit without trading
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import hashlib

import httpx
import pyotp
from dotenv import load_dotenv
from scipy.stats import norm

# ── algo_platform imports ─────────────────────────────────────────────────────
from algo_platform.core.config import load_config, LOT_SIZES
from algo_platform.core.types import (
    FeatureVector, Instrument, MarketBar, OptionChain, OptionQuote, OptionType,
)
from algo_platform.data.chain_builder import SyntheticChainBuilder
from algo_platform.strategies import ProductionThetaStrategy

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv(override=True)
IST = ZoneInfo("Asia/Kolkata")

# ── Strategy parameters ───────────────────────────────────────────────────────
STRATEGIES = {
    "A_base":   dict(entry_time="13:15", entry_end_time="13:25",
                     buffer_mult=0.50, cat_premium_mult=4.0),
    "B_tight":  dict(entry_time="13:15", entry_end_time="13:25",
                     buffer_mult=0.40, cat_premium_mult=3.0),
}
INSTRUMENT   = "NIFTY"
LOT_SIZE     = LOT_SIZES[INSTRUMENT].lot_size       # 75
LOTS         = 1
THETA_CAPITAL = 120_000.0
POLL_SECS    = 30    # check every 30 seconds (NOT HFT — this is a weekly strategy)
SQUARE_OFF   = "15:15"

# ── Fyers endpoints and credentials ──────────────────────────────────────────
DATA_URL     = "https://api-t1.fyers.in/data"
API_URL      = "https://api-t1.fyers.in/api/v3"
VAGATOR_URL  = "https://api-t2.fyers.in/vagator/v2"

APP_ID       = os.environ.get("BROKER_APP_ID",       "").strip()
SECRET_KEY   = os.environ.get("BROKER_SECRET_KEY",   "").strip()
CLIENT_ID    = os.environ.get("BROKER_CLIENT_ID",    "").strip()
TOTP_KEY     = os.environ.get("BROKER_TOTP_KEY",     "").strip()
PIN          = os.environ.get("BROKER_PIN",           "").strip()
REDIRECT_URI = os.environ.get("BROKER_REDIRECT_URI", "http://127.0.0.1:8080/callback").strip()
TOKEN        = os.environ.get("BROKER_ACCESS_TOKEN", "").strip()


# ── Auto-authentication (ported from paper_trader.py) ────────────────────────

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


async def fyers_login() -> Optional[str]:
    """
    Headless 5-step Fyers login using TOTP + PIN.
    Requires BROKER_CLIENT_ID, BROKER_TOTP_KEY, BROKER_PIN, BROKER_APP_ID,
    BROKER_SECRET_KEY in .env.
    Returns new access_token or None on failure.
    """
    if not all([CLIENT_ID, TOTP_KEY, PIN, APP_ID, SECRET_KEY]):
        logging.warning(
            "Auto-auth skipped: missing credentials. "
            "Set BROKER_CLIENT_ID / BROKER_TOTP_KEY / BROKER_PIN / "
            "BROKER_SECRET_KEY in .env for automatic token refresh."
        )
        return None

    totp = pyotp.TOTP(TOTP_KEY.replace(" ", "").upper())
    logging.info("Starting Fyers auto-authentication...")

    async with httpx.AsyncClient(timeout=30) as c:
        # Step 1: Request OTP
        r = await c.post(f"{VAGATOR_URL}/send_login_otp",
                         json={"fy_id": CLIENT_ID, "app_id": "2"})
        d = r.json()
        if d.get("s") != "ok":
            logging.error("Auth step 1 failed: %s", d)
            return None
        rk = d["request_key"]

        # Step 2: Verify TOTP — wait for fresh token window
        import time as _time
        remaining = 30 - (int(_time.time()) % 30)
        if remaining < 8:
            logging.info("Waiting %ds for fresh TOTP window...", remaining + 1)
            await asyncio.sleep(remaining + 1)
        r = await c.post(f"{VAGATOR_URL}/verify_otp",
                         json={"request_key": rk, "otp": totp.now()})
        d = r.json()
        if d.get("s") != "ok":
            logging.error("Auth step 2 (TOTP) failed: %s", d)
            return None
        rk = d["request_key"]

        # Step 3: Verify PIN (try plain then SHA256)
        session_token = None
        for pin_val in [PIN, _sha256(PIN)]:
            r = await c.post(f"{VAGATOR_URL}/verify_pin",
                             json={"request_key": rk, "identity_type": "pin",
                                   "recaptcha_token": "", "pin": pin_val})
            d = r.json()
            session_token = (d.get("data") or {}).get("token")
            if session_token:
                break
        if not session_token:
            logging.error("Auth step 3 (PIN) failed: %s", d)
            return None

        # Step 4: Generate auth code
        app_type = APP_ID.split("-")[-1]
        r = await c.post(f"{API_URL}/generate-authcode",
                         headers={"Authorization": session_token},
                         json={"fyers_id": CLIENT_ID, "app_id": APP_ID,
                               "redirect_uri": REDIRECT_URI, "appType": app_type,
                               "code_challenge": "", "state": "None",
                               "scope": "", "nonce": "", "response_type": "code",
                               "create_cookie": True})
        d = r.json()
        url = d.get("Url", "")
        if "auth_code=" not in url:
            logging.error("Auth step 4 failed: %s", d)
            return None
        from urllib.parse import urlparse, parse_qs
        auth_code = parse_qs(urlparse(url).query).get("auth_code", [None])[0]

        # Step 5: Exchange for access token
        r = await c.post(f"{API_URL}/validate-authcode",
                         json={"grant_type":   "authorization_code",
                               "appIdHash":    _sha256(f"{APP_ID}:{SECRET_KEY}"),
                               "code":         auth_code})
        d = r.json()
        token = d.get("access_token")
        if not token:
            logging.error("Auth step 5 failed: %s", d)
            return None

        logging.info("Fyers auth successful (token len=%d)", len(token))
        return token


async def ensure_valid_token() -> str:
    """
    Return a valid token, fetching a fresh one if needed.
    Writes the new token back to .env so subsequent runs pick it up.
    """
    global TOKEN

    # Quick validation: try a lightweight quotes call
    if TOKEN:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{DATA_URL}/quotes",
                                headers=_auth_headers(),
                                params={"symbols": "NSE:NIFTY50-INDEX"})
                if r.json().get("s") == "ok":
                    logging.info("Existing token is valid.")
                    return TOKEN
        except Exception:
            pass
        logging.warning("Existing token invalid or expired — re-authenticating.")

    new_token = await fyers_login()
    if new_token:
        TOKEN = new_token
        # Persist back to .env so the next nohup/cron restart picks it up
        try:
            env_path = Path(".env")
            if env_path.exists():
                lines = env_path.read_text().splitlines()
                updated = []
                found = False
                for line in lines:
                    if line.startswith("BROKER_ACCESS_TOKEN"):
                        updated.append(f"BROKER_ACCESS_TOKEN={new_token}")
                        found = True
                    else:
                        updated.append(line)
                if not found:
                    updated.append(f"BROKER_ACCESS_TOKEN={new_token}")
                env_path.write_text("\n".join(updated) + "\n")
                logging.info("Token written back to .env")
        except Exception as e:
            logging.warning("Could not write token to .env: %s", e)

    return TOKEN

# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(log_dir: str, day: date) -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    log_path = f"{log_dir}/theta_{day}.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level   = logging.INFO,
        format  = fmt,
        handlers = [
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("live_theta")


def _trade_log_path(trades_dir: str, day: date, mode: str) -> str:
    Path(trades_dir).mkdir(exist_ok=True)
    return f"{trades_dir}/theta_{mode}_{day}.jsonl"


def _log_event(path: str, event: dict) -> None:
    """Append one JSON event to the trade log (append-only, crash-safe)."""
    event["logged_at"] = datetime.now(IST).isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


# ── Fyers data helpers ────────────────────────────────────────────────────────

def _auth_headers() -> dict:
    return {"Authorization": f"{APP_ID}:{TOKEN}"}


async def _fyers_get(client: httpx.AsyncClient, endpoint: str,
                     params: dict) -> Optional[dict]:
    try:
        r = await client.get(
            f"{DATA_URL}{endpoint}", headers=_auth_headers(), params=params, timeout=10
        )
        d = r.json()
        if d.get("s") == "ok":
            return d
        logging.warning("Fyers %s returned: %s", endpoint, d.get("message", d))
    except Exception as e:
        logging.error("Fyers %s error: %s", endpoint, e)
    return None


async def get_nifty_spot(client: httpx.AsyncClient) -> Optional[float]:
    """Current NIFTY spot via Fyers quotes."""
    d = await _fyers_get(client, "/quotes", {"symbols": "NSE:NIFTY50-INDEX"})
    if d:
        q = (d.get("d") or {}).get("q", [])
        if q:
            return float(q[0]["v"]["lp"])
    return None


async def get_nifty_bars(client: httpx.AsyncClient,
                         from_dt: str, to_dt: str,
                         resolution: int = 1) -> List[dict]:
    """
    Fetch NIFTY 1-min bars from Fyers history API.
    from_dt / to_dt: "YYYY-MM-DD HH:MM:SS" strings in IST.
    Returns list of {"t": epoch, "o": open, "h": high, "l": low, "c": close, "v": vol}.
    """
    d = await _fyers_get(client, "/history", {
        "symbol":      "NSE:NIFTY50-INDEX",
        "resolution":  str(resolution),
        "date_format": "1",
        "range_from":  from_dt,
        "range_to":    to_dt,
        "cont_flag":   "1",
    })
    if d:
        candles = d.get("candles", [])
        return [
            {"t": c[0], "o": c[1], "h": c[2], "l": c[3], "c": c[4], "v": c[5]}
            for c in candles
        ]
    return []


async def get_option_quotes(
    client:   httpx.AsyncClient,
    sc_sym:   str,
    sp_sym:   str,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Fetch bid/ask for both option legs in one call.
    Returns (ce_quote_dict, pe_quote_dict) with keys: bid, ask, ltp.
    """
    symbols = f"{sc_sym},{sp_sym}"
    d = await _fyers_get(client, "/quotes", {"symbols": symbols})
    if not d:
        return None, None

    quotes_by_sym = {}
    for q in (d.get("d") or {}).get("q", []):
        sym = q.get("n", "")
        v   = q.get("v", {})
        quotes_by_sym[sym] = {
            "bid": float(v.get("bid_price", v.get("lp", 0))),
            "ask": float(v.get("ask_price", v.get("lp", 0))),
            "ltp": float(v.get("lp", 0)),
        }
    return quotes_by_sym.get(sc_sym), quotes_by_sym.get(sp_sym)


def _fyers_option_symbol(expiry_date: date, strike: float, is_call: bool) -> str:
    """
    Build Fyers NSE option symbol.
    Format: NSE:NIFTY{YY}{MON}{DD}{STRIKE}{CE/PE}
    e.g.   NSE:NIFTY25JUL2424000CE
    """
    exp = expiry_date.strftime("%y%b%d").upper()
    suffix = "CE" if is_call else "PE"
    return f"NSE:NIFTY{exp}{int(strike)}{suffix}"


# ── Bar / chain builders ──────────────────────────────────────────────────────

def _make_bar(spot: float, bars_hist: List[dict], ts: datetime) -> MarketBar:
    """
    Create a MarketBar from the current spot and last completed 1-min bar.
    """
    if bars_hist:
        last = bars_hist[-1]
        return MarketBar(
            timestamp = ts,
            open      = float(last["o"]),
            high      = float(last["h"]),
            low       = float(last["l"]),
            close     = spot,      # use live spot as close
            volume    = float(last["v"]),
        )
    return MarketBar(timestamp=ts, open=spot, high=spot,
                     low=spot, close=spot, volume=0.0)


def _make_features(bars_hist: List[dict], ts: datetime,
                   vix: float = 14.0) -> FeatureVector:
    """
    Build a minimal FeatureVector from recent 1-min bars.
    Only timestamp and realized_vol are set; all other fields use dataclass defaults.
    """
    if len(bars_hist) >= 20:
        closes   = [b["c"] for b in bars_hist[-20:]]
        log_rets = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
        import numpy as np
        rv = float(np.std(log_rets) * math.sqrt(252 * 375))
    else:
        rv = vix / 100.0 if vix else 0.14

    return FeatureVector(
        timestamp    = ts,
        realized_vol = rv,
        # All other fields (iv_rank, atr, entropy, hurst, etc.) default to 0.0 / 0.5
    )


def _make_chain_from_quotes(
    spot:        float,
    expiry:      date,
    ts:          datetime,
    sc_strike:   float,
    sp_strike:   float,
    ce_q:        Optional[dict],
    pe_q:        Optional[dict],
    vix:         float,
    chain_builder: SyntheticChainBuilder,
) -> OptionChain:
    """
    Build an OptionChain prioritising real Fyers bid/ask for the two traded strikes,
    falling back to the synthetic chain for other strikes (needed for should_exit
    and compute_exit_value on the non-stopped leg).
    """
    # Start with a full synthetic chain for context
    tte = max(0.0, (15.5 - ts.hour - ts.minute / 60.0)) / (6.25 * 365)
    tte = max(1e-6, tte)
    chain = chain_builder.build(
        instrument = INSTRUMENT,
        spot       = spot,
        timestamp  = ts,
        atm_iv     = vix / 100.0,
    )

    # Patch the two traded strikes with real Fyers bid/ask
    patched = []
    for q in chain.quotes:
        if abs(q.strike - sc_strike) < 0.5 and q.option_type == OptionType.CALL and ce_q:
            q = OptionQuote(
                symbol      = q.symbol,
                instrument  = q.instrument,
                strike      = q.strike,
                option_type = q.option_type,
                expiry      = q.expiry,
                ltp         = ce_q["ltp"],
                bid         = ce_q["bid"],
                ask         = ce_q["ask"],
                oi          = q.oi,
                oi_change   = q.oi_change,
                volume      = q.volume,
                iv          = q.iv,
                delta       = q.delta,
                gamma       = q.gamma,
                theta       = q.theta,
                vega        = q.vega,
            )
        elif abs(q.strike - sp_strike) < 0.5 and q.option_type == OptionType.PUT and pe_q:
            q = OptionQuote(
                symbol      = q.symbol,
                instrument  = q.instrument,
                strike      = q.strike,
                option_type = q.option_type,
                expiry      = q.expiry,
                ltp         = pe_q["ltp"],
                bid         = pe_q["bid"],
                ask         = pe_q["ask"],
                oi          = q.oi,
                oi_change   = q.oi_change,
                volume      = q.volume,
                iv          = q.iv,
                delta       = q.delta,
                gamma       = q.gamma,
                theta       = q.theta,
                vega        = q.vega,
            )
        patched.append(q)

    chain.quotes = patched
    return chain


# ── Paper order simulator ─────────────────────────────────────────────────────

class PaperOrderRouter:
    """Simulates order fills using real bid/ask prices. No real orders placed."""

    def sell(self, symbol: str, qty: int, bid: float) -> dict:
        fill = round(bid - 0.5, 2)   # pessimistic: fill at bid - 0.5
        return {"order_id": f"PAPER-{uuid.uuid4().hex[:8]}", "fill_price": fill}

    def buy(self, symbol: str, qty: int, ask: float) -> dict:
        fill = round(ask + 0.5, 2)   # pessimistic: fill at ask + 0.5
        return {"order_id": f"PAPER-{uuid.uuid4().hex[:8]}", "fill_price": fill}


class LiveOrderRouter:
    """Places real orders via Fyers REST API."""

    async def sell(self, client: httpx.AsyncClient, symbol: str,
                   qty: int, price: float) -> dict:
        payload = {
            "symbol":          symbol,
            "qty":             qty,
            "type":            2,          # LIMIT
            "side":            -1,         # SELL
            "productType":     "INTRADAY",
            "limitPrice":      price,
            "stopPrice":       0,
            "validity":        "DAY",
            "filledQty":       0,
            "disclosedQty":    0,
            "offlineOrder":    False,
        }
        try:
            r = await client.post(
                f"{API_URL}/orders/sync",
                headers={**_auth_headers(), "Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )
            d = r.json()
            return {"order_id": d.get("id", ""), "message": d.get("message", "")}
        except Exception as e:
            return {"order_id": "", "error": str(e)}

    async def buy(self, client: httpx.AsyncClient, symbol: str,
                  qty: int, price: float) -> dict:
        payload = {
            "symbol":       symbol,
            "qty":          qty,
            "type":         2,
            "side":         1,             # BUY to close short
            "productType":  "INTRADAY",
            "limitPrice":   price,
            "stopPrice":    0,
            "validity":     "DAY",
            "filledQty":    0,
            "disclosedQty": 0,
            "offlineOrder": False,
        }
        try:
            r = await client.post(
                f"{API_URL}/orders/sync",
                headers={**_auth_headers(), "Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )
            d = r.json()
            return {"order_id": d.get("id", ""), "message": d.get("message", "")}
        except Exception as e:
            return {"order_id": "", "error": str(e)}


# ── Strategy runner ───────────────────────────────────────────────────────────

class StrategyRunner:
    """
    Wraps one ProductionThetaStrategy instance for live/paper trading.
    Handles entry, exit, and order routing for one parameter set.
    """

    def __init__(
        self,
        name:        str,
        strategy:    ProductionThetaStrategy,
        trade_log:   str,
        mode:        str,
        paper_router: PaperOrderRouter,
        live_router:  Optional[LiveOrderRouter] = None,
    ):
        self.name          = name
        self.strategy      = strategy
        self.trade_log     = trade_log
        self.mode          = mode
        self._paper        = paper_router
        self._live         = live_router
        self._trade_id:    Optional[str] = None
        self._sc_sym:      Optional[str] = None
        self._sp_sym:      Optional[str] = None
        self._ce_order_id: Optional[str] = None
        self._pe_order_id: Optional[str] = None
        self._ce_fill:     float = 0.0
        self._pe_fill:     float = 0.0
        self._entry_time:  Optional[datetime] = None
        self._expiry:      Optional[date] = None

        self.log = logging.getLogger(f"theta.{name}")

    async def tick(
        self,
        client:       httpx.AsyncClient,
        bar:          MarketBar,
        chain:        OptionChain,
        features:     FeatureVector,
        today:        date,
        vix:          float,
    ) -> None:
        """Called every POLL_SECS. Drives generate_signal → should_exit → order routing."""

        if not self.strategy._in_trade:
            # ── Try entry ──────────────────────────────────────────────────────
            signal = self.strategy.generate_signal(bar, chain, features)
            if signal is None:
                return

            # Entry signal generated — place orders for both legs
            ce_leg = next((l for l in signal.legs if l.option_type == OptionType.CALL), None)
            pe_leg = next((l for l in signal.legs if l.option_type == OptionType.PUT), None)
            if ce_leg is None or pe_leg is None:
                self.log.error("Signal missing legs — skipping")
                return

            self._trade_id  = str(uuid.uuid4())[:12]
            self._expiry    = today
            self._sc_sym    = _fyers_option_symbol(today, ce_leg.strike, True)
            self._sp_sym    = _fyers_option_symbol(today, pe_leg.strike, False)
            self._entry_time = bar.timestamp

            if self.mode == "paper":
                ce_res = self._paper.sell(self._sc_sym, LOT_SIZE, ce_leg.limit_price)
                pe_res = self._paper.sell(self._sp_sym, LOT_SIZE, pe_leg.limit_price)
            else:
                ce_res = await self._live.sell(client, self._sc_sym, LOT_SIZE, ce_leg.limit_price)
                pe_res = await self._live.sell(client, self._sp_sym, LOT_SIZE, pe_leg.limit_price)

            self._ce_order_id = ce_res["order_id"]
            self._pe_order_id = pe_res["order_id"]
            self._ce_fill     = ce_res.get("fill_price", ce_leg.limit_price)
            self._pe_fill     = pe_res.get("fill_price", pe_leg.limit_price)

            self.log.info(
                "ENTRY [%s] SC=%s @%.2f  SP=%s @%.2f  credit=%.2f  buf=%.0f",
                self.name, self._sc_sym, self._ce_fill,
                self._sp_sym, self._pe_fill,
                self._ce_fill + self._pe_fill, self.strategy._buffer,
            )

            _log_event(self.trade_log, {
                "event":          "entry",
                "strategy":       self.name,
                "trade_id":       self._trade_id,
                "mode":           self.mode,
                "sc_strike":      ce_leg.strike,
                "sp_strike":      pe_leg.strike,
                "sc_symbol":      self._sc_sym,
                "sp_symbol":      self._sp_sym,
                "ce_model_bid":   ce_leg.limit_price,
                "pe_model_bid":   pe_leg.limit_price,
                "ce_fill":        self._ce_fill,
                "pe_fill":        self._pe_fill,
                "credit_model":   ce_leg.limit_price + pe_leg.limit_price,
                "credit_actual":  self._ce_fill + self._pe_fill,
                "fill_slippage":  (ce_leg.limit_price + pe_leg.limit_price)
                                   - (self._ce_fill + self._pe_fill),
                "morning_range":  self.strategy._morning_range,
                "buffer":         self.strategy._buffer,
                "vix":            vix,
                "spot_entry":     bar.close,
            })

        else:
            # ── Check exit ────────────────────────────────────────────────────
            should, reason = self.strategy.should_exit(bar, features, 0.0, chain)
            if not should:
                # Log current position status every 5 minutes (every 10 polls)
                return

            # Exit triggered — determine which legs need to close
            ce_stopped = self.strategy._ce_stopped
            pe_stopped = self.strategy._pe_stopped

            # Legs that hit a STOP have an estimated exit price already stored.
            # The surviving leg (if any) exits at current ask.
            ce_exit_price = self.strategy._ce_stop_price
            pe_exit_price = self.strategy._pe_stop_price

            # Fetch current quotes for legs without a stop price
            if ce_exit_price is None or pe_exit_price is None:
                ce_q, pe_q = await get_option_quotes(
                    client, self._sc_sym, self._sp_sym
                )
                if ce_exit_price is None and ce_q:
                    ce_exit_price = ce_q["ask"]   # buying back a short
                if pe_exit_price is None and pe_q:
                    pe_exit_price = pe_q["ask"]

            ce_exit_price = ce_exit_price or 0.0
            pe_exit_price = pe_exit_price or 0.0

            # Place close orders
            if self.mode == "paper":
                ce_close = self._paper.buy(self._sc_sym, LOT_SIZE, ce_exit_price)
                pe_close = self._paper.buy(self._sp_sym, LOT_SIZE, pe_exit_price)
            else:
                ce_close = await self._live.buy(client, self._sc_sym, LOT_SIZE, ce_exit_price)
                pe_close = await self._live.buy(client, self._sp_sym, LOT_SIZE, pe_exit_price)

            ce_close_fill = ce_close.get("fill_price", ce_exit_price)
            pe_close_fill = pe_close.get("fill_price", pe_exit_price)

            pnl = (self._ce_fill + self._pe_fill - ce_close_fill - pe_close_fill) * LOT_SIZE
            duration_min = int((bar.timestamp - self._entry_time).total_seconds() / 60)

            self.log.info(
                "EXIT  [%s] reason=%-18s  CE_close=%.2f  PE_close=%.2f  "
                "PnL=₹%+,.0f  held=%dm",
                self.name, reason, ce_close_fill, pe_close_fill, pnl, duration_min,
            )

            _log_event(self.trade_log, {
                "event":            "exit",
                "strategy":         self.name,
                "trade_id":         self._trade_id,
                "exit_reason":      reason,
                "ce_exit_model":    ce_exit_price,
                "pe_exit_model":    pe_exit_price,
                "ce_exit_actual":   ce_close_fill,
                "pe_exit_actual":   pe_close_fill,
                "exit_slippage":    (ce_exit_price + pe_exit_price)
                                    - (ce_close_fill + pe_close_fill),
                "pnl_gross":        pnl,
                "pnl_net":          pnl - 300,    # approx ₹300 tx cost per trade
                "hold_minutes":     duration_min,
                "spot_exit":        bar.close,
                "ce_stopped":       ce_stopped,
                "pe_stopped":       pe_stopped,
            })

            # Reset for next session (new_session called by engine on next day)
            self._trade_id = self._sc_sym = self._sp_sym = None
            self._ce_fill  = self._pe_fill = 0.0


# ── Main orchestrator ─────────────────────────────────────────────────────────

class ThetaOrchestrator:

    def __init__(self, mode: str, log_dir: str, trades_dir: str) -> None:
        self.mode       = mode
        self.log        = logging.getLogger("theta.orch")
        self.today      = date.today()
        self.trade_log  = _trade_log_path(trades_dir, self.today, mode)
        self._running   = True

        cfg             = load_config()
        chain_builder   = SyntheticChainBuilder(cfg.risk_free_rate)
        paper_router    = PaperOrderRouter()
        live_router     = LiveOrderRouter() if mode == "live" else None

        self._runners: List[StrategyRunner] = []
        for name, params in STRATEGIES.items():
            strat = ProductionThetaStrategy(
                Instrument(INSTRUMENT), cfg,
                quantity         = LOTS,
                otm_sigma_mult   = 0.30,
                **params,
            )
            self._runners.append(StrategyRunner(
                name=name, strategy=strat,
                trade_log=self.trade_log, mode=mode,
                paper_router=paper_router, live_router=live_router,
            ))

        self._chain_builder = chain_builder
        self._bars_hist: List[dict] = []
        self._vix: float = 14.0

    def _in_trading_window(self) -> bool:
        now = datetime.now(IST).time()
        from datetime import time as dtime
        return dtime(9, 0) <= now <= dtime(15, 30)

    def _is_expiry_day(self) -> bool:
        """Only run on Thursdays (NIFTY expiry day = weekday 3)."""
        return self.today.weekday() == LOT_SIZES[INSTRUMENT].expiry_weekday

    async def _load_morning_range_and_history(self, client: httpx.AsyncClient) -> None:
        """Fetch 9:15–13:00 bars for morning range and feature computation."""
        self.log.info("Loading morning bars for morning-range computation...")
        today_str = self.today.strftime("%Y-%m-%d")
        bars = await get_nifty_bars(client,
                                    f"{today_str} 09:15:00",
                                    f"{today_str} 12:59:00")
        if bars:
            self._bars_hist = bars
            h = max(b["h"] for b in bars)
            l = min(b["l"] for b in bars)
            self.log.info("Morning range: %.0f pts (high=%.0f low=%.0f) from %d bars",
                          h - l, h, l, len(bars))
        else:
            self.log.warning("No morning bars fetched — morning range will be zero")

    async def run(self) -> None:
        signal.signal(signal.SIGINT,  lambda s, f: setattr(self, "_running", False))
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, "_running", False))

        if not self._is_expiry_day():
            self.log.info("Today (%s, weekday=%d) is not NIFTY expiry Thursday — exiting.",
                          self.today, self.today.weekday())
            return

        self.log.info("=" * 60)
        self.log.info("ProductionTheta %s MODE — %s", self.mode.upper(), self.today)
        self.log.info("Strategies: %s", list(STRATEGIES.keys()))
        self.log.info("Poll interval: %ds", POLL_SECS)
        self.log.info("=" * 60)

        _log_event(self.trade_log, {
            "event":      "session_start",
            "date":       str(self.today),
            "mode":       self.mode,
            "strategies": list(STRATEGIES.keys()),
        })

        # ── Ensure we have a valid Fyers token before doing anything ──────────
        token = await ensure_valid_token()
        if not token:
            self.log.error(
                "No valid Fyers token and auto-auth failed. "
                "Set BROKER_TOTP_KEY + BROKER_PIN + BROKER_SECRET_KEY in .env "
                "OR manually set BROKER_ACCESS_TOKEN. Exiting."
            )
            return

        async with httpx.AsyncClient() as client:
            # Load morning range at startup or when entering trading window
            await self._load_morning_range_and_history(client)

            # Reset all strategies for the new session
            for r in self._runners:
                r.strategy.new_session()

            while self._running:
                now = datetime.now(IST)

                if not self._in_trading_window():
                    if now.hour >= 15 and now.minute >= 30:
                        self.log.info("Market closed. Session done.")
                        break
                    await asyncio.sleep(60)
                    continue

                # Fetch live data
                spot = await get_nifty_spot(client)
                if spot is None:
                    self.log.warning("Could not fetch NIFTY spot — skipping tick")
                    await asyncio.sleep(POLL_SECS)
                    continue

                # Get latest 1-min bar (last completed bar = request last 5 mins)
                today_str = self.today.strftime("%Y-%m-%d")
                ts_from = (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
                recent = await get_nifty_bars(client, ts_from,
                                              now.strftime("%Y-%m-%d %H:%M:%S"))
                if recent:
                    # Add to history and deduplicate by timestamp
                    existing_ts = {b["t"] for b in self._bars_hist}
                    for b in recent:
                        if b["t"] not in existing_ts:
                            self._bars_hist.append(b)
                    self._bars_hist.sort(key=lambda x: x["t"])

                # Get VIX (use cached; refresh every 5 minutes approximately)
                # For simplicity, use the chain builder's default if not available
                bar      = _make_bar(spot, self._bars_hist, now)
                features = _make_features(self._bars_hist, now, self._vix)

                # Build chain (synthetic base + real quotes for traded strikes if available)
                chain = self._chain_builder.build(
                    instrument = INSTRUMENT,
                    spot       = spot,
                    timestamp  = now,
                    atm_iv     = self._vix / 100.0,
                )

                # Drive each strategy
                for runner in self._runners:
                    # If strategy is in trade, patch chain with real option quotes
                    if (runner.strategy._in_trade
                            and runner._sc_sym and runner._sp_sym):
                        ce_q, pe_q = await get_option_quotes(
                            client, runner._sc_sym, runner._sp_sym
                        )
                        if ce_q and pe_q and runner.strategy._sc_strike:
                            chain = _make_chain_from_quotes(
                                spot       = spot,
                                expiry     = self.today,
                                ts         = now,
                                sc_strike  = runner.strategy._sc_strike,
                                sp_strike  = runner.strategy._sp_strike,
                                ce_q       = ce_q,
                                pe_q       = pe_q,
                                vix        = self._vix,
                                chain_builder = self._chain_builder,
                            )

                    await runner.tick(client, bar, chain, features, self.today, self._vix)

                # Status line (replaces stdout every poll)
                in_trade = sum(1 for r in self._runners if r.strategy._in_trade)
                print(f"\r  {now.strftime('%H:%M:%S')}  NIFTY={spot:.0f}  "
                      f"in_trade={in_trade}/{len(self._runners)}  "
                      f"mode={self.mode}  ", end="", flush=True)

                await asyncio.sleep(POLL_SECS)

        _log_event(self.trade_log, {"event": "session_end", "date": str(self.today)})
        self.log.info("Session complete. Trade log: %s", self.trade_log)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Live/Paper theta strategy runner")
    p.add_argument("--live",    action="store_true",
                   help="Place REAL orders. Default is paper mode.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print schedule and parameter summary, then exit.")
    p.add_argument("--log-dir",    default="logs")
    p.add_argument("--trades-dir", default="trades")
    args = p.parse_args()

    today = date.today()
    log   = _setup_logging(args.log_dir, today)

    if args.dry_run:
        print("\nProductionTheta DRY RUN")
        print(f"  Today    : {today} (weekday={today.weekday()}, "
              f"Thursday={'YES' if today.weekday()==3 else 'NO'})")
        print(f"  App ID   : {APP_ID[:8]}..." if APP_ID else "  App ID   : NOT SET")
        print(f"  Token    : {'SET' if TOKEN else 'NOT SET'}")
        print()
        for name, params in STRATEGIES.items():
            print(f"  Strategy {name}:")
            for k, v in params.items():
                print(f"    {k:<22} = {v}")
        print()
        return

    if args.live:
        print("\n" + "!"*60)
        print("  LIVE MODE — real orders will be placed via Fyers.")
        print("  Paper trade for at least 8 Thursdays before going live.")
        print("  Current allocation: ₹1.2L per strategy × 2 = ₹2.4L")
        confirm = input("  Type CONFIRM to proceed: ")
        if confirm.strip() != "CONFIRM":
            print("  Aborted.")
            return
        print("!"*60 + "\n")

    mode = "live" if args.live else "paper"
    orchestrator = ThetaOrchestrator(mode, args.log_dir, args.trades_dir)
    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
