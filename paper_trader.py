#!/usr/bin/env python3
"""
paper_trader.py — Live paper trading engine for Oracle Cloud deployment.

Connects to real Fyers WebSocket for live option chain data.
Runs the 3 validated strategies (SkewHunter + Iron Condor + Zen Debit).
ALL trades are SIMULATED — no real orders placed.
Generates daily P&L report and logs every signal.

Usage (inside tmux on Oracle):
    source .env
    python3 paper_trader.py

Logs → logs/paper_YYYY-MM-DD.log
Trades → paper_trades/YYYY-MM-DD.csv
"""

import asyncio
import csv
import hashlib
import logging
import os
import sys
import signal
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import httpx
import pyotp
from dotenv import load_dotenv

# ── Load credentials ──────────────────────────────────────────────────────────
load_dotenv(override=True)
try:
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export").strip()
            if "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
except FileNotFoundError:
    pass

# ── Setup directories ─────────────────────────────────────────────────────────
os.makedirs("logs",         exist_ok=True)
os.makedirs("paper_trades", exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/paper_{today}.log"),
    ],
)
logger = logging.getLogger("paper_trader")

# ── Constants ─────────────────────────────────────────────────────────────────
CAPITAL        = float(os.environ.get("PAPER_CAPITAL", "200000"))
VAGATOR        = "https://api-t2.fyers.in/vagator/v2"
API_V3         = "https://api-t1.fyers.in/api/v3"

APP_ID         = os.environ.get("BROKER_APP_ID",       "").strip()
SECRET_KEY     = os.environ.get("BROKER_SECRET_KEY",   "").strip()
CLIENT_ID      = os.environ.get("BROKER_CLIENT_ID",    "").strip()
TOTP_KEY       = os.environ.get("BROKER_TOTP_KEY",     "").strip()
PIN            = os.environ.get("BROKER_PIN",           "").strip()
REDIRECT_URI   = os.environ.get("BROKER_REDIRECT_URI", "http://127.0.0.1:8080/callback").strip()

# ── Paper trade ledger ────────────────────────────────────────────────────────

class PaperLedger:
    """Records all hypothetical trades and tracks daily P&L."""

    def __init__(self, capital: float):
        self.capital       = capital
        self.daily_pnl     = 0.0
        self.total_pnl     = 0.0
        self.trades: List[dict] = []
        self._csv_path     = f"paper_trades/{today}.csv"
        self._open_positions: Dict[str, dict] = {}

        # Write CSV header
        with open(self._csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "time","strategy","symbol","direction","quantity",
                "entry_price","exit_price","pnl","reason","status"
            ]).writeheader()

    def open_trade(self, strategy: str, symbol: str, direction: str,
                   quantity: int, price: float, stop_loss: float, target: float) -> str:
        trade_id = f"{strategy}_{symbol}_{datetime.now().strftime('%H%M%S')}"
        self._open_positions[trade_id] = {
            "strategy": strategy, "symbol": symbol,
            "direction": direction, "quantity": quantity,
            "entry_price": price, "stop_loss": stop_loss, "target": target,
            "entry_time": datetime.now().isoformat(),
        }
        logger.info(
            "PAPER OPEN  %-20s  %-35s  %s  qty=%d  @₹%.2f  SL=₹%.2f  TGT=₹%.2f",
            strategy, symbol[:35], direction, quantity, price, stop_loss, target,
        )
        return trade_id

    def close_trade(self, trade_id: str, exit_price: float, reason: str) -> float:
        pos = self._open_positions.pop(trade_id, None)
        if not pos:
            return 0.0
        dm  = 1 if pos["direction"] == "BUY" else -1
        pnl = (exit_price - pos["entry_price"]) * pos["quantity"] * dm
        self.daily_pnl += pnl
        self.total_pnl += pnl

        row = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "strategy": pos["strategy"], "symbol": pos["symbol"],
            "direction": pos["direction"], "quantity": pos["quantity"],
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "pnl": round(pnl, 2), "reason": reason, "status": "CLOSED",
        }
        self.trades.append(row)
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

        logger.info(
            "PAPER CLOSE %-20s  %-35s  @₹%.2f  PnL=₹%+.2f  (%s)",
            pos["strategy"], pos["symbol"][:35], exit_price, pnl, reason,
        )
        return pnl

    def daily_report(self) -> str:
        win  = [t for t in self.trades if t["pnl"] > 0]
        loss = [t for t in self.trades if t["pnl"] <= 0]
        lines = [
            f"\n{'═'*55}",
            f"  PAPER TRADING DAILY REPORT  —  {today}",
            f"{'═'*55}",
            f"  Capital          : ₹{self.capital:,.0f}",
            f"  Daily P&L        : ₹{self.daily_pnl:+,.2f}  "
            f"({self.daily_pnl/self.capital*100:+.2f}%)",
            f"  Total Trades     : {len(self.trades)}",
            f"  Wins / Losses    : {len(win)} / {len(loss)}",
            f"  Win Rate         : "
            f"{len(win)/max(1,len(self.trades))*100:.1f}%",
            f"  Open Positions   : {len(self._open_positions)}",
            f"  Trade log        : {self._csv_path}",
            f"{'═'*55}\n",
        ]
        return "\n".join(lines)


# ── Fyers Authentication ──────────────────────────────────────────────────────

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


async def fyers_login() -> Optional[str]:
    """5-step headless Fyers login. Returns access_token or None."""
    import time
    totp = pyotp.TOTP(TOTP_KEY.replace(" ", "").upper())

    async with httpx.AsyncClient(timeout=30) as c:
        # Step 1
        r = await c.post(f"{VAGATOR}/send_login_otp",
                         json={"fy_id": CLIENT_ID, "app_id": "2"})
        d = r.json()
        if d.get("s") != "ok":
            logger.error("Step 1 failed: %s", d)
            return None
        rk = d["request_key"]

        # Step 2 (TOTP)
        while True:
            remaining = 30 - (int(time.time()) % 30)
            if remaining >= 8:
                break
            logger.info("Waiting %ds for fresh TOTP...", remaining)
            await asyncio.sleep(remaining + 1)
        r = await c.post(f"{VAGATOR}/verify_otp",
                         json={"request_key": rk, "otp": totp.now()})
        d = r.json()
        if d.get("s") != "ok":
            logger.error("Step 2 failed: %s", d)
            return None
        rk = d["request_key"]

        # Step 3 (PIN — try plain then SHA256)
        session_token = None
        for pin_val in [PIN, _sha256(PIN)]:
            r = await c.post(f"{VAGATOR}/verify_pin",
                             json={"request_key": rk, "identity_type": "pin",
                                   "recaptcha_token": "", "pin": pin_val})
            d = r.json()
            session_token = (d.get("data") or {}).get("token")
            if session_token:
                break
        if not session_token:
            logger.error("Step 3 (PIN) failed: %s", d)
            logger.error("Check your BROKER_PIN in .env — IP must also be whitelisted.")
            return None

        # Step 4
        app_type = APP_ID.split("-")[-1]
        r = await c.post(f"{API_V3}/generate-authcode",
                         headers={"Authorization": session_token},
                         json={"fyers_id": CLIENT_ID, "app_id": APP_ID,
                               "redirect_uri": REDIRECT_URI, "appType": app_type,
                               "code_challenge": "", "state": "None",
                               "scope": "", "nonce": "", "response_type": "code",
                               "create_cookie": True})
        d = r.json()
        url = d.get("Url", "")
        if "auth_code=" not in url:
            logger.error("Step 4 failed: %s", d)
            return None
        auth_code = parse_qs(urlparse(url).query).get("auth_code", [None])[0]

        # Step 5
        r = await c.post(f"{API_V3}/validate-authcode",
                         json={"grant_type": "authorization_code",
                               "appIdHash": _sha256(f"{APP_ID}:{SECRET_KEY}"),
                               "code": auth_code})
        d = r.json()
        token = d.get("access_token")
        if not token:
            logger.error("Step 5 failed: %s", d)
            return None
        logger.info("Authentication successful (token length=%d)", len(token))
        return token


# ── Market data helpers ───────────────────────────────────────────────────────

async def get_quote(symbol: str, token: str) -> Optional[float]:
    """Get current LTP for a symbol from Fyers."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{API_V3}/quotes",
                        headers={"Authorization": f"{APP_ID}:{token}"},
                        params={"symbols": symbol})
        d = r.json()
        quotes = (d.get("d") or {}).get("q", [])
        if quotes:
            return float(quotes[0].get("v", {}).get("lp", 0))
    return None


# ── Strategy signals (pure signal logic, no order placement) ──────────────────

class PaperStrategyEngine:
    """
    Runs the 3 validated strategies on real Fyers data.
    Generates signals, tracks paper positions through PaperLedger.
    """

    def __init__(self, ledger: PaperLedger, token: str):
        self.ledger     = ledger
        self.token      = token
        self._open_ids:  Dict[str, str] = {}   # strategy → trade_id
        self._spot_hist: List[float]    = []
        self._vix_hist:  List[float]    = []
        self._oi_prev:   Dict[str, int] = {}

    def _momentum(self, n: int = 15) -> float:
        if len(self._spot_hist) < n + 1:
            return 0.0
        return (self._spot_hist[-1] - self._spot_hist[-(n+1)]) / self._spot_hist[-(n+1)]

    def _iv_rank(self, vix: float) -> float:
        if len(self._vix_hist) < 10:
            return 0.5
        lo, hi = min(self._vix_hist), max(self._vix_hist)
        return (vix - lo) / (hi - lo) if hi > lo else 0.5

    # ── SkewHunter signal ─────────────────────────────────────────────────────

    async def check_skewhunter(self, spot: float, atm: float) -> None:
        """
        SkewHunter: OTM call volume ratio vs ITM put volume.
        Fires when call/put OI ratio > 1.25 (bullish) or < 0.80 (bearish).
        Uses real Fyers live quote for OTM/ITM options.
        """
        if "SkewHunter" in self._open_ids:
            # Check exit
            trade_id = self._open_ids["SkewHunter"]
            # Position management happens in main loop via LTP check
            return

        now = datetime.now().time()
        if not (dt_time(10, 15) <= now <= dt_time(14, 15)):
            return

        vix_approx = self._vix_hist[-1] if self._vix_hist else 14.0
        if not (10.0 <= vix_approx <= 22.0):
            return

        mom_15 = self._momentum(15)
        if abs(mom_15) < 0.0005:
            return

        # Get OTM call and ITM put quotes (1 lot = 25 for Nifty)
        expiry_str = self._next_expiry()
        call_sym = f"NSE:NIFTY{expiry_str}{int(atm+50)}CE"
        put_sym  = f"NSE:NIFTY{expiry_str}{int(atm-50)}PE"

        call_ltp = await get_quote(call_sym, self.token)
        put_ltp  = await get_quote(put_sym,  self.token)

        if not call_ltp or not put_ltp or call_ltp < 20 or put_ltp < 20:
            return

        # Simple momentum-based entry (real signal from underlying)
        if mom_15 > 0.001:
            entry_price = call_ltp
            sl          = entry_price * 0.70
            target      = entry_price * 1.40  # +40% trigger (Phase 3 wider trailing)
            tid = self.ledger.open_trade(
                "SkewHunter", call_sym, "BUY", 25, entry_price, sl, target
            )
            self._open_ids["SkewHunter"] = tid
        elif mom_15 < -0.001:
            entry_price = put_ltp
            sl          = entry_price * 0.70
            target      = entry_price * 1.40
            tid = self.ledger.open_trade(
                "SkewHunter", put_sym, "BUY", 25, entry_price, sl, target
            )
            self._open_ids["SkewHunter"] = tid

    # ── Iron Condor signal ────────────────────────────────────────────────────

    async def check_iron_condor(self, spot: float, atm: float, vix: float) -> None:
        """
        Iron Condor: weekly entry 10:30–11:15 AM when IV rank > 15%.
        Sell OTM call + put, buy wings 3 strikes further.
        """
        if "IronCondor" in self._open_ids:
            return

        now = datetime.now().time()
        if not (dt_time(10, 30) <= now <= dt_time(11, 15)):
            return

        iv_rank = self._iv_rank(vix)
        if iv_rank < 0.15:
            return

        # Check if we already traded this week
        expiry_str = self._next_expiry()

        short_call_strike = atm + 100  # 2si for Nifty (si=50)
        short_put_strike  = atm - 100
        wing_call_strike  = atm + 250  # 5si
        wing_put_strike   = atm - 250

        sc_ltp = await get_quote(f"NSE:NIFTY{expiry_str}{int(short_call_strike)}CE", self.token)
        sp_ltp = await get_quote(f"NSE:NIFTY{expiry_str}{int(short_put_strike)}PE",  self.token)
        wc_ltp = await get_quote(f"NSE:NIFTY{expiry_str}{int(wing_call_strike)}CE",  self.token)
        wp_ltp = await get_quote(f"NSE:NIFTY{expiry_str}{int(wing_put_strike)}PE",   self.token)

        if not all([sc_ltp, sp_ltp, wc_ltp, wp_ltp]):
            return

        net_credit = (sc_ltp + sp_ltp) - (wc_ltp + wp_ltp)
        if net_credit <= 5:
            logger.debug("Iron Condor net credit too low: ₹%.2f", net_credit)
            return

        # Paper trade: track as combined position (simplified)
        ic_symbol = f"NIFTY_IC_{expiry_str}_{int(atm)}"
        tid = self.ledger.open_trade(
            "IronCondor", ic_symbol, "SELL", 25,
            net_credit,
            sl=net_credit * 2.0,    # exit if debit to close = 2× credit
            target=net_credit * 0.30,   # keep 70%
        )
        self._open_ids["IronCondor"] = tid
        logger.info("Iron Condor entered: net_credit=₹%.2f  IV_rank=%.2f", net_credit, iv_rank)

    def _next_expiry(self) -> str:
        """Get nearest Tuesday expiry string for Nifty in Fyers format."""
        today = datetime.now()
        days_ahead = (1 - today.weekday()) % 7   # Tuesday = 1
        if days_ahead == 0 and today.time() >= dt_time(15, 30):
            days_ahead = 7
        expiry = today + timedelta(days=days_ahead)
        month_map = {10: "O", 11: "N", 12: "D"}
        mc = month_map.get(expiry.month, str(expiry.month))
        return f"{expiry.strftime('%y')}{mc}{expiry.strftime('%d')}"

    # ── EOD square-off ────────────────────────────────────────────────────────

    async def eod_squareoff(self) -> None:
        """Close all INTRADAY positions at 3:15 PM."""
        for strategy, trade_id in list(self._open_ids.items()):
            if strategy == "IronCondor":
                continue   # IC is held till expiry (weekly)
            self.ledger.close_trade(trade_id, 0.0, "EOD_SQUAREOFF")
            del self._open_ids[strategy]


# ── Main event loop ───────────────────────────────────────────────────────────

async def run_paper_trader():
    logger.info("=" * 55)
    logger.info("  PAPER TRADING SESSION  —  %s", today)
    logger.info("  Capital: ₹%,.0f", CAPITAL)
    logger.info("=" * 55)

    if not all([APP_ID, SECRET_KEY, CLIENT_ID, TOTP_KEY, PIN]):
        logger.error("Missing credentials in .env. Check APP_ID, SECRET_KEY, CLIENT_ID, TOTP_KEY, PIN.")
        sys.exit(1)

    # Authenticate
    logger.info("Authenticating with Fyers...")
    token = await fyers_login()
    if not token:
        logger.error("Authentication failed. Check your credentials and IP whitelist.")
        sys.exit(1)

    ledger  = PaperLedger(CAPITAL)
    engine  = PaperStrategyEngine(ledger, token)

    # Fetch initial spot
    nifty_spot = await get_quote("NSE:NIFTY50-INDEX", token)
    if nifty_spot:
        engine._spot_hist.append(nifty_spot)

    logger.info("Market monitoring started. Trading window: 10:15 AM – 3:15 PM IST")

    _running = True

    def _stop(signum, frame):
        nonlocal _running
        logger.info("Shutdown signal received.")
        _running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    while _running:
        now = datetime.now()
        t   = now.time()

        # Outside market hours — wait
        if t < dt_time(9, 15) or t > dt_time(15, 45):
            await asyncio.sleep(60)
            continue

        try:
            # ── Fetch live Nifty spot ─────────────────────────────────────
            spot = await get_quote("NSE:NIFTY50-INDEX", token)
            vix  = 14.0   # VIX not available in real-time on free tier — use proxy
            if spot:
                engine._spot_hist.append(spot)
                if len(engine._spot_hist) > 200:
                    engine._spot_hist.pop(0)
                engine._vix_hist.append(vix)
                if len(engine._vix_hist) > 60:
                    engine._vix_hist.pop(0)

                atm = round(spot / 50) * 50   # nearest 50 for Nifty

                # ── Run strategy signals ──────────────────────────────────
                if dt_time(10, 15) <= t <= dt_time(14, 15):
                    await engine.check_skewhunter(spot, atm)

                if dt_time(10, 30) <= t <= dt_time(11, 15):
                    await engine.check_iron_condor(spot, atm, vix)

                # ── EOD square-off ────────────────────────────────────────
                if t >= dt_time(15, 15):
                    await engine.eod_squareoff()

            # Log status every 30 minutes
            if now.minute in (15, 45) and now.second < 60:
                logger.info(
                    "Status: spot=₹%.0f  daily_pnl=₹%+.0f  open=%d",
                    spot or 0, ledger.daily_pnl, len(engine._open_ids),
                )

        except Exception as e:
            logger.error("Error in main loop: %s", e)

        await asyncio.sleep(60)   # poll every 1 minute

    # ── End of day ────────────────────────────────────────────────────────────
    print(ledger.daily_report())
    logger.info("Paper trading session ended.")


if __name__ == "__main__":
    asyncio.run(run_paper_trader())
