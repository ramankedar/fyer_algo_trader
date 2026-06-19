#!/usr/bin/env python3
"""
run_real_backtest.py — Authenticate with Fyers and run the real data backtest.

Reads all credentials from .env — no manual copy-paste needed.
Supports exporting a .export file so CI/EC2 can re-use the token.

Usage (local):
    python3 run_real_backtest.py
    python3 run_real_backtest.py --instrument banknifty
    python3 run_real_backtest.py --instrument sensex --months 12
    python3 run_real_backtest.py --all          # Nifty + BankNifty + Sensex

Usage (EC2 — run inside tmux):
    source .env && python3 run_real_backtest.py --all --months 12
"""

import argparse
import asyncio
import hashlib
import os
import sys
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse, parse_qs

import httpx
from dotenv import load_dotenv

# ── Load .env (handles both `KEY=VAL` and `export KEY=VAL` formats) ──────────
load_dotenv(override=True)
# Also handle the `export KEY=VALUE` format that shell-style .env files use
_raw_env = {}
try:
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export").strip()
            if "=" in line:
                k, _, v = line.partition("=")
                _raw_env[k.strip()] = v.strip().strip('"').strip("'")
    os.environ.update(_raw_env)
except FileNotFoundError:
    pass


def _require(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        print(f"  ERROR: {key} is not set in .env")
        sys.exit(1)
    return val


# ══════════════════════════════════════════════════════════════════════════════
# Fyers 5-step headless authentication
# ══════════════════════════════════════════════════════════════════════════════

VAGATOR = "https://api-t2.fyers.in/vagator/v2"
API_V3  = "https://api-t1.fyers.in/api/v3"


async def _fyers_login(
    client_id:   str,
    app_id:      str,
    secret_key:  str,
    totp_key:    str,
    pin:         str,
    redirect_uri: str,
) -> Optional[str]:
    """Run the 5-step Fyers automated login. Returns access_token or None."""
    import pyotp

    def sha256(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    totp = pyotp.TOTP(totp_key.replace(" ", "").upper())

    async with httpx.AsyncClient(timeout=30) as c:
        # Step 1 — send_login_otp
        print("  [1/5] Sending login OTP ...", end=" ", flush=True)
        r = await c.post(f"{VAGATOR}/send_login_otp",
                         json={"fy_id": client_id, "app_id": "2"})
        d = r.json()
        if d.get("s") != "ok":
            print(f"FAILED: {d}")
            return None
        rk = d["request_key"]
        print("OK")

        # Step 2 — verify_otp (TOTP)
        print("  [2/5] Verifying TOTP ...", end=" ", flush=True)
        # Wait for a TOTP code that has at least 8 seconds left
        import time
        while True:
            remaining = 30 - (int(time.time()) % 30)
            if remaining >= 8:
                break
            print(f"(waiting {remaining}s for fresh code)", end=" ", flush=True)
            await asyncio.sleep(remaining + 1)
        otp_code = totp.now()
        r = await c.post(f"{VAGATOR}/verify_otp",
                         json={"request_key": rk, "otp": otp_code})
        d = r.json()
        if d.get("s") != "ok":
            print(f"FAILED: {d}")
            return None
        rk = d["request_key"]
        print("OK")

        # Step 3 — verify_pin
        # Fyers API accepts either plain PIN or SHA-256 hashed PIN depending on
        # the account / API version. Try plain first, fall back to hashed.
        print("  [3/5] Verifying PIN ...", end=" ", flush=True)
        session_token = None
        for pin_val in [pin, sha256(pin)]:
            r = await c.post(f"{VAGATOR}/verify_pin",
                             json={"request_key": rk,
                                   "identity_type": "pin",
                                   "recaptcha_token": "",
                                   "pin": pin_val})
            d = r.json()
            session_token = (d.get("data") or {}).get("token")
            if session_token:
                break
        if not session_token:
            print(f"FAILED: {d}")
            print("\n  Troubleshooting checklist:")
            print("  1. Open Fyers app/website and confirm your PIN works there")
            print("  2. Check .env: BROKER_PIN must be your 4-digit trading PIN")
            print("     (NOT your mobile OTP, NOT your app password)")
            print("  3. Run:  grep BROKER_PIN .env  — make sure there are no")
            print("     extra spaces, quotes, or 'export' left in the value")
            return None
        print("OK")

        # Step 4 — generate-authcode
        print("  [4/5] Generating auth code ...", end=" ", flush=True)
        r = await c.post(
            f"{API_V3}/generate-authcode",
            headers={"Authorization": session_token},
            json={"fyers_id": client_id, "app_id": app_id,
                  "redirect_uri": redirect_uri, "appType": "100",
                  "code_challenge": "", "state": "None",
                  "scope": "", "nonce": "", "response_type": "code",
                  "create_cookie": True},
        )
        d = r.json()
        url = d.get("Url", "")
        if "auth_code=" not in url:
            print(f"FAILED: {d}")
            return None
        auth_code = parse_qs(urlparse(url).query).get("auth_code", [None])[0]
        if not auth_code:
            print(f"FAILED: could not parse auth_code from {url}")
            return None
        print("OK")

        # Step 5 — validate-authcode → access_token
        print("  [5/5] Exchanging auth code for access token ...", end=" ", flush=True)
        app_id_hash = sha256(f"{app_id}:{secret_key}")
        r = await c.post(f"{API_V3}/validate-authcode",
                         json={"grant_type": "authorization_code",
                               "appIdHash": app_id_hash,
                               "code": auth_code})
        d = r.json()
        token = d.get("access_token")
        if not token:
            print(f"FAILED: {d}")
            return None
        print("OK")
        return token


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

async def _main(args) -> None:
    print(f"\n{'━'*60}")
    print("  REAL DATA BACKTEST — Fyers Historical API")
    print(f"{'━'*60}\n")

    # ── Authenticate ─────────────────────────────────────────────────────────
    client_id   = _require("BROKER_CLIENT_ID")
    app_id      = _require("BROKER_APP_ID")
    secret_key  = _require("BROKER_SECRET_KEY")
    totp_key    = _require("BROKER_TOTP_KEY")
    pin         = _require("BROKER_PIN")
    redirect    = os.environ.get("BROKER_REDIRECT_URI", "https://127.0.0.1:8080/callback")

    print("Authenticating with Fyers ...")
    print(f"  client_id length : {len(client_id)}")
    print(f"  app_id           : {app_id[:6]}...{app_id[-4:]}")
    print(f"  pin length       : {len(pin)} chars  ({'digits only' if pin.isdigit() else 'WARNING: contains non-digits'})")
    access_token = await _fyers_login(
        client_id, app_id, secret_key, totp_key, pin, redirect
    )
    if not access_token:
        print("\n  Authentication failed. Check credentials in .env")
        sys.exit(1)

    print(f"\n  Access token obtained (length={len(access_token)})")

    # Inject into environment so backtest.py can read it
    os.environ["BROKER_ACCESS_TOKEN"] = access_token

    # Optionally save so EC2 cron can reuse within same day
    if args.save_token:
        with open(".token_cache", "w") as f:
            f.write(access_token)
        print("  Token saved to .token_cache (valid until midnight)")

    # ── Run backtest(s) ───────────────────────────────────────────────────────
    from backtest import run_backtest

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=args.months * 30)).strftime("%Y-%m-%d")

    instruments = (
        ["nifty", "banknifty", "sensex"]
        if args.all
        else [args.instrument]
    )

    all_results = []
    for ikey in instruments:
        print(f"\n{'━'*60}")
        print(f"  Backtesting {ikey.upper()}  {start_date} → {end_date}")
        print(f"{'━'*60}")
        csv_out = f"real_{ikey}_{start_date}_{end_date}.csv"
        result  = await run_backtest(
            instrument_key=ikey,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            output_csv=csv_out,
        )
        all_results.append(result)

    # ── Comparative table if multiple instruments ─────────────────────────────
    if len(all_results) > 1:
        from backtest import _comparative_table  # noqa
        # _comparative_table is defined in backtest_offline context; reproduce here
        W = 70
        print("\n" + "═" * W)
        print("  COMPARATIVE SUMMARY")
        print("═" * W)
        print(f"  {'Instrument':<22} {'Trades':>6} {'WinRate':>8} "
              f"{'P&L':>12} {'Return':>8} {'MaxDD':>7} {'PF':>6}")
        print("  " + "─" * (W - 2))
        for r in all_results:
            if not r:
                continue
            print(f"  {r.get('label','')[:22]:<22} {r.get('trades',0):>6} "
                  f"{r.get('win_rate',0):>7.1f}% "
                  f"₹{r.get('total_pnl',0):>10,.0f} "
                  f"{r.get('return_pct',0):>+7.2f}% "
                  f"{r.get('max_dd',0):>6.2f}% "
                  f"{r.get('profit_factor',0):>6.2f}")
        print("═" * W + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Authenticate with Fyers and run real-data strategy backtest"
    )
    p.add_argument(
        "--instrument", default="nifty",
        choices=["nifty", "banknifty", "finnifty", "sensex", "bankex"],
        help="Single instrument to test (default: nifty)",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Run Nifty + BankNifty + Sensex sequentially",
    )
    p.add_argument(
        "--months", type=int, default=6,
        help="How many months of history to fetch (default: 6)",
    )
    p.add_argument(
        "--capital", type=float, default=500_000,
        help="Starting capital in INR (default: 500000)",
    )
    p.add_argument(
        "--save-token", action="store_true",
        help="Save today's access token to .token_cache for reuse",
    )
    args = p.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
