#!/usr/bin/env python3
"""
get_token_browser.py — Get Fyers access token via browser login.

Use when automated PIN auth fails. Browser login works fine.
Token is valid for the current trading day.

Usage:
    python3 get_token_browser.py
"""

import asyncio
import hashlib
import os
import sys
from urllib.parse import urlparse, parse_qs, quote

import httpx
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
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

APP_ID     = os.environ.get("BROKER_APP_ID",     "").strip()
SECRET_KEY = os.environ.get("BROKER_SECRET_KEY", "").strip()

if not APP_ID or not SECRET_KEY:
    print("ERROR: BROKER_APP_ID and BROKER_SECRET_KEY must be set in .env")
    sys.exit(1)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


async def main():
    print("=" * 65)
    print("  FYERS BROWSER-BASED TOKEN GENERATOR")
    print("=" * 65)

    # ── Confirm exact redirect URI ─────────────────────────────────────────────
    default_uri = os.environ.get("BROKER_REDIRECT_URI", "http://127.0.0.1:8080/callback").strip()
    print(f"""
  ACTION REQUIRED before we start:

  1. Open  myaccount.fyers.in → API → My Apps → (your app) → Edit
  2. Look at the "Redirect URI" field and copy it EXACTLY

  Default we will use: {default_uri}
""")
    override = input("  Press ENTER to use the default, OR paste the exact URI from Fyers: ").strip()
    redirect_uri = override if override else default_uri

    # URL-encode the redirect_uri for safe inclusion in query string
    redirect_encoded = quote(redirect_uri, safe="")

    # Build the Fyers OAuth URL
    auth_url = (
        f"https://api-t1.fyers.in/api/v3/generate-authcode"
        f"?client_id={APP_ID}"
        f"&redirect_uri={redirect_encoded}"
        f"&response_type=code"
        f"&state=None"
    )

    print(f"""
  Using redirect URI : {redirect_uri}
  App ID             : {APP_ID}

  ─────────────────────────────────────────────────────────────────

  STEP 1 — Copy this entire URL and open it in Chrome/Safari:

  {auth_url}

  ─────────────────────────────────────────────────────────────────

  STEP 2 — Log in with your Fyers ID + password + TOTP.
           (This is your normal web login — no API PIN asked here.)

  STEP 3 — After login, the browser will try to open:
              {redirect_uri}?auth_code=XXXXX&state=None&status=success

           The page WILL show "This site can't be reached" — that is
           completely normal. You do NOT need a local server running.

  STEP 4 — Look at your browser's ADDRESS BAR.
           Copy the FULL URL (it starts with {redirect_uri.split('?')[0]}?auth_code=)

  ─────────────────────────────────────────────────────────────────
""")

    redirect_url = input("  Paste the full redirect URL from the address bar: ").strip()

    # ── Parse auth_code out of the URL ────────────────────────────────────────
    try:
        parsed    = urlparse(redirect_url)
        params    = parse_qs(parsed.query)
        auth_code = params.get("auth_code", [None])[0]
        status    = params.get("status",    [""])[0]
    except Exception as e:
        print(f"\n  ERROR parsing URL: {e}")
        sys.exit(1)

    if status and status != "success":
        print(f"\n  Fyers returned status='{status}'.")
        if status == "error":
            print("  This usually means redirect URI mismatch.")
            print(f"  Check that  {redirect_uri}  is EXACTLY what is")
            print("  saved in your Fyers API app's Redirect URI field.")
            print("  (Check for http vs https, port, trailing slash)")
        sys.exit(1)

    if not auth_code:
        print("\n  ERROR: No auth_code found in the URL.")
        print("  Make sure you copied the full address-bar URL after login.")
        sys.exit(1)

    print(f"\n  auth_code found ({len(auth_code)} chars) — exchanging for access_token ...")

    # ── Exchange auth_code → access_token ──────────────────────────────────────
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            "https://api-t1.fyers.in/api/v3/validate-authcode",
            json={
                "grant_type": "authorization_code",
                "appIdHash":  _sha256(f"{APP_ID}:{SECRET_KEY}"),
                "code":       auth_code,
            },
        )
        d = r.json()

    if d.get("s") != "ok" or not d.get("access_token"):
        print(f"\n  ERROR: Token exchange failed: {d}")
        sys.exit(1)

    access_token = d["access_token"]
    print(f"  ✓ Access token obtained! (length={len(access_token)})")

    # ── Save token to .env ─────────────────────────────────────────────────────
    env_path = ".env"
    lines = []
    token_written = False
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith("export BROKER_ACCESS_TOKEN=") or \
                   line.startswith("BROKER_ACCESS_TOKEN="):
                    lines.append(f"export BROKER_ACCESS_TOKEN={access_token}\n")
                    token_written = True
                else:
                    lines.append(line)
    except FileNotFoundError:
        lines = []

    if not token_written:
        lines.append(f"export BROKER_ACCESS_TOKEN={access_token}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)

    print("  ✓ Token saved to .env as BROKER_ACCESS_TOKEN")
    print()
    print("  Now run the backtest:")
    print("  python3 run_backtest_with_token.py --all --months 12 --capital 500000")
    print()


asyncio.run(main())
