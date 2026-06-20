#!/usr/bin/env python3
"""
debug_fyers_auth.py — Step-by-step Fyers auth diagnostic.

Run this to find out exactly why PIN verification is failing.
Shows full API responses at each step without hiding anything.

Usage:
    python3 debug_fyers_auth.py
"""

import asyncio
import hashlib
import os
import time
from dotenv import load_dotenv

import httpx
import pyotp

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

VAGATOR = "https://api-t2.fyers.in/vagator/v2"

# ── Read credentials ───────────────────────────────────────────────────────────
CLIENT_ID = os.environ.get("BROKER_CLIENT_ID", "").strip()
TOTP_KEY  = os.environ.get("BROKER_TOTP_KEY",  "").strip()
PIN       = os.environ.get("BROKER_PIN",        "").strip()

print("=" * 60)
print("  FYERS AUTH DIAGNOSTIC")
print("=" * 60)

# ── PIN forensics ──────────────────────────────────────────────────────────────
print("\n[PIN Analysis]")
print(f"  Length      : {len(PIN)}")
print(f"  Digits only : {PIN.isdigit()}")
print(f"  Raw bytes   : {list(PIN.encode('utf-8'))}")          # exact bytes
print(f"  SHA256      : {hashlib.sha256(PIN.encode()).hexdigest()}")
print(f"  First char  : {repr(PIN[0]) if PIN else 'EMPTY'}")
print(f"  Last char   : {repr(PIN[-1]) if PIN else 'EMPTY'}")

if not PIN:
    print("\n  ERROR: BROKER_PIN is empty! Check your .env file.")
    exit(1)

if not PIN.isdigit():
    print(f"\n  WARNING: PIN contains non-digit characters: {[c for c in PIN if not c.isdigit()]}")

if len(PIN) != 4:
    print(f"\n  WARNING: Fyers PIN is 4 digits but yours has {len(PIN)} characters.")

# ── TOTP check ─────────────────────────────────────────────────────────────────
print("\n[TOTP Analysis]")
try:
    totp = pyotp.TOTP(TOTP_KEY.replace(" ", "").upper())
    remaining = 30 - (int(time.time()) % 30)
    print(f"  Current code  : {totp.now()}")
    print(f"  Valid for     : {remaining} more seconds")
    print(f"  Key length    : {len(TOTP_KEY)} chars")
except Exception as e:
    print(f"  ERROR: TOTP setup failed: {e}")
    exit(1)

print(f"\n[Client ID]  {CLIENT_ID}")

# ── Live API test ──────────────────────────────────────────────────────────────
async def run():
    print("\n" + "=" * 60)
    print("  LIVE API STEPS")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30) as c:

        # Step 1
        print("\n[Step 1] send_login_otp ...")
        r = await c.post(f"{VAGATOR}/send_login_otp",
                         json={"fy_id": CLIENT_ID, "app_id": "2"})
        d = r.json()
        print(f"  HTTP {r.status_code}")
        print(f"  Response: {d}")
        if d.get("s") != "ok":
            print("\n  STOPPED at step 1.")
            return
        rk = d["request_key"]

        # Step 2
        print("\n[Step 2] verify_otp (TOTP) ...")
        while True:
            remaining = 30 - (int(time.time()) % 30)
            if remaining >= 8:
                break
            print(f"  (waiting {remaining}s for fresh TOTP code)")
            await asyncio.sleep(remaining + 1)
        code = totp.now()
        print(f"  Sending TOTP code: {code}")
        r = await c.post(f"{VAGATOR}/verify_otp",
                         json={"request_key": rk, "otp": code})
        d = r.json()
        print(f"  HTTP {r.status_code}")
        print(f"  Response: {d}")
        if d.get("s") != "ok":
            print("\n  STOPPED at step 2.")
            return
        rk = d["request_key"]

        # Step 3 — try SHA256 first, show full details
        pin_sha256 = hashlib.sha256(PIN.encode()).hexdigest()
        print("\n[Step 3a] verify_pin with SHA256(PIN) ...")
        print(f"  Sending pin value (SHA256): {pin_sha256[:12]}...")
        payload = {"request_key": rk, "identity_type": "pin",
                   "recaptcha_token": "", "pin": pin_sha256}
        print(f"  Full payload: {payload}")
        r = await c.post(f"{VAGATOR}/verify_pin", json=payload)
        d = r.json()
        print(f"  HTTP {r.status_code}")
        print(f"  Response: {d}")

        if (d.get("data") or {}).get("token"):
            print("\n  ✓ SHA256 PIN worked!")
            return

        # Restart steps 1+2 for the plain-PIN retry
        print("\n  SHA256 failed. Restarting steps 1+2 for plain PIN retry ...")
        r2 = await c.post(f"{VAGATOR}/send_login_otp",
                          json={"fy_id": CLIENT_ID, "app_id": "2"})
        d2 = r2.json()
        print(f"  Step 1 again: {d2}")
        if d2.get("s") != "ok":
            print("  STOPPED — could not restart step 1 (account may be rate-limited).")
            return
        rk2 = d2["request_key"]

        while True:
            remaining = 30 - (int(time.time()) % 30)
            if remaining >= 8:
                break
            print(f"  (waiting {remaining}s for fresh TOTP)")
            await asyncio.sleep(remaining + 1)
        code2 = totp.now()
        print(f"  Step 2 again with TOTP: {code2}")
        r2 = await c.post(f"{VAGATOR}/verify_otp",
                          json={"request_key": rk2, "otp": code2})
        d2 = r2.json()
        print(f"  Step 2 response: {d2}")
        if d2.get("s") != "ok":
            print("  STOPPED at step 2 retry.")
            return
        rk2 = d2["request_key"]

        print("\n[Step 3b] verify_pin with PLAIN PIN ...")
        print(f"  Sending pin value (plain): {'*' * len(PIN)}  (length={len(PIN)})")
        payload2 = {"request_key": rk2, "identity_type": "pin",
                    "recaptcha_token": "", "pin": PIN}
        r = await c.post(f"{VAGATOR}/verify_pin", json=payload2)
        d = r.json()
        print(f"  HTTP {r.status_code}")
        print(f"  Response: {d}")

        if (d.get("data") or {}).get("token"):
            print("\n  ✓ Plain PIN worked!")
        else:
            print("\n  ✗ Both SHA256 and plain PIN failed.")
            print("\n  LIKELY CAUSES:")
            print("  1. Account is temporarily locked — Fyers locks for 30 min after")
            print("     3–5 wrong PIN attempts. Wait 30 min and try again.")
            print("  2. Redirect URI mismatch — check BROKER_REDIRECT_URI in .env")
            print("     matches exactly what is set in your Fyers API app settings.")
            print("  3. The PIN in .env has a hidden character (see byte list above).")

asyncio.run(run())
