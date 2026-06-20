#!/usr/bin/env python3
"""
probe_fyers_api.py — Find the correct Fyers historical data endpoint.

Tries multiple known URL patterns and shows what each returns.
Run once; take note of which URL gives {"s":"ok",...}.
"""
import asyncio, os
from dotenv import load_dotenv
import httpx

load_dotenv(override=True)
try:
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            line = line.removeprefix("export").strip()
            if "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
except FileNotFoundError:
    pass

APP_ID = os.environ.get("BROKER_APP_ID", "")
TOKEN  = os.environ.get("BROKER_ACCESS_TOKEN", "")

HEADERS = {"Authorization": f"{APP_ID}:{TOKEN}"}

# Test with just 2 days of Nifty data
PARAMS = {
    "symbol":      "NSE:NIFTY50-INDEX",
    "resolution":  "D",          # Daily (faster to test than 1-min)
    "date_format": "1",
    "range_from":  "2025-06-01",
    "range_to":    "2025-06-10",
    "cont_flag":   "1",
}

ENDPOINTS = [
    ("api-t1  /api/v3/history",       "https://api-t1.fyers.in/api/v3/history"),
    ("api-t2  /data/history",          "https://api-t2.fyers.in/data/history"),
    ("api-t1  /api/v2/history",        "https://api-t1.fyers.in/api/v2/history"),
    ("api-t2  /api/v3/history",        "https://api-t2.fyers.in/api/v3/history"),
    ("api     /v3/historical-candle",  "https://api.fyers.in/v3/historical-candle"),
    ("api-t1  /data/history",          "https://api-t1.fyers.in/data/history"),
]

async def main():
    print(f"App ID : {APP_ID[:6]}...{APP_ID[-4:]}")
    print(f"Token  : {TOKEN[:10]}...  (length={len(TOKEN)})\n")

    async with httpx.AsyncClient(timeout=10) as c:
        for label, url in ENDPOINTS:
            try:
                r = await c.get(url, headers=HEADERS, params=PARAMS)
                body = r.text[:120].replace("\n", " ")
                print(f"  [{r.status_code}] {label}")
                print(f"         {body}\n")
            except Exception as e:
                print(f"  [ERR] {label}: {e}\n")

asyncio.run(main())
