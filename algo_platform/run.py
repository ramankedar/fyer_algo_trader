"""
Platform entry point.

Modes
-----
  download    — fetch & cache real OHLCV data from Fyers for all instruments
  research    — run feature research on historical data
  backtest    — run a single-strategy backtest with real underlying + synthetic chains
  walkforward — walk-forward optimisation for a strategy
  paper       — paper-trading loop (no real orders)

Usage (from repo root)
----------------------
  # 1. Download 5 years of data first (run once)
  python -m algo_platform.run download --start 2020-01-01

  # 2. Research features
  python -m algo_platform.run research --instrument NIFTY --start 2022-01-01 --end 2025-01-01

  # 3. Backtest
  python -m algo_platform.run backtest --strategy A --instrument NIFTY \\
                                       --start 2022-01-01 --end 2025-01-01

  # 4. Walk-forward
  python -m algo_platform.run walkforward --strategy B --instrument BANKNIFTY

  # 5. Paper trade
  python -m algo_platform.run paper --instrument NIFTY
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime
from typing import Dict, List, Optional

from algo_platform.core.config import load_config, PlatformConfig
from algo_platform.core.types import Instrument, MarketBar

logger = logging.getLogger("algo_platform.run")

# Absolute path of the project root (one level above this file's package).
# Needed so data/cache/ is found regardless of the working directory.
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO", log_file: str = "platform.log") -> None:
    from logging.handlers import RotatingFileHandler
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Resolve log file to project root so logs aren't scattered
    import os
    abs_log = log_file if os.path.isabs(log_file) else os.path.join(
        _PROJECT_ROOT, log_file
    )
    fh = RotatingFileHandler(abs_log, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _make_downloader(config: PlatformConfig):
    """
    Construct a FyersDownloader using credentials from .env.
    Cache dir is always resolved to an absolute path so the platform can be
    run from any working directory — not just the project root.
    Priority: ALGO_CACHE_DIR env var → project_root/data/cache
    """
    import os
    from algo_platform.data.downloader import FyersDownloader

    cache_dir = os.getenv("ALGO_CACHE_DIR") or os.path.join(
        _PROJECT_ROOT, "data", "cache"
    )
    return FyersDownloader(
        client_id    = config.broker.app_id,
        access_token = config.broker.access_token,
        cache_dir    = cache_dir,
    )


def _make_loader(config: PlatformConfig):
    """Construct a MarketDataLoader wrapping the downloader."""
    from algo_platform.data.loader import MarketDataLoader
    return MarketDataLoader(_make_downloader(config))


def _load_bars_and_vix(
    instrument: str,
    start:      str,
    end:        str,
    config:     PlatformConfig,
) -> tuple[List[MarketBar], Dict[date, float]]:
    """Load 1-min bars + VIX daily series. Uses cache; downloads only what's missing."""
    loader    = _make_loader(config)
    start_d   = date.fromisoformat(start)
    end_d     = date.fromisoformat(end)

    bars = loader.load_bars(instrument, start_d, end_d, resolution="1")
    vix  = loader.load_vix(start_d, end_d)

    if not bars:
        logger.error(
            "No bars for %s [%s→%s]. Run 'download' first.", instrument, start, end
        )
    return bars, vix


# ── Mode: download ─────────────────────────────────────────────────────────────

def run_download(args, config: PlatformConfig) -> None:
    """
    Download and cache real OHLCV data for all instruments + India VIX.
    Safe to run repeatedly — only fetches missing date ranges.
    """
    from algo_platform.data.downloader import FyersDownloader, FYERS_SYMBOLS

    dl        = _make_downloader(config)
    start_d   = date.fromisoformat(args.start)
    end_d     = date.fromisoformat(args.end)
    # Resolve instrument list
    if args.instrument and args.instrument.upper() == "VIX":
        # VIX-only mode: just fetch daily
        print(f"\nDownloading VIX (daily): {start_d} → {end_d}\n")
        df_vix = dl.download("VIX", start_d, end_d, resolution="D")
        print(f"  VIX: {len(df_vix)} daily bars\n")
        for fname, info in dl.status().items():
            print(f"  {fname}: {info}")
        return

    instruments = (
        [args.instrument.upper()] if args.instrument
        else ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX", "BANKEX", "BSEIT"]
    )

    print(f"\nDownloading data: {start_d} → {end_d}")
    print(f"Instruments: {instruments}  +  VIX (daily)")
    print(f"Cache dir  : data/cache/\n")

    # Download India VIX (daily) first
    print("--- India VIX (daily) ---")
    df_vix = dl.download("VIX", start_d, end_d, resolution="D")
    print(f"  VIX: {len(df_vix)} daily bars\n")

    # Download 1-min OHLCV for each index
    for inst in instruments:
        print(f"--- {inst} (1-min) ---")
        df = dl.download(inst, start_d, end_d, resolution="1")
        days = df.index.normalize().nunique() if not df.empty else 0
        print(f"  {inst}: {len(df):,} bars across {days} trading days\n")

    # Print cache status
    print("=== Cache status ===")
    for fname, info in dl.status().items():
        print(f"  {fname}: {info}")


# ── Mode: research ─────────────────────────────────────────────────────────────

def run_research(args, config: PlatformConfig) -> None:
    """Run the feature research pipeline on real downloaded data."""
    import numpy as np
    import pandas as pd
    from algo_platform.research.engine import ResearchEngine
    from algo_platform.research.features import FeatureEngine

    bars, vix = _load_bars_and_vix(args.instrument, args.start, args.end, config)
    if not bars:
        return

    inst   = Instrument(args.instrument.upper())
    lsize  = config.lot_size(inst.value)
    feat_engine = FeatureEngine(
        instrument        = inst.value,
        lot_size          = lsize,
        atr_period        = config.research.atr_period,
        rv_window         = config.research.rv_window,
        percentile_window = config.research.percentile_window,
        iv_rank_window    = config.research.iv_rank_window,
    )

    # Forward-fill VIX to get IV per date
    def _iv(d: date) -> float:
        # walk back up to 5 days for a VIX value
        for k in range(5):
            from datetime import timedelta
            v = vix.get(d - timedelta(days=k))
            if v:
                return v / 100.0
        return 0.15

    records = []
    current_date = None
    feat_engine.new_session()
    for i, bar in enumerate(bars):
        if bar.timestamp.date() != current_date:
            current_date = bar.timestamp.date()
            feat_engine.new_session()

        # Use VIX as ATM IV proxy for options features
        breadth = 0.5   # real breadth requires constituent data
        atm_iv  = _iv(bar.timestamp.date())

        # Build a lightweight chain just for option-specific features
        # (skipped here for speed — features computed from underlying only)
        fv = feat_engine.update(bar, None, breadth)
        if fv is None:
            continue

        fwd = ((bars[i + 1].close - bar.close) / bar.close
               if i + 1 < len(bars) else 0.0)
        row = {n: getattr(fv, n) for n in fv.FEATURE_NAMES}
        row["forward_return"] = fwd
        records.append(row)

    if not records:
        logger.error("No feature records generated.")
        return

    df     = pd.DataFrame(records)
    engine = ResearchEngine(
        min_ic_abs = config.research.min_ic_abs,
        min_t_stat = config.research.min_t_stat,
        max_pvalue = config.research.max_pvalue,
        min_mi     = config.research.min_mi,
    )
    stats  = engine.run_full_research(df)
    engine.print_report(stats)
    sig = engine.select_features(stats)
    print(f"\nDeploy-ready features: {sig}")


# ── Mode: backtest ─────────────────────────────────────────────────────────────

def run_backtest(args, config: PlatformConfig) -> None:
    """Single-strategy backtest using real underlying data + synthetic BS chains."""
    from algo_platform.backtest.engine import BacktestEngine
    from algo_platform.data.chain_builder import SyntheticChainBuilder
    from algo_platform.strategies import (
        VolatilityCompressionStrategy, TrendFollowingStrategy,
        GammaExpansionStrategy, ShortStraddleStrategy,
        IronCondorStrategy, ShortStrangleStrategy,
        IronButterflyStrategy, AdaptiveStrangleStrategy,
    )

    bars, vix = _load_bars_and_vix(args.instrument, args.start, args.end, config)
    if not bars:
        return

    inst     = Instrument(args.instrument.upper())
    cls_map  = {
        "A": VolatilityCompressionStrategy,
        "B": TrendFollowingStrategy,
        "C": GammaExpansionStrategy,
        "D": ShortStraddleStrategy,    # Short expiry-day straddle (SELL premium)
        "E": IronCondorStrategy,       # Weekly iron condor (SELL spread)
        "F": ShortStrangleStrategy,    # Short OTM strangle (improved D, wider buffer)
        "G": IronButterflyStrategy,    # Short iron butterfly (defined-risk straddle)
        "H": AdaptiveStrangleStrategy, # Adaptive strangle (all improvements combined)
    }
    cls = cls_map.get(args.strategy.upper())
    if cls is None:
        logger.error("Unknown strategy '%s'. Choose A-E.", args.strategy)
        return

    strategy      = cls(inst, config, quantity=1)
    chain_builder = SyntheticChainBuilder(risk_free_rate=config.risk_free_rate)
    engine        = BacktestEngine(config)

    print(f"\nRunning backtest: Strategy {args.strategy} | {inst.value} "
          f"| {args.start} → {args.end}")
    print(f"Bars: {len(bars):,}  |  VIX dates: {len(vix)}")

    report = engine.run(
        strategy      = strategy,
        bars          = bars,
        chain_builder = chain_builder,
        vix_by_date   = vix,
    )

    print("\n" + "=" * 60)
    print(report.summary())
    print("=" * 60)


# ── Mode: walkforward ─────────────────────────────────────────────────────────

def run_walkforward(args, config: PlatformConfig) -> None:
    """Walk-forward optimisation with Bayesian parameter search."""
    from algo_platform.backtest.walk_forward import WalkForwardOptimizer
    from algo_platform.optimization.bayesian_opt import (
        STRATEGY_A_SPACE, STRATEGY_B_SPACE, STRATEGY_C_SPACE,
    )
    from algo_platform.strategies import (
        VolatilityCompressionStrategy, TrendFollowingStrategy, GammaExpansionStrategy,
    )

    bars, _ = _load_bars_and_vix(args.instrument, args.start, args.end, config)
    if not bars:
        return

    inst     = Instrument(args.instrument.upper())
    cls_map  = {"A": (VolatilityCompressionStrategy, STRATEGY_A_SPACE),
                "B": (TrendFollowingStrategy,        STRATEGY_B_SPACE),
                "C": (GammaExpansionStrategy,        STRATEGY_C_SPACE)}
    cls, space = cls_map.get(args.strategy.upper(),
                             (VolatilityCompressionStrategy, STRATEGY_A_SPACE))

    optimizer = WalkForwardOptimizer(
        config           = config,
        strategy_factory = lambda i, c: cls(i, c),
        instrument       = inst,
        param_space      = space,
        n_trials         = getattr(args, "trials", 50),
    )
    result = optimizer.run(bars)
    print("\n" + result.summary())


# ── Mode: paper ───────────────────────────────────────────────────────────────

async def run_paper(args, config: PlatformConfig) -> None:
    """Paper-trading event loop using live Fyers WebSocket data."""
    from algo_platform.monitoring.dashboard import TradingDashboard
    from algo_platform.risk.manager import PlatformRiskManager

    dashboard    = TradingDashboard(config.monitoring.refresh_rate)
    risk_manager = PlatformRiskManager(config)

    logger.info("Paper trading: %s. Ctrl+C to stop.", args.instrument)
    print("Paper trading — connect your data feed to extend this loop.")

    dash_task = asyncio.create_task(dashboard.start_async())
    try:
        while True:
            await asyncio.sleep(5)
            state = risk_manager.on_bar(datetime.now(), [])
            dashboard.update(state, [])
    except asyncio.CancelledError:
        pass
    finally:
        dashboard.stop()
        dash_task.cancel()


# ── Mode: token refresh ───────────────────────────────────────────────────────

def run_refresh_token(config: PlatformConfig) -> None:
    """
    Refresh the Fyers access token via browser OAuth and save it to .env.

    Fyers tokens expire at midnight IST every day. Run this once per day
    before backtesting or live trading.

    Flow:
      1.  Opens a URL → you log in with Fyers credentials + TOTP in browser
      2.  Browser redirects to localhost (page won't load — that's expected)
      3.  Copy the full URL from the address bar and paste it here
      4.  Token is saved to .env automatically
    """
    import hashlib, re
    from urllib.parse import urlparse, parse_qs, quote
    import httpx

    env_path  = _os.path.join(_PROJECT_ROOT, ".env")
    app_id    = config.broker.app_id
    secret    = config.broker.secret_key
    redir_uri = "http://127.0.0.1:8080/callback"

    if not app_id or not secret:
        print("ERROR: BROKER_APP_ID and BROKER_SECRET_KEY must be set in .env")
        return

    redir_encoded = quote(redir_uri, safe="")
    auth_url = (
        f"https://api-t1.fyers.in/api/v3/generate-authcode"
        f"?client_id={app_id}"
        f"&redirect_uri={redir_encoded}"
        f"&response_type=code"
        f"&state=None"
    )

    print("=" * 65)
    print("  FYERS TOKEN REFRESH (browser flow)")
    print("=" * 65)
    print()
    print("STEP 1 — Open this URL in Chrome/Safari:")
    print()
    print(f"  {auth_url}")
    print()
    print("STEP 2 — Log in with your Fyers ID + password + TOTP.")
    print()
    print("STEP 3 — After login the browser goes to a 'site can't be reached'")
    print("         page. That is NORMAL. Look at the address bar — it starts with:")
    print(f"         {redir_uri}?auth_code=...")
    print()
    print("STEP 4 — Copy the FULL URL from the address bar and paste it below.")
    print()

    redirect_url = input("  Paste redirect URL here: ").strip()
    if not redirect_url:
        print("Cancelled.")
        return

    try:
        params    = parse_qs(urlparse(redirect_url).query)
        auth_code = params.get("auth_code", [None])[0]
        status    = params.get("status", [""])[0]
    except Exception as exc:
        print(f"ERROR parsing URL: {exc}")
        return

    if status and status != "success":
        print(f"ERROR: Fyers returned status='{status}'. Check redirect URI matches exactly.")
        return

    if not auth_code:
        print("ERROR: No auth_code found in the URL. Did you copy the full address bar URL?")
        return

    print(f"\nExchanging auth_code for access_token…")

    try:
        # verify=False works around macOS Python.org SSL cert issue
        with httpx.Client(verify=False, timeout=30) as c:
            r = c.post(
                "https://api-t1.fyers.in/api/v3/validate-authcode",
                json={
                    "grant_type": "authorization_code",
                    "appIdHash":  hashlib.sha256(f"{app_id}:{secret}".encode()).hexdigest(),
                    "code":       auth_code,
                },
            )
            d = r.json()

        if d.get("s") != "ok" or not d.get("access_token"):
            print(f"ERROR: Token exchange failed: {d}")
            return

        new_token = d["access_token"]
        print(f"  Access token obtained (length={len(new_token)})")

        # Write back to .env
        if _os.path.exists(env_path):
            text = open(env_path).read()
            if "BROKER_ACCESS_TOKEN" in text:
                text = re.sub(
                    r'^export BROKER_ACCESS_TOKEN=.*$',
                    f'export BROKER_ACCESS_TOKEN={new_token}',
                    text, flags=re.MULTILINE,
                )
            else:
                text += f'\nexport BROKER_ACCESS_TOKEN={new_token}\n'
            open(env_path, "w").write(text)
            print(f"  .env updated: {env_path}")
        else:
            print(f"  .env not found — add this line manually:")
            print(f"  export BROKER_ACCESS_TOKEN={new_token}")

        print()
        print("Done. Now run:")
        print(f"  source {env_path}")
        print("  python3 -m algo_platform.run download --start 2020-01-01")

    except Exception as exc:
        print(f"ERROR: {exc}")


# ── Mode: realbacktest ────────────────────────────────────────────────────────

def run_realbacktest(args, config: PlatformConfig) -> None:
    """
    Backtest Strategy C using REAL NSE option settlement prices.
    Downloads NSE bhavcopy (free) for expiry-day option prices.
    Exit price = 100% real NSE settlement.
    Entry price = real VIX × Black-Scholes (much more accurate than synthetic).
    """
    from algo_platform.data.real_options import NseBhavcopDownloader, RealOptionsStrategyC
    from algo_platform.data.downloader import FyersDownloader
    from algo_platform.data.loader import MarketDataLoader

    inst        = args.instrument.upper()
    start_d     = date.fromisoformat(args.start)
    end_d       = date.fromisoformat(args.end)
    bhavcopy_dir= _os.path.join(_PROJECT_ROOT, "nse_option_cache")

    print(f"\nReal-data backtest: Strategy C | {inst} | {start_d} → {end_d}")
    print(f"Step 1: Downloading NSE bhavcopy (expiry-day option prices)...")

    from algo_platform.core.config import LOT_SIZES
    spec = LOT_SIZES.get(inst)
    if spec is None:
        print(f"ERROR: Unknown instrument {inst}")
        return

    dl_bhav = NseBhavcopDownloader(bhavcopy_dir)
    n = dl_bhav.download_range(start_d, end_d, expiry_weekday=spec.expiry_weekday)
    print(f"  {n} new bhavcopy files downloaded.\n")

    print("Step 2: Loading NIFTY/underlying bars and VIX from cache...")
    fyers_dl = FyersDownloader(config.broker.app_id, config.broker.access_token,
                               _os.path.join(_PROJECT_ROOT, "data", "cache"))
    ldr      = MarketDataLoader(fyers_dl)
    bars     = ldr.load_bars(inst, start_d, end_d, "1")
    vix      = ldr.load_vix(start_d, end_d)

    if not bars:
        print(f"ERROR: No bars for {inst}. Run 'download' first.")
        return

    print(f"  Bars: {len(bars):,}   VIX days: {len(vix)}\n")
    print("Step 3: Running real-data backtest...")

    backtester = RealOptionsStrategyC(config, bhavcopy_dir)
    report     = backtester.run(inst, bars, vix, lots=1)

    print()
    print(report.summary())

    if report.trades:
        print("\nSample trades:")
        for t in report.trades[:5]:
            src = "✓real" if t.get("exit_source") == "bhavcopy" else "~est"
            print(f"  {t['date']} | ATM={t['atm_strike']:.0f} | "
                  f"move={t['move_pts']:.0f}pts | "
                  f"entry=₹{t['entry_per_sh']:.0f}/sh exit=₹{t['exit_per_sh']:.0f}/sh "
                  f"[{src}] | P&L=₹{t['pnl']:+,.0f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Algo Options Trading Platform — Indian Markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'download' first, then 'backtest' or 'research'.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # download
    p_dl = sub.add_parser("download",
                          help="Fetch & cache real OHLCV from Fyers (run once)")
    p_dl.add_argument("--instrument", default=None,
                      help="Single instrument (default: all — NIFTY, BANKNIFTY, FINNIFTY)")
    p_dl.add_argument("--start",      default="2020-01-01",
                      help="Start date YYYY-MM-DD (default: 2020-01-01 = 5 years)")
    p_dl.add_argument("--end",        default=str(date.today()))

    # research
    p_res = sub.add_parser("research", help="Feature research report")
    p_res.add_argument("--instrument", default="NIFTY")
    p_res.add_argument("--start",      default="2022-01-01")
    p_res.add_argument("--end",        default=str(date.today()))

    # backtest
    p_bt = sub.add_parser("backtest", help="Single-strategy backtest")
    p_bt.add_argument("--strategy",   required=True, choices=["A","B","C","D","E","F","G","H"])
    p_bt.add_argument("--instrument", default="NIFTY",
                      choices=["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX", "BANKEX", "BSEIT"])
    p_bt.add_argument("--start",      default="2022-01-01")
    p_bt.add_argument("--end",        default=str(date.today()))

    # walkforward
    p_wf = sub.add_parser("walkforward", help="Walk-forward optimisation")
    p_wf.add_argument("--strategy",   required=True, choices=["A", "B", "C"])
    p_wf.add_argument("--instrument", default="NIFTY")
    p_wf.add_argument("--start",      default="2020-01-01")
    p_wf.add_argument("--end",        default=str(date.today()))
    p_wf.add_argument("--trials",     type=int, default=50)

    # paper
    p_pp = sub.add_parser("paper", help="Paper-trading mode")
    p_pp.add_argument("--instrument", default="NIFTY")

    # realbacktest — Strategy C with real NSE option prices
    p_rb = sub.add_parser("realbacktest",
                          help="Backtest Strategy C with real NSE option settlement prices")
    p_rb.add_argument("--instrument", default="NIFTY",
                      choices=["NIFTY", "BANKNIFTY", "FINNIFTY"])
    p_rb.add_argument("--start",      default="2022-01-01")
    p_rb.add_argument("--end",        default=str(date.today()))

    # token — refresh Fyers access token and update .env
    sub.add_parser("token", help="Refresh Fyers access token (run daily before trading)")

    args   = parser.parse_args()
    config = load_config()
    _setup_logging(config.monitoring.log_level, config.monitoring.log_file)

    if   args.mode == "download":     run_download(args, config)
    elif args.mode == "research":     run_research(args, config)
    elif args.mode == "backtest":     run_backtest(args, config)
    elif args.mode == "walkforward":  run_walkforward(args, config)
    elif args.mode == "paper":        asyncio.run(run_paper(args, config))
    elif args.mode == "realbacktest": run_realbacktest(args, config)
    elif args.mode == "token":        run_refresh_token(config)


if __name__ == "__main__":
    main()
