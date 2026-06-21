"""
Live terminal dashboard using the `rich` library.

Panels:
  ┌──────────────────────┬────────────────────────────┐
  │  PnL Summary         │  Open Positions            │
  ├──────────────────────┼────────────────────────────┤
  │  Risk Metrics / VaR  │  Strategy Attribution      │
  ├──────────────────────┴────────────────────────────┤
  │  Drawdown Bar                                     │
  └───────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from algo_platform.core.types import Position, RiskState

logger = logging.getLogger("platform.monitoring.dashboard")

_RICH_AVAILABLE = False
try:
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    logger.warning("rich not installed — dashboard disabled (pip install rich)")


def _colour_pnl(val: float) -> str:
    """Return a rich markup string coloured green/red by sign."""
    s = f"₹{val:+,.2f}"
    return f"[green]{s}[/green]" if val >= 0 else f"[red]{s}[/red]"


def _colour_pct(val: float) -> str:
    s = f"{val:+.2%}"
    return f"[green]{s}[/green]" if val >= 0 else f"[red]{s}[/red]"


def _drawdown_bar(dd_pct: float, width: int = 40) -> str:
    """ASCII drawdown bar: red fill proportional to drawdown."""
    filled = int(width * min(dd_pct, 1.0))
    bar    = "█" * filled + "░" * (width - filled)
    colour = "red" if dd_pct > 0.10 else "yellow" if dd_pct > 0.05 else "green"
    return f"[{colour}]{bar}[/{colour}] {dd_pct:.1%}"


class TradingDashboard:
    """
    Real-time trading dashboard. Use `start()` to run the live update loop.
    Safe to call `update()` from any thread.
    """

    def __init__(self, refresh_rate: float = 1.0) -> None:
        self._refresh  = refresh_rate
        self._state:   Optional[RiskState]  = None
        self._positions: List[Position]     = []
        self._running  = False

        if _RICH_AVAILABLE:
            self._console = Console()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, state: RiskState, positions: List[Position]) -> None:
        """Thread-safe state update; dashboard picks it up on next render."""
        self._state     = state
        self._positions = positions

    def start(self) -> None:
        """Block and run the live dashboard until KeyboardInterrupt."""
        if not _RICH_AVAILABLE:
            logger.warning("rich not available — using text fallback")
            self._text_loop()
            return
        self._running = True
        with Live(self._render(), console=self._console,
                  refresh_per_second=int(1.0 / self._refresh)) as live:
            try:
                while self._running:
                    import time
                    time.sleep(self._refresh)
                    live.update(self._render())
            except KeyboardInterrupt:
                pass

    async def start_async(self) -> None:
        """Async variant for integration with asyncio event loops."""
        if not _RICH_AVAILABLE:
            self._text_loop()
            return
        self._running = True
        with Live(self._render(), console=self._console,
                  refresh_per_second=int(1.0 / self._refresh)) as live:
            try:
                while self._running:
                    await asyncio.sleep(self._refresh)
                    live.update(self._render())
            except asyncio.CancelledError:
                pass

    def stop(self) -> None:
        self._running = False

    def print_snapshot(self) -> None:
        """Single-shot print (useful for end-of-day reports)."""
        if _RICH_AVAILABLE:
            self._console.print(self._render())
        else:
            self._print_text()

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self):
        if not _RICH_AVAILABLE:
            return ""
        layout = Layout()
        layout.split_column(
            Layout(name="top",    ratio=3),
            Layout(name="bottom", ratio=1),
        )
        layout["top"].split_row(
            Layout(name="left",  ratio=1),
            Layout(name="right", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="pnl",  ratio=1),
            Layout(name="risk", ratio=1),
        )
        layout["right"].split_column(
            Layout(name="positions",   ratio=2),
            Layout(name="attribution", ratio=1),
        )

        layout["pnl"].update(self._pnl_panel())
        layout["risk"].update(self._risk_panel())
        layout["positions"].update(self._positions_panel())
        layout["attribution"].update(self._attribution_panel())
        layout["bottom"].update(self._drawdown_panel())
        return layout

    def _pnl_panel(self) -> "Panel":
        s = self._state
        t = Table.grid(padding=1)
        t.add_column(style="bold", justify="right")
        t.add_column()

        if s is None:
            t.add_row("Waiting for data…", "")
        else:
            t.add_row("NAV",          _colour_pnl(s.nav))
            t.add_row("Daily P&L",    _colour_pnl(s.daily_pnl))
            t.add_row("Weekly P&L",   _colour_pnl(s.weekly_pnl))
            t.add_row("Total P&L",    _colour_pnl(s.total_pnl))
            t.add_row("Open trades",  str(s.open_positions))
            ts = s.timestamp.strftime("%H:%M:%S") if s else "--"
            t.add_row("As of",        ts)

        return Panel(t, title="[bold cyan]P & L[/bold cyan]", border_style="cyan")

    def _risk_panel(self) -> "Panel":
        s = self._state
        t = Table.grid(padding=1)
        t.add_column(style="bold", justify="right")
        t.add_column()

        if s is None:
            t.add_row("Waiting…", "")
        else:
            t.add_row("Δ Delta",      f"{s.total_delta:+.2f}")
            t.add_row("Γ Gamma",      f"{s.total_gamma:+.4f}")
            t.add_row("Θ Theta",      f"{s.total_theta:+.2f}")
            t.add_row("V Vega",       f"{s.total_vega:+.2f}")
            t.add_row("VaR (99% 1d)", _colour_pnl(-s.daily_var_99))
            status = "[red]HALTED[/red]" if s.is_trading_halted else "[green]ACTIVE[/green]"
            t.add_row("Status",       status)

        return Panel(t, title="[bold yellow]Risk & Greeks[/bold yellow]",
                     border_style="yellow")

    def _positions_panel(self) -> "Panel":
        table = Table(show_header=True, header_style="bold magenta",
                      border_style="dim")
        table.add_column("Symbol",    width=20)
        table.add_column("Side",      width=5)
        table.add_column("Qty",       justify="right", width=6)
        table.add_column("Entry",     justify="right", width=8)
        table.add_column("Current",   justify="right", width=8)
        table.add_column("Unreal P&L",justify="right", width=12)
        table.add_column("Δ",         justify="right", width=7)

        for p in self._positions:
            pnl_str = _colour_pnl(p.unrealized_pnl)
            table.add_row(
                p.symbol[:20],
                p.side.value,
                str(p.quantity),
                f"{p.entry_price:.2f}",
                f"{p.current_price:.2f}",
                pnl_str,
                f"{p.delta:+.3f}",
            )

        if not self._positions:
            table.add_row("[dim]No open positions[/dim]",
                          "", "", "", "", "", "")

        return Panel(table, title="[bold magenta]Open Positions[/bold magenta]",
                     border_style="magenta")

    def _attribution_panel(self) -> "Panel":
        s = self._state
        t = Table.grid(padding=1)
        t.add_column(justify="right", style="bold")
        t.add_column()

        if s and s.strategy_pnl:
            for strat, pnl in sorted(s.strategy_pnl.items(),
                                     key=lambda x: -abs(x[1])):
                t.add_row(strat, _colour_pnl(pnl))
        else:
            t.add_row("[dim]No attribution data[/dim]", "")

        return Panel(t, title="[bold white]Strategy Attribution[/bold white]",
                     border_style="white")

    def _drawdown_panel(self) -> "Panel":
        s = self._state
        if s is None:
            bar = "[dim]Awaiting data[/dim]"
        else:
            bar = _drawdown_bar(s.current_drawdown)
            bar = f"Current DD  {bar}    (Max: {s.max_drawdown:.1%})"

        return Panel(bar, title="[bold red]Drawdown[/bold red]",
                     border_style="red")

    # ── Plain-text fallback ────────────────────────────────────────────────────

    def _print_text(self) -> None:
        s = self._state
        if s is None:
            print("[Dashboard] No state yet.")
            return
        print(
            f"[{s.timestamp:%H:%M:%S}] "
            f"NAV=₹{s.nav:,.0f}  Daily=₹{s.daily_pnl:+,.0f}  "
            f"DD={s.current_drawdown:.1%}  VaR=₹{s.daily_var_99:,.0f}  "
            f"Positions={s.open_positions}"
        )

    def _text_loop(self) -> None:
        import time
        try:
            while self._running:
                self._print_text()
                time.sleep(self._refresh)
        except KeyboardInterrupt:
            pass
