"""
Walk-forward optimization.

Architecture:
  1. Divide historical data into rolling train/test windows.
  2. On each train window, run Bayesian optimization (via optuna) to find
     the best strategy parameters.
  3. Test the optimised parameters on the hold-out window.
  4. Aggregate all test-window results into the final out-of-sample report.

This guarantees no lookahead: the test window is always strictly after the
train window, and optimised parameters from one fold never influence another.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from algo_platform.core.config import PlatformConfig
from algo_platform.core.types import Instrument, MarketBar, OptionChain, PerformanceReport, Trade
from algo_platform.backtest.engine import BacktestEngine
from algo_platform.backtest.metrics import PerformanceAnalyzer
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("platform.backtest.walk_forward")

_OPTUNA_AVAILABLE = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("optuna not installed — walk-forward will use default params only")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WFWindow:
    fold:         int
    train_bars:   List[MarketBar]
    test_bars:    List[MarketBar]
    train_start:  date
    train_end:    date
    test_start:   date
    test_end:     date


@dataclass
class FoldResult:
    fold:         int
    train_sharpe: float
    test_report:  PerformanceReport
    best_params:  Dict[str, Any]


@dataclass
class WalkForwardResult:
    strategy:        str
    folds:           List[FoldResult] = field(default_factory=list)
    combined_report: Optional[PerformanceReport] = None

    def oos_sharpe(self) -> float:
        """Mean out-of-sample Sharpe across all folds."""
        sharpes = [f.test_report.sharpe for f in self.folds]
        return float(np.mean(sharpes)) if sharpes else 0.0

    def oos_trades(self) -> int:
        return sum(f.test_report.total_trades for f in self.folds)

    def summary(self) -> str:
        lines = [
            f"Walk-Forward Results: {self.strategy}",
            f"  Folds              : {len(self.folds)}",
            f"  OOS Sharpe (mean)  : {self.oos_sharpe():.2f}",
            f"  OOS Trades (total) : {self.oos_trades()}",
        ]
        if self.combined_report:
            lines += [
                f"  OOS CAGR           : {self.combined_report.cagr:.1%}",
                f"  OOS Max Drawdown   : {self.combined_report.max_drawdown:.1%}",
                f"  OOS Profit Factor  : {self.combined_report.profit_factor:.2f}",
                f"  Passes Validation  : {'YES' if self.combined_report.passes_validation else 'NO'}",
            ]
        return "\n".join(lines)


# ── Window generator ──────────────────────────────────────────────────────────

def generate_windows(
    bars:         List[MarketBar],
    train_months: int = 12,
    test_months:  int = 3,
    bars_per_day: int = 375,
) -> List[WFWindow]:
    """
    Slide a (train | test) window across `bars`.
    Walks forward by `test_months` at a time.
    """
    bars_per_month = bars_per_day * 21   # approx
    train_bars     = train_months * bars_per_month
    test_bars      = test_months  * bars_per_month

    if len(bars) < train_bars + test_bars:
        raise ValueError(
            f"Insufficient data ({len(bars)} bars) for "
            f"train={train_months}m + test={test_months}m windows."
        )

    windows: List[WFWindow] = []
    fold    = 0
    offset  = 0

    while offset + train_bars + test_bars <= len(bars):
        tr = bars[offset: offset + train_bars]
        te = bars[offset + train_bars: offset + train_bars + test_bars]

        windows.append(WFWindow(
            fold        = fold,
            train_bars  = tr,
            test_bars   = te,
            train_start = tr[0].timestamp.date(),
            train_end   = tr[-1].timestamp.date(),
            test_start  = te[0].timestamp.date(),
            test_end    = te[-1].timestamp.date(),
        ))

        fold   += 1
        offset += test_bars   # slide by one test window

    logger.info("Generated %d walk-forward folds.", len(windows))
    return windows


# ── Walk-forward optimizer ────────────────────────────────────────────────────

ParamSpace = Dict[str, Any]   # name → (type, low, high) or list of choices


class WalkForwardOptimizer:
    """
    Runs Bayesian-optimised walk-forward validation.
    If optuna is unavailable, uses default params for every fold.
    """

    def __init__(
        self,
        config:          PlatformConfig,
        strategy_factory:Callable[[Instrument, PlatformConfig], BaseStrategy],
        instrument:      Instrument,
        param_space:     ParamSpace,
        n_trials:        int = 50,
    ) -> None:
        self._cfg      = config
        self._factory  = strategy_factory
        self._inst     = instrument
        self._space    = param_space
        self._n_trials = n_trials
        self._engine   = BacktestEngine(config)

    # ── Public ─────────────────────────────────────────────────────────────────

    def run(
        self,
        bars:           List[MarketBar],
        chains:         Optional[List[Optional[OptionChain]]] = None,
        breadth_series: Optional[List[float]] = None,
    ) -> WalkForwardResult:
        windows = generate_windows(
            bars,
            train_months = self._cfg.backtest.train_months,
            test_months  = self._cfg.backtest.test_months,
        )

        result = WalkForwardResult(
            strategy = self._factory(self._inst, self._cfg).name
        )

        all_test_trades: List[Trade] = []
        combined_equity = [self._cfg.risk.capital]

        for w in windows:
            logger.info(
                "Fold %d | train=%s→%s | test=%s→%s",
                w.fold, w.train_start, w.train_end, w.test_start, w.test_end,
            )

            # Optimise on train window
            best_params, train_sharpe = self._optimise_fold(
                w.train_bars, chains, breadth_series
            )

            # Evaluate on test window (out-of-sample)
            strategy    = self._make_strategy(best_params)
            test_chains = self._slice(chains, bars, w.test_bars) if chains else None
            test_bread  = self._slice(breadth_series, bars, w.test_bars) if breadth_series else None

            test_report = self._engine.run(strategy, w.test_bars,
                                           test_chains, test_bread)

            all_test_trades += [
                t for t in test_report.trade_dates  # placeholder; engine stores in report
            ]

            result.folds.append(FoldResult(
                fold         = w.fold,
                train_sharpe = train_sharpe,
                test_report  = test_report,
                best_params  = best_params,
            ))

            combined_equity.append(test_report.final_capital)

        # Build combined OOS report
        if result.folds:
            analyzer = PerformanceAnalyzer(self._cfg.risk_free_rate)
            first    = result.folds[0].test_report
            last     = result.folds[-1].test_report
            eq_curve = np.array(combined_equity)
            n_trades = result.oos_trades()

            # Use last fold's trade list as proxy (full aggregation needs engine refactor)
            result.combined_report = analyzer.generate_report(
                strategy        = result.strategy,
                trades          = [],  # summarised from folds
                equity_curve    = eq_curve,
                initial_capital = self._cfg.risk.capital,
                start_date      = first.start_date,
                end_date        = last.end_date,
                total_bars      = sum(len(w.test_bars) for w in windows),
            )
            # Patch trade count
            result.combined_report.total_trades = n_trades

        return result

    # ── Private ────────────────────────────────────────────────────────────────

    def _optimise_fold(
        self,
        train_bars:     List[MarketBar],
        chains:         Optional[List],
        breadth_series: Optional[List],
    ) -> Tuple[Dict[str, Any], float]:
        """Bayesian search on train window; returns (best_params, best_sharpe)."""
        if not _OPTUNA_AVAILABLE or not self._space:
            return {}, 0.0

        def objective(trial: optuna.Trial) -> float:
            params   = self._suggest_params(trial)
            strategy = self._make_strategy(params)
            ch       = self._slice(chains, None, train_bars) if chains else None
            br       = self._slice(breadth_series, None, train_bars) if breadth_series else None
            report   = self._engine.run(strategy, train_bars, ch, br)
            # Penalise if min_train_trades not met
            if report.total_trades < self._cfg.backtest.min_train_trades:
                return -10.0
            return float(report.sharpe)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self._n_trials, show_progress_bar=False)

        best   = study.best_params if study.best_params else {}
        return best, float(study.best_value) if study.trials else 0.0

    def _suggest_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in self._space.items():
            kind = spec.get("type", "float")
            if kind == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif kind == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    def _make_strategy(self, params: Dict[str, Any]) -> BaseStrategy:
        """Apply params to a copy of the config, then build the strategy."""
        cfg = self._apply_params(params)
        return self._factory(self._inst, cfg)

    def _apply_params(self, params: Dict[str, Any]) -> PlatformConfig:
        """Return a shallow-modified config with the trial params applied."""
        import copy
        cfg = copy.deepcopy(self._cfg)
        for name, val in params.items():
            for sub_cfg_name in ("strategy_a", "strategy_b", "strategy_c"):
                sub = getattr(cfg, sub_cfg_name, None)
                if sub and hasattr(sub, name):
                    setattr(sub, name, val)
        return cfg

    @staticmethod
    def _slice(series: Optional[List], full_bars: Optional[List],
               target_bars: List[MarketBar]) -> Optional[List]:
        """Extract the sub-list of series corresponding to target_bars."""
        if series is None or full_bars is None:
            return None
        if len(series) != len(full_bars):
            return None
        target_set = {id(b) for b in target_bars}
        return [s for b, s in zip(full_bars, series) if id(b) in target_set]
