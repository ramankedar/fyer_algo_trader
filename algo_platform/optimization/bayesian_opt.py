"""
Bayesian Optimization wrapper using Optuna (TPE sampler).

Usage
-----
optimizer = BayesianOptimizer(objective_fn, param_space, n_trials=100)
result    = optimizer.run()
print(result.best_params, result.best_value)

`objective_fn` receives a dict of suggested params and must return a scalar
(higher = better; Sharpe ratio is the natural choice).

Never brute-force: optuna's TPE models the posterior and focuses trials on
promising regions rather than exhaustive grid search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("platform.optimization.bayesian")

_OPTUNA_AVAILABLE = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("optuna not installed — BayesianOptimizer will be a no-op")


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    best_params:  Dict[str, Any]
    best_value:   float
    n_trials:     int
    n_complete:   int
    trial_values: List[float] = field(default_factory=list)
    trial_params: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Bayesian Optimisation Result",
            f"  Trials     : {self.n_complete}/{self.n_trials}",
            f"  Best value : {self.best_value:.4f}",
            f"  Best params: {self.best_params}",
        ]
        return "\n".join(lines)


# ── Parameter space helpers ───────────────────────────────────────────────────

def float_param(low: float, high: float, log: bool = False) -> Dict[str, Any]:
    return {"type": "float", "low": low, "high": high, "log": log}

def int_param(low: int, high: int) -> Dict[str, Any]:
    return {"type": "int", "low": low, "high": high}

def categorical_param(choices: list) -> Dict[str, Any]:
    return {"type": "categorical", "choices": choices}


# ── Predefined parameter spaces for each strategy ────────────────────────────

STRATEGY_A_SPACE: Dict[str, Dict] = {
    "atr_pct_threshold":     float_param(0.10, 0.30),
    "rv_pct_threshold":      float_param(0.10, 0.30),
    "entropy_pct_threshold": float_param(0.10, 0.30),
    "range_pct_threshold":   float_param(0.10, 0.30),
    "volume_spike_mult":     float_param(1.5, 3.5),
    "trail_atr_mult":        float_param(1.0, 3.0),
    "vol_stop_mult":         float_param(1.5, 4.0),
}

STRATEGY_B_SPACE: Dict[str, Dict] = {
    "adx_threshold":         float_param(18.0, 35.0),
    "breadth_threshold":     float_param(0.40, 0.65),
    "adx_period":            int_param(10, 20),
}

STRATEGY_C_SPACE: Dict[str, Dict] = {
    "gex_concentration_threshold": float_param(0.25, 0.60),
}


# ── Optimizer ─────────────────────────────────────────────────────────────────

class BayesianOptimizer:
    """
    Wraps optuna to provide a clean interface for strategy parameter search.

    Parameters
    ----------
    objective   : function(params: dict) -> float (higher is better)
    param_space : dict of param_name → spec (from float_param / int_param / etc.)
    n_trials    : number of TPE trials
    seed        : for reproducibility
    """

    def __init__(
        self,
        objective:   Callable[[Dict[str, Any]], float],
        param_space: Dict[str, Dict[str, Any]],
        n_trials:    int = 100,
        seed:        int = 42,
        n_jobs:      int = 1,
    ) -> None:
        self._objective  = objective
        self._space      = param_space
        self._n_trials   = n_trials
        self._seed       = seed
        self._n_jobs     = n_jobs

    def run(self) -> OptimizationResult:
        if not _OPTUNA_AVAILABLE:
            logger.error("optuna not installed — returning empty result")
            return OptimizationResult({}, 0.0, self._n_trials, 0)

        sampler = optuna.samplers.TPESampler(seed=self._seed)
        study   = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            self._wrapped_objective,
            n_trials  = self._n_trials,
            n_jobs    = self._n_jobs,
            show_progress_bar=False,
        )

        trial_values = [t.value for t in study.trials if t.value is not None]
        trial_params = [t.params for t in study.trials]

        return OptimizationResult(
            best_params  = study.best_params,
            best_value   = float(study.best_value),
            n_trials     = self._n_trials,
            n_complete   = len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.COMPLETE]),
            trial_values = trial_values,
            trial_params = trial_params,
        )

    def _wrapped_objective(self, trial: "optuna.Trial") -> float:
        params = self._suggest(trial)
        try:
            value = self._objective(params)
            return float(value) if value is not None else -999.0
        except Exception as exc:
            logger.debug("Trial failed: %s", exc)
            return -999.0

    def _suggest(self, trial: "optuna.Trial") -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in self._space.items():
            kind = spec.get("type", "float")
            if kind == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif kind == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params
