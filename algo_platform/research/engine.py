"""
Research engine: computes IC, MI, t-stat, Sharpe contribution, decile analysis,
and feature importance for each of the 13 features.
Only features that pass all significance gates are promoted to live strategies.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("platform.research.engine")

_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed — MI and feature importance disabled")

from algo_platform.core.types import FeatureStats, FeatureVector


# ── Low-level statistical primitives ─────────────────────────────────────────

def spearman_ic(feature: np.ndarray, fwd_return: np.ndarray) -> float:
    """Rank-correlation Information Coefficient (single period)."""
    if len(feature) < 10:
        return 0.0
    mask = np.isfinite(feature) & np.isfinite(fwd_return)
    if mask.sum() < 10:
        return 0.0
    rho, _ = stats.spearmanr(feature[mask], fwd_return[mask])
    return float(rho) if np.isfinite(rho) else 0.0


def rolling_ic(features: np.ndarray, fwd_returns: np.ndarray,
               window: int = 60) -> np.ndarray:
    """Compute IC for each rolling window of length `window`."""
    n = len(features)
    if n < window:
        return np.array([spearman_ic(features, fwd_returns)])
    ics = []
    for i in range(window, n + 1):
        ic = spearman_ic(features[i - window: i], fwd_returns[i - window: i])
        ics.append(ic)
    return np.array(ics)


def ic_t_stat(ic_series: np.ndarray) -> float:
    """t-statistic of the IC time series (IC_IR × √n)."""
    n = len(ic_series)
    if n < 3:
        return 0.0
    mu  = float(np.mean(ic_series))
    std = float(np.std(ic_series, ddof=1))
    if std < 1e-10:
        return 0.0
    return float(mu / std * np.sqrt(n))


def mutual_information(feature: np.ndarray, fwd_return: np.ndarray) -> float:
    """Mutual information between feature and forward return."""
    if not _SKLEARN_AVAILABLE:
        return 0.0
    mask = np.isfinite(feature) & np.isfinite(fwd_return)
    if mask.sum() < 30:
        return 0.0
    X = feature[mask].reshape(-1, 1)
    y = fwd_return[mask]
    mi = mutual_info_regression(X, y, random_state=42)
    return float(mi[0])


def sharpe_contribution(feature: np.ndarray, fwd_return: np.ndarray,
                         risk_free: float = 0.0) -> float:
    """
    Sharpe ratio of a long/short portfolio formed on feature sign.
    Goes long when feature > median, short when < median.
    """
    if len(feature) < 20:
        return 0.0
    mask = np.isfinite(feature) & np.isfinite(fwd_return)
    f, r = feature[mask], fwd_return[mask]
    if len(f) < 20:
        return 0.0
    median = float(np.median(f))
    position = np.where(f > median, 1.0, -1.0)
    pnl = position * r
    std_pnl = float(np.std(pnl, ddof=1))
    if std_pnl < 1e-10:
        return 0.0
    return float((np.mean(pnl) - risk_free) / std_pnl * np.sqrt(252))


def decile_analysis(feature: np.ndarray, fwd_return: np.ndarray) -> np.ndarray:
    """
    Returns 10-element array: mean forward return for each feature decile.
    Decile 1 = lowest feature values; Decile 10 = highest.
    """
    mask = np.isfinite(feature) & np.isfinite(fwd_return)
    f, r = feature[mask], fwd_return[mask]
    if len(f) < 30:
        return np.zeros(10)
    bins = np.percentile(f, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    if len(bins) < 3:
        return np.zeros(10)
    labels = np.digitize(f, bins[1:-1])
    result = np.zeros(10)
    for d in range(10):
        idx = labels == d
        if idx.sum() > 0:
            result[d] = float(np.mean(r[idx]))
    return result


def feature_importances(features_df: pd.DataFrame,
                          fwd_return: np.ndarray) -> pd.Series:
    """Gradient-boosting feature importances (normalised to sum=1)."""
    if not _SKLEARN_AVAILABLE:
        return pd.Series(
            np.ones(len(features_df.columns)) / len(features_df.columns),
            index=features_df.columns,
        )
    mask = np.all(np.isfinite(features_df.values), axis=1) & np.isfinite(fwd_return)
    X, y = features_df.values[mask], fwd_return[mask]
    if len(X) < 50:
        return pd.Series(np.zeros(features_df.shape[1]), index=features_df.columns)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    gb = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    gb.fit(X, y)
    imp = gb.feature_importances_
    return pd.Series(imp / imp.sum() if imp.sum() > 0 else imp,
                     index=features_df.columns)


# ── Research engine ───────────────────────────────────────────────────────────

class ResearchEngine:
    """
    Runs the full research pipeline on a panel of features + forward returns.
    Input: DataFrame with columns = feature names + 'forward_return'.
    Output: dict of FeatureStats keyed by feature name.
    """

    FEATURE_NAMES = FeatureVector.FEATURE_NAMES

    def __init__(
        self,
        min_ic_abs:    float = 0.03,
        min_t_stat:    float = 2.0,
        max_pvalue:    float = 0.05,
        min_mi:        float = 0.01,
        ic_window:     int   = 60,
    ) -> None:
        self.min_ic_abs = min_ic_abs
        self.min_t_stat = min_t_stat
        self.max_pvalue = max_pvalue
        self.min_mi     = min_mi
        self.ic_window  = ic_window

    def analyze_feature(
        self,
        name:        str,
        feature:     np.ndarray,
        fwd_return:  np.ndarray,
        all_features_df: Optional[pd.DataFrame] = None,
    ) -> FeatureStats:
        """Full statistical analysis of a single feature."""
        ic_series = rolling_ic(feature, fwd_return, self.ic_window)
        mean_ic   = float(np.mean(ic_series))
        std_ic    = float(np.std(ic_series, ddof=1)) if len(ic_series) > 1 else 1.0
        t_stat    = ic_t_stat(ic_series)
        n         = len(ic_series)

        # p-value from t-distribution
        if n > 2 and abs(t_stat) > 0:
            p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        else:
            p_value = 1.0

        mi   = mutual_information(feature, fwd_return)
        sc   = sharpe_contribution(feature, fwd_return)
        dec  = decile_analysis(feature, fwd_return)

        # Feature importance: computed on the full panel if available
        if all_features_df is not None and name in all_features_df.columns:
            imp_series = feature_importances(all_features_df, fwd_return)
            importance = float(imp_series.get(name, 0.0))
        else:
            importance = 0.0

        is_sig = (
            abs(mean_ic) >= self.min_ic_abs
            and abs(t_stat) >= self.min_t_stat
            and p_value    <= self.max_pvalue
            and mi         >= self.min_mi
        )

        return FeatureStats(
            name                = name,
            ic                  = mean_ic,
            ic_std              = std_ic,
            ic_pvalue           = p_value,
            mutual_info         = mi,
            t_stat              = t_stat,
            sharpe_contribution = sc,
            decile_returns      = dec,
            feature_importance  = importance,
            is_significant      = is_sig,
            n_observations      = n,
        )

    def run_full_research(self, data: pd.DataFrame) -> Dict[str, FeatureStats]:
        """
        Parameters
        ----------
        data : DataFrame with columns matching FEATURE_NAMES + 'forward_return'.

        Returns
        -------
        dict mapping feature name → FeatureStats (all features, not just significant).
        """
        if "forward_return" not in data.columns:
            raise ValueError("data must contain a 'forward_return' column")

        fwd = data["forward_return"].to_numpy(dtype=float)
        available = [c for c in self.FEATURE_NAMES if c in data.columns]
        features_df = data[available].copy() if available else pd.DataFrame()

        results: Dict[str, FeatureStats] = {}
        for name in available:
            feat = data[name].to_numpy(dtype=float)
            stats_ = self.analyze_feature(
                name, feat, fwd,
                all_features_df=features_df if _SKLEARN_AVAILABLE else None,
            )
            results[name] = stats_
            logger.info(
                "Feature %-20s | IC=%+.3f t=%.2f p=%.3f MI=%.3f Sharpe=%.2f %s",
                name, stats_.ic, stats_.t_stat, stats_.ic_pvalue,
                stats_.mutual_info, stats_.sharpe_contribution,
                "✓ SIGNIFICANT" if stats_.is_significant else "✗",
            )

        return results

    def select_features(self, stats: Dict[str, FeatureStats]) -> List[str]:
        """Return feature names that pass statistical significance."""
        sig = [n for n, s in stats.items() if s.is_significant]
        logger.info("Significant features (%d/%d): %s",
                    len(sig), len(stats), sig)
        return sig

    def print_report(self, stats: Dict[str, FeatureStats]) -> None:
        """Pretty-print the research report to stdout."""
        header = (
            f"{'Feature':<22} {'IC':>7} {'t-stat':>8} {'p-val':>7} "
            f"{'MI':>7} {'Sharpe':>8} {'Sig':>5}"
        )
        print("\n" + "=" * len(header))
        print("FEATURE RESEARCH REPORT")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for name, s in sorted(stats.items(), key=lambda x: -abs(x[1].ic)):
            print(
                f"{name:<22} {s.ic:>+7.3f} {s.t_stat:>8.2f} {s.ic_pvalue:>7.4f} "
                f"{s.mutual_info:>7.4f} {s.sharpe_contribution:>8.2f} "
                f"{'YES' if s.is_significant else 'no':>5}"
            )

        n_sig = sum(1 for s in stats.values() if s.is_significant)
        print("=" * len(header))
        print(f"  {n_sig}/{len(stats)} features pass significance gates.")
        print("=" * len(header) + "\n")
