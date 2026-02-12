"""
Commodity Trading Bandit Patterns - Domain-specific recipes
Each recipe solves ONE commodity trading problem in < 20 lines
"""

import numpy as np
import pandas as pd
from typing import Dict, List

# ============================================================================
# RECIPE 1: Weekly Commodity Allocation (18 lines)
# Problem: Allocate capital across commodities using bandit for sector tilt
# ============================================================================

def weekly_commodity_tilt(prices: pd.DataFrame, arms: Dict[str, Dict],
                          core_weight: float = 0.6, bandit_weight: float = 0.4) -> Dict[str, float]:
    """Core-satellite allocation with bandit tilt

    Args:
        prices: DataFrame with commodity prices (columns = tickers)
        arms: {"ticker": {"successes": int, "failures": int}, ...}
        core_weight: Allocation to equal-weighted core (e.g., 0.6 = 60%)
        bandit_weight: Allocation to bandit sleeve (e.g., 0.4 = 40%)
    Returns:
        {"ticker": final_allocation_pct, ...}
    """
    tickers = list(prices.columns)

    # Core: equal-weighted
    core_alloc = {t: core_weight / len(tickers) for t in tickers}

    # Bandit: Thompson Sampling for sleeve
    samples = {t: np.random.beta(arms[t]["successes"] + 1, arms[t]["failures"] + 1) for t in tickers}
    total_samples = sum(samples.values())
    bandit_alloc = {t: bandit_weight * (samples[t] / total_samples) for t in tickers}

    # Combine
    return {t: core_alloc[t] + bandit_alloc[t] for t in tickers}

# Example usage:
# prices = pd.DataFrame({"GLD": [100, 102], "SLV": [20, 21], "USO": [50, 49]})
# arms = {"GLD": {"successes": 15, "failures": 5}, "SLV": {"successes": 10, "failures": 10}, "USO": {"successes": 5, "failures": 15}}
# allocation = weekly_commodity_tilt(prices, arms, core_weight=0.6, bandit_weight=0.4)


# ============================================================================
# RECIPE 2: Regime Detection Feature (15 lines)
# Problem: Detect market regime (trending/mean-reverting) for contextual bandits
# ============================================================================

def compute_regime_features(prices: pd.Series, lookback: int = 20) -> Dict[str, float]:
    """Compute market regime features for contextual bandits

    Args:
        prices: Price series for a commodity
        lookback: Days to look back for regime calculation
    Returns:
        {"trend_strength": float, "volatility": float, "momentum": float}
    """
    returns = prices.pct_change().iloc[-lookback:]

    # Trend strength: R-squared of linear regression
    x = np.arange(len(returns))
    trend_strength = np.corrcoef(x, returns.fillna(0))[0, 1] ** 2

    # Volatility: annualized standard deviation
    volatility = returns.std() * np.sqrt(252)

    # Momentum: cumulative return over period
    momentum = (prices.iloc[-1] / prices.iloc[-lookback]) - 1

    return {"trend_strength": trend_strength, "volatility": volatility, "momentum": momentum}

# Example usage:
# prices = pd.Series([100, 102, 101, 105, 107, 106, 110, 112, 111, 115])
# regime = compute_regime_features(prices, lookback=5)


# ============================================================================
# RECIPE 3: Risk-Adjusted Reward (14 lines)
# Problem: Calculate reward that balances return with drawdown risk
# ============================================================================

def risk_adjusted_reward(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Sharpe-like reward with drawdown penalty

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annualized risk-free rate (e.g., 0.04 = 4%)
    Returns:
        Risk-adjusted reward score (normalized to ~[0, 1])
    """
    excess_return = returns.mean() - (risk_free_rate / 252)
    volatility = returns.std()

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # Combined score: Sharpe ratio penalized by drawdown
    sharpe = excess_return / (volatility + 1e-8)
    reward = sharpe * (1 - max_drawdown)

    # Normalize to [0, 1] using sigmoid
    return 1 / (1 + np.exp(-reward * 2))

# Example usage:
# returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.04])
# reward = risk_adjusted_reward(returns, risk_free_rate=0.04)


# ============================================================================
# RECIPE 4: Seasonal Bandit Adjustment (16 lines)
# Problem: Weight recent months more for seasonal commodities (e.g., agriculture)
# ============================================================================

def seasonal_reward_weighting(returns: pd.Series, current_month: int,
                             seasonal_window: int = 3) -> float:
    """Weight returns by seasonal similarity

    Args:
        returns: Series with DatetimeIndex
        current_month: Current month (1-12)
        seasonal_window: Months around current to emphasize (e.g., 3 = ±3 months)
    Returns:
        Seasonally-weighted mean return
    """
    weights = []
    for date in returns.index:
        month_diff = abs((date.month - current_month + 6) % 12 - 6)  # Circular distance
        weight = np.exp(-month_diff / seasonal_window)  # Exponential decay
        weights.append(weight)

    weights = np.array(weights)
    weights /= weights.sum()  # Normalize

    return (returns * weights).sum()

# Example usage:
# dates = pd.date_range("2023-01-01", periods=100, freq="D")
# returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
# weighted_return = seasonal_reward_weighting(returns, current_month=6, seasonal_window=2)


# ============================================================================
# RECIPE 5: Correlation Guardrail (19 lines)
# Problem: Prevent concentration in highly correlated commodities
# ============================================================================

def correlation_guardrail(allocation: Dict[str, float], prices: pd.DataFrame,
                         max_correlated_allocation: float = 0.6) -> Dict[str, float]:
    """Limit allocation to correlated commodity groups

    Args:
        allocation: Proposed allocation {"ticker": weight, ...}
        prices: DataFrame of commodity prices
        max_correlated_allocation: Max total allocation to correlated group
    Returns:
        Adjusted allocation satisfying correlation constraint
    """
    # Compute correlation matrix
    returns = prices.pct_change().dropna()
    corr_matrix = returns.corr()

    # Find correlated groups (correlation > 0.7)
    for ticker in allocation:
        correlated = [t for t in allocation if corr_matrix.loc[ticker, t] > 0.7 and t != ticker]
        group_allocation = allocation[ticker] + sum(allocation[t] for t in correlated)

        # Scale down if over limit
        if group_allocation > max_correlated_allocation:
            scale = max_correlated_allocation / group_allocation
            allocation[ticker] *= scale
            for t in correlated:
                allocation[t] *= scale

    # Renormalize to sum to 1
    total = sum(allocation.values())
    return {k: v / total for k, v in allocation.items()}

# Example usage:
# allocation = {"GLD": 0.4, "SLV": 0.3, "USO": 0.3}  # GLD and SLV highly correlated
# prices = pd.DataFrame({"GLD": [100, 102, 101], "SLV": [20, 20.5, 20.2], "USO": [50, 49, 51]})
# adjusted = correlation_guardrail(allocation, prices, max_correlated_allocation=0.6)


# ============================================================================
# RECIPE 6: Position Sizing with Kelly Criterion (12 lines)
# Problem: Size positions based on win rate and payoff ratio
# ============================================================================

def kelly_position_size(win_rate: float, avg_win: float, avg_loss: float,
                       max_allocation: float = 0.25) -> float:
    """Calculate position size using Kelly Criterion

    Args:
        win_rate: Probability of winning trade (0-1)
        avg_win: Average win amount
        avg_loss: Average loss amount (positive number)
        max_allocation: Cap on Kelly allocation
    Returns:
        Optimal allocation (0-1)
    """
    if avg_loss == 0:
        return max_allocation

    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    return min(max(kelly_fraction, 0), max_allocation)  # Clamp to [0, max_allocation]

# Example usage:
# position_size = kelly_position_size(win_rate=0.6, avg_win=1000, avg_loss=500, max_allocation=0.25)


# ============================================================================
# RECIPE 7: Volatility-Scaled Allocation (10 lines)
# Problem: Adjust allocation inversely to volatility (risk parity)
# ============================================================================

def volatility_scaled_allocation(prices: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """Risk parity allocation (inverse volatility weighting)

    Args:
        prices: DataFrame of commodity prices
        lookback: Days to calculate volatility
    Returns:
        {"ticker": allocation, ...}
    """
    returns = prices.pct_change().iloc[-lookback:]
    volatilities = returns.std() * np.sqrt(252)  # Annualized
    inv_vol = 1 / volatilities
    total_inv_vol = inv_vol.sum()
    return {ticker: inv_vol[ticker] / total_inv_vol for ticker in prices.columns}

# Example usage:
# prices = pd.DataFrame({"GLD": [100, 102, 101, 103], "USO": [50, 55, 52, 58]})  # USO more volatile
# allocation = volatility_scaled_allocation(prices, lookback=3)
