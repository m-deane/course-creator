"""
Production Regime-Aware Trading Allocator - Complete Solution

Reference implementation with all components.
Study this to understand production bandit systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import json

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class FeaturePipeline:
    """Feature extraction - COMPLETE."""

    def __init__(self, lookback_window=20):
        self.lookback = lookback_window

    def extract_features(self, prices_df, current_idx):
        if current_idx < self.lookback:
            return self._default_features()

        recent_prices = prices_df.iloc[max(0, current_idx-self.lookback):current_idx]
        features = {}

        returns = recent_prices.pct_change().dropna()
        features['realized_vol'] = returns.std().mean()
        features['vol_percentile'] = self._percentile(
            features['realized_vol'],
            returns.std().rolling(52).mean().dropna()
        )

        if len(recent_prices) >= 20:
            ma_20 = recent_prices.iloc[-20:].mean()
            ma_50 = recent_prices.iloc[-50:].mean() if len(recent_prices) >= 50 else ma_20
            features['momentum'] = (recent_prices.iloc[-1] / recent_prices.iloc[-20] - 1).mean()
            features['ma_cross'] = 1 if (ma_20 > ma_50).mean() > 0.5 else 0
        else:
            features['momentum'] = 0
            features['ma_cross'] = 0

        current_date = prices_df.index[current_idx]
        features['month'] = current_date.month
        features['quarter'] = (current_date.month - 1) // 3 + 1
        features['contango_indicator'] = 1 if features['momentum'] > 0 else 0

        return features

    def _percentile(self, value, historical_values):
        if len(historical_values) == 0:
            return 0.5
        return (historical_values < value).mean()

    def _default_features(self):
        return {
            'realized_vol': 0.02,
            'vol_percentile': 0.5,
            'momentum': 0.0,
            'ma_cross': 0,
            'month': 1,
            'quarter': 1,
            'contango_indicator': 0
        }

    def feature_vector(self, features):
        return np.array([
            features['realized_vol'],
            features['vol_percentile'],
            features['momentum'],
            features['ma_cross'],
            features['month'] / 12.0,
            features['quarter'] / 4.0,
            features['contango_indicator']
        ])


class ContextualBanditAllocator:
    """LinUCB allocator - COMPLETE SOLUTION."""

    def __init__(self, commodities, feature_dim=7, alpha=1.0, lambda_reg=1.0):
        self.commodities = commodities
        self.K = len(commodities)
        self.d = feature_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        self.A = [np.eye(self.d) * lambda_reg for _ in range(self.K)]
        self.b = [np.zeros(self.d) for _ in range(self.K)]

    def select_arms(self, context):
        """LinUCB with Thompson Sampling - COMPLETE."""
        predicted_rewards = np.zeros(self.K)

        for i in range(self.K):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]

            # Thompson Sampling: sample from posterior
            cov = self.alpha * A_inv
            theta_sample = np.random.multivariate_normal(theta, cov)

            predicted_rewards[i] = context @ theta_sample

        # Softmax to weights
        exp_rewards = np.exp(predicted_rewards - predicted_rewards.max())
        weights = exp_rewards / exp_rewards.sum()

        return weights

    def update(self, context, returns):
        """Bayesian update - COMPLETE."""
        for i in range(self.K):
            self.A[i] += np.outer(context, context)
            self.b[i] += returns[i] * context


class GuardrailSystem:
    """Production guardrails - COMPLETE SOLUTION."""

    def __init__(
        self,
        max_position=0.40,
        min_position=0.05,
        max_drawdown=0.15,
        vol_scale_threshold=30,
        correlation_threshold=0.8
    ):
        self.max_position = max_position
        self.min_position = min_position
        self.max_drawdown = max_drawdown
        self.vol_scale_threshold = vol_scale_threshold
        self.correlation_threshold = correlation_threshold
        self.peak_value = None
        self.circuit_breaker_active = False

    def apply_guardrails(self, weights, context, portfolio_state):
        """Apply all guardrails - COMPLETE."""
        alerts = []
        adjusted = weights.copy()

        # 1. Position limits
        if np.any(adjusted > self.max_position):
            alerts.append(f"Position limit: max {adjusted.max():.1%}")
            adjusted = np.clip(adjusted, self.min_position, self.max_position)
            adjusted = adjusted / adjusted.sum()

        if np.any(adjusted < self.min_position):
            alerts.append(f"Min allocation: min {adjusted.min():.1%}")
            adjusted = np.clip(adjusted, self.min_position, self.max_position)
            adjusted = adjusted / adjusted.sum()

        # 2. Volatility scaling
        if context.get('vol_percentile', 0.5) > 0.8:
            alerts.append("High volatility: reducing exposure")
            # Scale down aggressive positions
            adjusted = 0.7 * adjusted + 0.3 * np.ones(len(adjusted)) / len(adjusted)

        # 3. Correlation monitoring
        if portfolio_state.get('correlation', 0) > self.correlation_threshold:
            alerts.append(f"High correlation: {portfolio_state['correlation']:.2f}")
            # Force more diversification
            adjusted = 0.5 * adjusted + 0.5 * np.ones(len(adjusted)) / len(adjusted)

        # 4. Drawdown circuit breaker
        current_value = portfolio_state.get('current_value', 100000)
        if self.peak_value is None:
            self.peak_value = current_value
        else:
            self.peak_value = max(self.peak_value, current_value)

        drawdown = (current_value - self.peak_value) / self.peak_value

        if drawdown < -self.max_drawdown:
            if not self.circuit_breaker_active:
                alerts.append(f"⚠️  CIRCUIT BREAKER: {drawdown:.1%} drawdown")
                self.circuit_breaker_active = True
            adjusted = np.ones(len(weights)) / len(weights)
        elif drawdown > -0.10 and self.circuit_breaker_active:
            alerts.append("✅ Circuit breaker deactivated")
            self.circuit_breaker_active = False

        return adjusted, alerts


class MonitoringSystem:
    """Monitoring and logging - COMPLETE SOLUTION."""

    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        self.decisions = []

    def log_decision(self, timestamp, context, allocation, predictions, alerts):
        """Structured logging - COMPLETE."""
        decision = {
            'timestamp': str(timestamp),
            'context': context,
            'allocation': allocation.tolist(),
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'alerts': alerts
        }
        self.decisions.append(decision)

    def check_anomalies(self, allocation, historical_allocations):
        """Anomaly detection - COMPLETE."""
        alerts = []

        if len(historical_allocations) > 5:
            recent_avg = np.mean(historical_allocations[-5:], axis=0)
            diff = np.abs(allocation - recent_avg)
            if np.max(diff) > 0.15:
                alerts.append(f"Large shift: {np.max(diff):.1%}")

        # Concentration check
        if np.max(allocation) > 0.35:
            alerts.append(f"High concentration: {np.max(allocation):.1%}")

        return alerts


def run_production_backtest(prices_df, initial_capital=100000):
    """Full production backtest - COMPLETE."""

    commodities = prices_df.columns.tolist()

    feature_pipeline = FeaturePipeline()
    allocator = ContextualBanditAllocator(commodities, feature_dim=7)
    guardrails = GuardrailSystem()
    monitoring = MonitoringSystem()

    portfolio_values = [initial_capital]
    allocation_history = []
    current_value = initial_capital

    print("=== PRODUCTION COMMODITY ALLOCATOR BACKTEST ===\n")
    print(f"Period: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
    print(f"Starting Capital: ${initial_capital:,.0f}\n")

    returns_df = prices_df.pct_change().dropna()

    for i in range(20, len(returns_df)):
        date = returns_df.index[i]

        features = feature_pipeline.extract_features(prices_df, i)
        context = feature_pipeline.feature_vector(features)

        raw_weights = allocator.select_arms(context)

        portfolio_state = {
            'current_value': current_value,
            'correlation': np.corrcoef(returns_df.iloc[max(0,i-20):i].T).mean()
        }

        final_weights, alerts = guardrails.apply_guardrails(
            raw_weights, features, portfolio_state
        )

        week_returns = returns_df.iloc[i].values
        portfolio_return = np.dot(final_weights, week_returns)
        current_value *= (1 + portfolio_return)
        portfolio_values.append(current_value)

        allocator.update(context, week_returns)

        anomaly_alerts = monitoring.check_anomalies(
            final_weights,
            [h['weights'] for h in allocation_history[-10:]]
        )

        monitoring.log_decision(
            date, features, final_weights,
            raw_weights, alerts + anomaly_alerts
        )

        allocation_history.append({
            'date': date,
            'weights': final_weights,
            'returns': week_returns,
            'portfolio_return': portfolio_return,
            'value': current_value,
            'features': features,
            'alerts': alerts + anomaly_alerts
        })

        if i % 10 == 0:
            print(f"Week {i}: ${current_value:,.0f} ({((current_value/initial_capital-1)*100):+.1f}%)")

    print(f"\n{'='*60}")
    print("PRODUCTION BACKTEST RESULTS")
    print('='*60)

    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100

    print(f"\nFinal Value: ${final_value:,.0f}")
    print(f"Total Return: {total_return:+.2f}%")

    portfolio_returns = [h['portfolio_return'] for h in allocation_history]
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(52)
    print(f"Sharpe Ratio: {sharpe:.2f}")

    all_alerts = [a for h in allocation_history for a in h['alerts']]
    print(f"\nTotal Alerts: {len(all_alerts)}")

    return allocation_history, allocator, monitoring


def generate_weekly_report(allocation_history, commodities):
    """Executive report - COMPLETE."""

    print("\n" + "="*60)
    print("WEEKLY ALLOCATION REPORT")
    print("="*60)

    latest = allocation_history[-1]
    print(f"\nDate: {latest['date'].date()}")

    print(f"\nCurrent Allocation:")
    for i, commodity in enumerate(commodities):
        print(f"  {commodity:8s}: {latest['weights'][i]*100:5.1f}%")

    print(f"\nRegime Features:")
    for key, value in latest['features'].items():
        print(f"  {key:20s}: {value}")

    print(f"\nRecent Alerts ({len(latest['alerts'])}):")
    for alert in latest['alerts']:
        print(f"  - {alert}")

    recent_returns = [h['portfolio_return'] for h in allocation_history[-4:]]
    print(f"\n4-Week Performance: {sum(recent_returns)*100:+.2f}%")


if __name__ == "__main__":
    print("Generating commodity data...")
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='W')
    commodities = ['WTI', 'Gold', 'Copper', 'NatGas', 'Corn']

    prices = {}
    for commodity in commodities:
        drift = 0.0005
        vol = np.random.uniform(0.03, 0.08)
        returns = np.random.normal(drift, vol, len(dates))
        prices[commodity] = 100 * np.exp(np.cumsum(returns))

    prices_df = pd.DataFrame(prices, index=dates)

    allocation_history, allocator, monitoring = run_production_backtest(prices_df)
    generate_weekly_report(allocation_history, commodities)

    print("\n✅ Complete production system demonstrated!")
