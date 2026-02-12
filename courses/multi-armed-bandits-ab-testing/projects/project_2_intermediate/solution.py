"""
Commodity Allocation Engine - Complete Solution

This is the reference implementation with all TODOs completed.
Only look at this if you're stuck!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class CommodityDataLoader:
    """Load historical commodity data - COMPLETE."""

    TICKERS = {
        'WTI': 'CL=F',
        'Gold': 'GC=F',
        'Copper': 'HG=F',
        'NatGas': 'NG=F',
        'Corn': 'ZC=F'
    }

    def __init__(self, start_date='2023-01-01', end_date='2024-01-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.commodities = list(self.TICKERS.keys())

    def load_data(self):
        if YFINANCE_AVAILABLE:
            try:
                return self._load_real_data()
            except Exception as e:
                print(f"⚠️  yfinance error: {e}, using synthetic data")
                return self._generate_synthetic_data()
        else:
            return self._generate_synthetic_data()

    def _load_real_data(self):
        print("📊 Loading real commodity data from Yahoo Finance...")
        prices = {}
        for name, ticker in self.TICKERS.items():
            data = yf.download(ticker, start=self.start_date,
                             end=self.end_date, progress=False)
            prices[name] = data['Adj Close'] if len(data) > 0 else None

        df = pd.DataFrame(prices)
        for col in df.columns:
            if df[col].isna().all():
                df[col] = self._synthetic_price_series(len(df))

        weekly_prices = df.resample('W').last()
        weekly_returns = weekly_prices.pct_change().dropna()
        print(f"✅ Loaded {len(weekly_returns)} weeks of data")
        return weekly_returns

    def _generate_synthetic_data(self):
        print("🔧 Generating synthetic commodity data...")
        params = {
            'WTI':    {'mean': 0.002, 'std': 0.06},
            'Gold':   {'mean': 0.001, 'std': 0.03},
            'Copper': {'mean': 0.0015, 'std': 0.04},
            'NatGas': {'mean': 0.001, 'std': 0.08},
            'Corn':   {'mean': 0.0005, 'std': 0.05}
        }

        n_weeks = 52
        returns = {c: np.random.normal(p['mean'], p['std'], n_weeks)
                  for c, p in params.items()}

        df = pd.DataFrame(returns)
        df.index = pd.date_range(start=self.start_date, periods=n_weeks, freq='W')
        print(f"✅ Generated {len(df)} weeks of synthetic data")
        return df

    def _synthetic_price_series(self, n):
        returns = np.random.normal(0.001, 0.04, n)
        return 100 * np.exp(np.cumsum(returns))


class TwoWalletAllocator:
    """
    Two-wallet commodity allocator - COMPLETE SOLUTION.
    """

    def __init__(
        self,
        commodities,
        core_pct=0.8,
        bandit_pct=0.2,
        prior_mean=0.001,
        prior_std=0.02,
        min_allocation=0.05,
        max_allocation=0.50,
        max_tilt_speed=0.20
    ):
        self.commodities = commodities
        self.K = len(commodities)
        self.core_pct = core_pct
        self.bandit_pct = bandit_pct

        # Normal priors (returns are Gaussian, not Bernoulli!)
        self.means = np.full(self.K, prior_mean)
        self.stds = np.full(self.K, prior_std)
        self.n = np.zeros(self.K)

        # Guardrails
        self.min_alloc = min_allocation
        self.max_alloc = max_allocation
        self.max_tilt = max_tilt_speed

        self.prev_bandit_weights = np.ones(self.K) / self.K

    def get_core_weights(self):
        """Equal-weight core allocation."""
        return np.ones(self.K) / self.K

    def get_bandit_weights(self):
        """Thompson Sampling with guardrails - COMPLETE."""

        # Sample from Normal posteriors
        samples = np.array([norm.rvs(loc=self.means[i], scale=self.stds[i])
                           for i in range(self.K)])

        # Convert to weights via softmax
        exp_samples = np.exp(samples - samples.max())
        weights = exp_samples / exp_samples.sum()

        # Apply min/max allocation
        weights = np.clip(weights, self.min_alloc, self.max_alloc)
        weights = weights / weights.sum()

        # Apply tilt speed limit
        tilt = np.abs(weights - self.prev_bandit_weights)
        if np.any(tilt > self.max_tilt):
            # Blend with previous allocation
            alpha = 0.8
            weights = alpha * weights + (1 - alpha) * self.prev_bandit_weights
            weights = weights / weights.sum()

        self.prev_bandit_weights = weights.copy()
        return weights

    def get_total_weights(self):
        """Combine core and bandit."""
        w_core = self.get_core_weights()
        w_bandit = self.get_bandit_weights()
        return self.core_pct * w_core + self.bandit_pct * w_bandit

    def update(self, returns, reward_type='sharpe'):
        """Bayesian update - COMPLETE."""

        # Compute rewards
        if reward_type == 'raw':
            rewards = returns
        elif reward_type == 'sharpe':
            volatilities = np.maximum(self.stds, 0.01)
            rewards = returns / volatilities
        elif reward_type == 'regret':
            best_return = np.max(returns)
            rewards = returns - best_return
        else:
            rewards = returns

        # Bayesian update for Normal-Normal
        for i in range(self.K):
            self.n[i] += 1
            learning_rate = 1 / (self.n[i] + 1)
            self.means[i] = (1 - learning_rate) * self.means[i] + learning_rate * rewards[i]
            self.stds[i] = self.stds[i] / np.sqrt(1 + self.n[i] * 0.1)


def run_backtest(returns_df, initial_capital=100000, reward_type='sharpe'):
    """Run backtest - COMPLETE."""

    commodities = returns_df.columns.tolist()
    allocator = TwoWalletAllocator(commodities)

    portfolio_values = [initial_capital]
    allocation_history = []
    dates = returns_df.index

    print(f"\n=== COMMODITY ALLOCATION ENGINE BACKTEST ===\n")
    print(f"Period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} weeks)")
    print(f"Starting Capital: ${initial_capital:,.0f}")
    print(f"Reward Type: {reward_type}\n")

    current_value = initial_capital

    for i, date in enumerate(dates):
        weights = allocator.get_total_weights()
        week_returns = returns_df.iloc[i].values
        portfolio_return = np.dot(weights, week_returns)
        current_value *= (1 + portfolio_return)
        portfolio_values.append(current_value)
        allocator.update(week_returns, reward_type=reward_type)

        allocation_history.append({
            'date': date,
            'weights': weights.copy(),
            'returns': week_returns.copy(),
            'portfolio_return': portfolio_return,
            'value': current_value
        })

        if i % 10 == 0:
            print(f"Week {i+1:2d}: ${current_value:,.0f} "
                  f"({((current_value/initial_capital - 1) * 100):+.1f}%)")

    # Results
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print('='*60)

    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100

    print(f"\nFinal Portfolio Value: ${final_value:,.0f}")
    print(f"Total Return: {total_return:+.2f}%")

    portfolio_returns = [h['portfolio_return'] for h in allocation_history]
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(52)
    print(f"Sharpe Ratio: {sharpe:.2f}")

    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (np.array(portfolio_values) - peak) / peak
    max_dd = np.min(drawdown) * 100
    print(f"Max Drawdown: {max_dd:.2f}%")

    wins = sum(1 for r in portfolio_returns if r > 0)
    win_rate = wins / len(portfolio_returns) * 100
    print(f"Win Rate: {win_rate:.1f}% ({wins}/{len(portfolio_returns)} weeks)")

    print(f"\nFinal Allocation:")
    final_weights = allocation_history[-1]['weights']
    for commodity, weight in zip(commodities, final_weights):
        print(f"  {commodity:8s}: {weight*100:5.1f}%")

    return allocation_history, allocator


def compare_strategies(returns_df, initial_capital=100000):
    """
    Compare bandit vs benchmarks - BONUS.
    """
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60 + "\n")

    commodities = returns_df.columns.tolist()

    # 1. Two-wallet bandit
    allocator_bandit = TwoWalletAllocator(commodities)
    value_bandit = initial_capital
    for i in range(len(returns_df)):
        weights = allocator_bandit.get_total_weights()
        returns = returns_df.iloc[i].values
        portfolio_return = np.dot(weights, returns)
        value_bandit *= (1 + portfolio_return)
        allocator_bandit.update(returns, reward_type='sharpe')

    # 2. Equal weight (core only)
    weights_equal = np.ones(len(commodities)) / len(commodities)
    value_equal = initial_capital
    for i in range(len(returns_df)):
        returns = returns_df.iloc[i].values
        portfolio_return = np.dot(weights_equal, returns)
        value_equal *= (1 + portfolio_return)

    # 3. Best single commodity (oracle)
    total_returns = (1 + returns_df).prod() - 1
    best_commodity = total_returns.idxmax()
    value_best = initial_capital * (1 + total_returns[best_commodity])

    print("Final Portfolio Values:")
    print(f"  Two-Wallet Bandit:  ${value_bandit:,.0f} ({(value_bandit/initial_capital-1)*100:+.2f}%)")
    print(f"  Equal Weight:       ${value_equal:,.0f} ({(value_equal/initial_capital-1)*100:+.2f}%)")
    print(f"  Best Single ({best_commodity:6s}): ${value_best:,.0f} ({(value_best/initial_capital-1)*100:+.2f}%)")

    print(f"\nBandit Improvement:")
    print(f"  vs Equal Weight: {((value_bandit/value_equal - 1) * 100):+.1f}%")


def plot_backtest_results(allocation_history, commodities):
    """Visualize results - COMPLETE."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dates = [h['date'] for h in allocation_history]
    values = [h['value'] for h in allocation_history]

    # 1. Portfolio value
    ax = axes[0, 0]
    ax.plot(dates, values, linewidth=2, color='steelblue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Portfolio Value Over Time')
    ax.grid(True, alpha=0.3)
    ax.axhline(values[0], color='r', linestyle='--', alpha=0.5, label='Starting Value')
    ax.legend()

    # 2. Allocation evolution
    ax = axes[0, 1]
    weights_array = np.array([h['weights'] for h in allocation_history])
    for i, commodity in enumerate(commodities):
        ax.plot(dates, weights_array[:, i] * 100, label=commodity, linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Allocation (%)')
    ax.set_title('Allocation Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Weekly returns
    ax = axes[1, 0]
    returns = [h['portfolio_return'] * 100 for h in allocation_history]
    colors = ['g' if r > 0 else 'r' for r in returns]
    ax.bar(range(len(returns)), returns, color=colors, alpha=0.6)
    ax.set_xlabel('Week')
    ax.set_ylabel('Return (%)')
    ax.set_title('Weekly Returns')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Drawdown
    ax = axes[1, 1]
    peak = np.maximum.accumulate(values)
    drawdown = (np.array(values) - peak) / peak * 100
    ax.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
    ax.plot(dates, drawdown, color='darkred', linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('commodity_backtest_solution.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved visualization to 'commodity_backtest_solution.png'")
    plt.show()


if __name__ == "__main__":
    loader = CommodityDataLoader(
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    returns_df = loader.load_data()

    # Run backtest
    allocation_history, allocator = run_backtest(
        returns_df,
        initial_capital=100000,
        reward_type='sharpe'
    )

    # Compare strategies
    compare_strategies(returns_df)

    # Visualize
    plot_backtest_results(allocation_history, loader.commodities)

    print("\n🎉 Complete solution demonstrated!")
