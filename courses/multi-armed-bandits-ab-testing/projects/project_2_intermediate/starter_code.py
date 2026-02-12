"""
Commodity Allocation Engine - Starter Code

Your task: Complete the TwoWalletAllocator class to build an adaptive
commodity portfolio using Thompson Sampling with safety guardrails.

TODOs are marked clearly. Data loading and backtesting framework are complete.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

# Try to import yfinance, fall back to synthetic data if unavailable
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not available, using synthetic data")


class CommodityDataLoader:
    """
    Load historical commodity data.

    This is COMPLETE - you don't need to modify it.
    """

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
        """Load commodity data, with synthetic fallback."""
        if YFINANCE_AVAILABLE:
            try:
                return self._load_real_data()
            except Exception as e:
                print(f"⚠️  yfinance error: {e}")
                print("   Falling back to synthetic data")
                return self._generate_synthetic_data()
        else:
            return self._generate_synthetic_data()

    def _load_real_data(self):
        """Load from Yahoo Finance."""
        print("📊 Loading real commodity data from Yahoo Finance...")

        prices = {}
        for name, ticker in self.TICKERS.items():
            data = yf.download(ticker, start=self.start_date,
                             end=self.end_date, progress=False)
            if len(data) > 0:
                prices[name] = data['Adj Close']
            else:
                print(f"   ⚠️  No data for {name}, using synthetic")
                prices[name] = None

        # Convert to DataFrame
        df = pd.DataFrame(prices)

        # Fill missing with synthetic
        for col in df.columns:
            if df[col].isna().all():
                df[col] = self._synthetic_price_series(len(df))

        # Resample to weekly and calculate returns
        weekly_prices = df.resample('W').last()
        weekly_returns = weekly_prices.pct_change().dropna()

        print(f"✅ Loaded {len(weekly_returns)} weeks of data")
        return weekly_returns

    def _generate_synthetic_data(self):
        """Generate realistic synthetic commodity returns."""
        print("🔧 Generating synthetic commodity data...")

        # Realistic parameters based on commodity characteristics
        params = {
            'WTI':    {'mean': 0.002, 'std': 0.06},  # High volatility
            'Gold':   {'mean': 0.001, 'std': 0.03},  # Low volatility, safe haven
            'Copper': {'mean': 0.0015, 'std': 0.04}, # Industrial, moderate vol
            'NatGas': {'mean': 0.001, 'std': 0.08},  # Highest volatility
            'Corn':   {'mean': 0.0005, 'std': 0.05}  # Seasonal, moderate vol
        }

        # Generate 52 weeks of returns
        n_weeks = 52
        returns = {}

        for commodity, p in params.items():
            returns[commodity] = np.random.normal(
                p['mean'], p['std'], n_weeks
            )

        df = pd.DataFrame(returns)
        df.index = pd.date_range(start=self.start_date, periods=n_weeks, freq='W')

        print(f"✅ Generated {len(df)} weeks of synthetic data")
        return df

    def _synthetic_price_series(self, n):
        """Generate a single synthetic price series."""
        returns = np.random.normal(0.001, 0.04, n)
        prices = 100 * np.exp(np.cumsum(returns))
        return prices


class TwoWalletAllocator:
    """
    Two-wallet commodity allocator with Thompson Sampling.

    YOUR TASK: Complete the methods marked with TODO.
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
        """
        Initialize two-wallet allocator.

        Args:
            commodities: List of commodity names
            core_pct: Core wallet percentage (e.g., 0.8 = 80%)
            bandit_pct: Bandit sleeve percentage (e.g., 0.2 = 20%)
            prior_mean: Prior mean return
            prior_std: Prior standard deviation
            min_allocation: Minimum weight per commodity (guardrail)
            max_allocation: Maximum weight per commodity (guardrail)
            max_tilt_speed: Maximum weekly allocation change (guardrail)
        """
        self.commodities = commodities
        self.K = len(commodities)
        self.core_pct = core_pct
        self.bandit_pct = bandit_pct

        # Thompson Sampling with Normal priors (not Beta - returns are Gaussian!)
        self.means = np.full(self.K, prior_mean)
        self.stds = np.full(self.K, prior_std)
        self.n = np.zeros(self.K)

        # Guardrails
        self.min_alloc = min_allocation
        self.max_alloc = max_allocation
        self.max_tilt = max_tilt_speed

        # Track previous allocation for tilt speed limit
        self.prev_bandit_weights = np.ones(self.K) / self.K

    def get_core_weights(self):
        """
        Get core wallet allocation.

        TODO: Implement equal-weight allocation

        Returns:
            weights: Array of length K with equal weights summing to 1.0
        """
        # TODO: YOUR CODE HERE
        return np.ones(self.K) / self.K

    def get_bandit_weights(self):
        """
        Get bandit sleeve allocation using Thompson Sampling with guardrails.

        TODO: Implement Thompson Sampling for Gaussian rewards

        Steps:
        1. Sample from Normal(mean_i, std_i) for each commodity i
        2. Convert samples to weights using softmax
        3. Apply guardrails (min/max allocation, tilt speed)
        4. Re-normalize to sum to 1.0

        Returns:
            weights: Array of length K with weights summing to 1.0
        """
        # TODO: YOUR CODE HERE

        # Step 1: Sample from Normal posteriors
        samples = np.zeros(self.K)
        for i in range(self.K):
            samples[i] = norm.rvs(loc=self.means[i], scale=self.stds[i])

        # Step 2: Convert to weights using softmax (makes them positive & sum to 1)
        # Softmax: w_i = exp(s_i) / sum(exp(s_j))
        exp_samples = np.exp(samples - samples.max())  # Subtract max for numerical stability
        weights = exp_samples / exp_samples.sum()

        # Step 3: Apply guardrails
        # TODO: Implement min/max allocation constraints
        weights = np.clip(weights, self.min_alloc, self.max_alloc)
        weights = weights / weights.sum()  # Re-normalize

        # TODO: Implement tilt speed limit
        # Prevent allocation from changing too fast week-over-week
        # Hint: If |weights - prev_weights| > max_tilt, blend with prev_weights
        tilt = np.abs(weights - self.prev_bandit_weights)
        if np.any(tilt > self.max_tilt):
            # Blend with previous allocation
            alpha = 0.8  # Move 80% of the way to new allocation
            weights = alpha * weights + (1 - alpha) * self.prev_bandit_weights
            weights = weights / weights.sum()

        # Store for next iteration
        self.prev_bandit_weights = weights.copy()

        return weights

    def get_total_weights(self):
        """
        Combine core and bandit allocations.

        TODO: Implement weighted combination

        Returns:
            weights: Total portfolio weights (core + bandit)
        """
        # TODO: YOUR CODE HERE
        w_core = self.get_core_weights()
        w_bandit = self.get_bandit_weights()
        return self.core_pct * w_core + self.bandit_pct * w_bandit

    def update(self, returns, reward_type='sharpe'):
        """
        Update beliefs based on observed returns.

        TODO: Implement Bayesian update for Normal posteriors

        Args:
            returns: Array of observed returns for each commodity
            reward_type: 'raw', 'sharpe', or 'regret'
        """
        # Compute rewards based on type
        if reward_type == 'raw':
            rewards = returns
        elif reward_type == 'sharpe':
            # TODO: Implement Sharpe-based reward
            # Sharpe = return / volatility
            # Use rolling estimate of volatility
            volatilities = np.maximum(self.stds, 0.01)  # Avoid division by zero
            rewards = returns / volatilities
        elif reward_type == 'regret':
            # TODO: Implement regret-relative reward
            # Regret = your return - best return in hindsight
            best_return = np.max(returns)
            rewards = returns - best_return
        else:
            rewards = returns

        # Bayesian update for Normal-Normal conjugacy
        # TODO: YOUR CODE HERE
        for i in range(self.K):
            self.n[i] += 1

            # Simple Bayesian update (Kalman-like)
            learning_rate = 1 / (self.n[i] + 1)
            self.means[i] = (1 - learning_rate) * self.means[i] + learning_rate * rewards[i]

            # Update uncertainty (shrink as we learn)
            self.stds[i] = self.stds[i] / np.sqrt(1 + self.n[i] * 0.1)


def run_backtest(returns_df, initial_capital=100000, reward_type='sharpe'):
    """
    Run backtest of two-wallet strategy.

    This is COMPLETE - you don't need to modify it.
    """
    commodities = returns_df.columns.tolist()
    allocator = TwoWalletAllocator(commodities)

    # Tracking
    portfolio_values = [initial_capital]
    allocation_history = []
    dates = returns_df.index

    print(f"\n=== COMMODITY ALLOCATION ENGINE BACKTEST ===\n")
    print(f"Period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} weeks)")
    print(f"Starting Capital: ${initial_capital:,.0f}")
    print(f"Reward Type: {reward_type}\n")

    current_value = initial_capital

    for i, date in enumerate(dates):
        # Get allocation
        weights = allocator.get_total_weights()

        # Observe returns
        week_returns = returns_df.iloc[i].values

        # Calculate portfolio return
        portfolio_return = np.dot(weights, week_returns)

        # Update value
        current_value *= (1 + portfolio_return)
        portfolio_values.append(current_value)

        # Update beliefs
        allocator.update(week_returns, reward_type=reward_type)

        # Record allocation
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

    # Final report
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print('='*60)

    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100

    print(f"\nFinal Portfolio Value: ${final_value:,.0f}")
    print(f"Total Return: {total_return:+.2f}%")

    # Calculate Sharpe
    portfolio_returns = [h['portfolio_return'] for h in allocation_history]
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(52)
    print(f"Sharpe Ratio: {sharpe:.2f}")

    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (np.array(portfolio_values) - peak) / peak
    max_dd = np.min(drawdown) * 100
    print(f"Max Drawdown: {max_dd:.2f}%")

    # Win rate
    wins = sum(1 for r in portfolio_returns if r > 0)
    win_rate = wins / len(portfolio_returns) * 100
    print(f"Win Rate: {win_rate:.1f}% ({wins}/{len(portfolio_returns)} weeks)")

    # Final allocation
    print(f"\nFinal Allocation:")
    final_weights = allocation_history[-1]['weights']
    for commodity, weight in zip(commodities, final_weights):
        print(f"  {commodity:8s}: {weight*100:5.1f}%")

    return allocation_history, allocator


def plot_backtest_results(allocation_history, commodities):
    """
    Visualize backtest results.

    This is COMPLETE - creates useful plots automatically.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dates = [h['date'] for h in allocation_history]
    values = [h['value'] for h in allocation_history]

    # 1. Portfolio value
    ax = axes[0, 0]
    ax.plot(dates, values, linewidth=2)
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
    plt.savefig('commodity_backtest_results.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved visualization to 'commodity_backtest_results.png'")
    plt.show()


if __name__ == "__main__":
    # Load data
    loader = CommodityDataLoader(
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    returns_df = loader.load_data()

    # Run backtest
    allocation_history, allocator = run_backtest(
        returns_df,
        initial_capital=100000,
        reward_type='sharpe'  # Try: 'raw', 'sharpe', 'regret'
    )

    # Visualize
    plot_backtest_results(allocation_history, loader.commodities)

    print("\n🎉 Backtest complete!")
    print("\nNext steps:")
    print("1. Try different reward types: 'raw', 'sharpe', 'regret'")
    print("2. Adjust guardrails (min_allocation, max_allocation, max_tilt_speed)")
    print("3. Add transaction costs")
    print("4. Test on different time periods")
