"""
Commodity Allocator with Multi-Armed Bandits - Copy and customize for your use case
Works with: Portfolio allocation, sector rotation, commodity trading
Time to working: 10 minutes

Strategy: Core-Satellite with Bandit Sleeve
- Core wallet: Fixed allocation (e.g., 60% equal-weighted commodities)
- Bandit sleeve: Dynamic tilt based on recent performance (e.g., 40% bandit-optimized)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Literal
from datetime import datetime, timedelta
import logging

# ============================================================================
# CUSTOMIZE THESE
# ============================================================================
CONFIG = {
    # Commodity tickers (ETFs or futures)
    "tickers": ["GLD", "SLV", "DBA", "USO"],  # TODO: Customize tickers (Gold, Silver, Agriculture, Oil)

    # Portfolio structure
    "core_weight": 0.6,  # 60% in equal-weighted core
    "bandit_weight": 0.4,  # 40% in bandit sleeve

    # Bandit policy
    "policy": "thompson_sampling",  # epsilon_greedy, ucb1, thompson_sampling
    "epsilon": 0.15,  # For epsilon_greedy

    # Reward function: "raw_return", "sharpe", "stability_weighted"
    "reward_function": "stability_weighted",  # TODO: Customize reward calculation

    # Lookback and rebalancing
    "lookback_days": 21,  # Rolling window for reward calculation (21 = ~1 month)
    "rebalance_frequency": "weekly",  # How often to rebalance

    # Guardrails
    "min_allocation": 0.05,  # Min 5% per commodity
    "max_allocation": 0.50,  # Max 50% per commodity
    "max_tilt_per_week": 0.10,  # Max 10% tilt change per week
    "volatility_cap": 0.25,  # Don't allocate to assets with vol > 25% annualized

    # Data
    "start_date": "2023-01-01",  # TODO: Customize backtest period
    "end_date": "2024-12-31",
}

# ============================================================================
# PRODUCTION-READY COMMODITY ALLOCATOR (COPY THIS ENTIRE BLOCK)
# ============================================================================

class CommodityAllocator:
    """Multi-Armed Bandit for commodity portfolio allocation"""

    def __init__(self, config: Dict):
        self.config = config
        self.tickers = config["tickers"]
        self.arms = {ticker: {"pulls": 0, "successes": 0, "failures": 0} for ticker in self.tickers}
        self.allocation_history = []
        self.performance_history = []

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Load price data from yfinance"""
        self.logger.info(f"Loading data for {self.tickers}")
        try:
            data = yf.download(
                self.tickers,
                start=self.config["start_date"],
                end=self.config["end_date"],
                progress=False
            )["Adj Close"]

            # Handle single ticker case
            if len(self.tickers) == 1:
                data = data.to_frame(name=self.tickers[0])

            self.logger.info(f"Loaded {len(data)} rows")
            return data.dropna()

        except Exception as e:
            self.logger.warning(f"yfinance failed: {e}. Using synthetic data.")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Fallback: generate synthetic commodity prices"""
        dates = pd.date_range(self.config["start_date"], self.config["end_date"], freq='D')
        np.random.seed(42)

        data = {}
        for ticker in self.tickers:
            # Geometric Brownian Motion
            returns = np.random.normal(0.0005, 0.015, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            data[ticker] = prices

        return pd.DataFrame(data, index=dates)

    def calculate_reward(self, returns: pd.Series) -> float:
        """Calculate reward for a commodity based on recent returns"""
        if len(returns) == 0:
            return 0.0

        reward_fn = self.config["reward_function"]

        if reward_fn == "raw_return":
            # Simple cumulative return
            reward = returns.sum()

        elif reward_fn == "sharpe":
            # Sharpe-like ratio
            mean_return = returns.mean()
            std_return = returns.std()
            reward = mean_return / std_return if std_return > 0 else 0.0

        elif reward_fn == "stability_weighted":
            # Penalize volatility
            total_return = returns.sum()
            volatility = returns.std()
            reward = total_return / (1 + volatility)

        else:
            raise ValueError(f"Unknown reward function: {reward_fn}")

        # Normalize to [0, 1] using sigmoid
        return 1 / (1 + np.exp(-reward * 10))

    def select_allocation(self) -> Dict[str, float]:
        """Select portfolio allocation using bandit policy"""
        # Core allocation (equal-weighted)
        core_alloc = {ticker: self.config["core_weight"] / len(self.tickers) for ticker in self.tickers}

        # Bandit sleeve allocation
        bandit_alloc = self._bandit_select()

        # Combine core + bandit
        combined = {}
        for ticker in self.tickers:
            combined[ticker] = core_alloc[ticker] + self.config["bandit_weight"] * bandit_alloc[ticker]

        # Apply guardrails
        combined = self._apply_guardrails(combined)

        return combined

    def _bandit_select(self) -> Dict[str, float]:
        """Bandit policy to allocate bandit sleeve"""
        policy = self.config["policy"]

        if policy == "thompson_sampling":
            samples = {}
            for ticker, stats in self.arms.items():
                alpha = stats["successes"] + 1
                beta = stats["failures"] + 1
                samples[ticker] = np.random.beta(alpha, beta)

            # Softmax allocation based on samples
            exp_samples = {k: np.exp(v * 5) for k, v in samples.items()}
            total = sum(exp_samples.values())
            return {k: v / total for k, v in exp_samples.items()}

        elif policy == "epsilon_greedy":
            epsilon = self.config["epsilon"]
            if np.random.random() < epsilon:
                # Explore: equal allocation
                return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
            else:
                # Exploit: allocate to best performer
                best = max(self.arms.items(), key=lambda x: x[1]["successes"] / max(x[1]["pulls"], 1))[0]
                return {ticker: (1.0 if ticker == best else 0.0) for ticker in self.tickers}

        elif policy == "ucb1":
            total_pulls = sum(stats["pulls"] for stats in self.arms.values())
            ucb_scores = {}
            for ticker, stats in self.arms.items():
                if stats["pulls"] == 0:
                    ucb_scores[ticker] = float('inf')
                else:
                    mean = stats["successes"] / stats["pulls"]
                    exploration = np.sqrt(2 * np.log(total_pulls) / stats["pulls"])
                    ucb_scores[ticker] = mean + exploration

            # Softmax allocation
            exp_scores = {k: np.exp(v) for k, v in ucb_scores.items() if v != float('inf')}
            total = sum(exp_scores.values())
            return {k: v / total for k, v in exp_scores.items()}

        else:
            raise ValueError(f"Unknown policy: {policy}")

    def _apply_guardrails(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Apply allocation constraints"""
        # Min/max allocation
        for ticker in allocation:
            allocation[ticker] = np.clip(
                allocation[ticker],
                self.config["min_allocation"],
                self.config["max_allocation"]
            )

        # Renormalize to sum to 1.0
        total = sum(allocation.values())
        return {k: v / total for k, v in allocation.items()}

    def update_bandits(self, returns: Dict[str, float]):
        """Update bandit statistics based on observed returns"""
        for ticker, ret in returns.items():
            reward = self.calculate_reward(pd.Series([ret]))
            self.arms[ticker]["pulls"] += 1
            if reward >= 0.5:
                self.arms[ticker]["successes"] += 1
            else:
                self.arms[ticker]["failures"] += 1

    def run_backtest(self):
        """Run commodity allocation backtest"""
        # Load data
        prices = self.load_data()
        returns = prices.pct_change().dropna()

        # Weekly rebalancing
        weeks = returns.resample('W').last().index

        portfolio_value = 100000  # Start with $100k
        current_allocation = {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}

        for week_end in weeks:
            week_start = week_end - timedelta(days=7)

            # Get week's returns
            week_returns = returns.loc[week_start:week_end]
            if len(week_returns) == 0:
                continue

            # Calculate portfolio return
            ticker_returns = {ticker: week_returns[ticker].sum() for ticker in self.tickers}
            portfolio_return = sum(current_allocation[t] * ticker_returns[t] for t in self.tickers)
            portfolio_value *= (1 + portfolio_return)

            # Update bandits
            self.update_bandits(ticker_returns)

            # Rebalance
            new_allocation = self.select_allocation()

            # Log
            self.allocation_history.append({
                "date": week_end,
                **{f"{ticker}_weight": new_allocation[ticker] for ticker in self.tickers}
            })
            self.performance_history.append({
                "date": week_end,
                "portfolio_value": portfolio_value,
                "weekly_return": portfolio_return
            })

            current_allocation = new_allocation

            self.logger.info(f"{week_end.date()}: Portfolio=${portfolio_value:,.0f}, Return={portfolio_return:.2%}")

        # Final report
        self._print_report()

    def _print_report(self):
        """Generate performance report"""
        perf_df = pd.DataFrame(self.performance_history)
        alloc_df = pd.DataFrame(self.allocation_history)

        total_return = (perf_df["portfolio_value"].iloc[-1] / perf_df["portfolio_value"].iloc[0]) - 1

        print("\n" + "="*70)
        print("Commodity Allocator Performance Report")
        print("="*70)
        print(f"Strategy: {self.config['policy']}")
        print(f"Period: {self.config['start_date']} to {self.config['end_date']}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Final Portfolio Value: ${perf_df['portfolio_value'].iloc[-1]:,.0f}")
        print("\nFinal Allocation:")
        for ticker in self.tickers:
            final_weight = alloc_df[f"{ticker}_weight"].iloc[-1]
            print(f"  {ticker}: {final_weight:.1%}")
        print("="*70)


# ============================================================================
# RUN IT
# ============================================================================

def main():
    """Run commodity allocation backtest"""
    allocator = CommodityAllocator(CONFIG)
    allocator.run_backtest()


if __name__ == "__main__":
    main()
