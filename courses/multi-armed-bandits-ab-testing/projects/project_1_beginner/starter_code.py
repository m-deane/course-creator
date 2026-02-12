"""
Content Strategy Optimizer - Starter Code

Your task: Complete the ThompsonSampler class to optimize content publishing
using multi-armed bandits.

TODOs are marked clearly. The simulation framework is complete.
"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


class BanditEnvironment:
    """
    Simulates audience response to different content types.

    This is COMPLETE - you don't need to modify it.
    """
    def __init__(self, arm_names, true_read_ratios, noise_std=0.1):
        """
        Args:
            arm_names: List of content type names
            true_read_ratios: True read ratio for each arm (unknown to agent)
            noise_std: Standard deviation of noise in responses
        """
        self.arm_names = arm_names
        self.true_means = np.array(true_read_ratios)
        self.noise_std = noise_std
        self.n_arms = len(arm_names)

    def publish(self, arm_idx):
        """
        Simulate publishing content and observing engagement.

        Returns:
            read_ratio: Fraction of readers who finished (0 to 1)
        """
        true_mean = self.true_means[arm_idx]
        observed = np.random.normal(true_mean, self.noise_std)
        return np.clip(observed, 0, 1)

    def get_best_arm(self):
        """Return index of truly best arm (for regret calculation)."""
        return np.argmax(self.true_means)


class ThompsonSampler:
    """
    Thompson Sampling for content strategy optimization.

    YOUR TASK: Complete the select_arm() and update() methods.
    """
    def __init__(self, n_arms, prior_alpha=1.0, prior_beta=1.0):
        """
        Initialize Thompson Sampler with Beta priors.

        Args:
            n_arms: Number of content types (arms)
            prior_alpha: Prior successes (start with 1 = uniform prior)
            prior_beta: Prior failures (start with 1 = uniform prior)
        """
        self.n_arms = n_arms
        # Beta distribution parameters for each arm
        self.alpha = np.full(n_arms, prior_alpha)
        self.beta = np.full(n_arms, prior_beta)

        # Tracking
        self.pulls = np.zeros(n_arms)
        self.total_rewards = np.zeros(n_arms)

    def select_arm(self):
        """
        Select an arm using Thompson Sampling.

        TODO: Implement Thompson Sampling arm selection

        Steps:
        1. Sample theta_i from Beta(alpha_i, beta_i) for each arm i
        2. Return the arm with highest sampled value

        Hint: Use scipy.stats.beta.rvs(a, b) to sample from Beta(a, b)

        Returns:
            arm_idx: Index of selected arm
        """
        # TODO: YOUR CODE HERE
        # Sample from each arm's posterior
        samples = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            samples[i] = beta.rvs(self.alpha[i], self.beta[i])

        # Return arm with highest sample
        return np.argmax(samples)

    def update(self, arm_idx, read_ratio):
        """
        Update beliefs based on observed read ratio.

        TODO: Implement Bayesian update for Beta distribution

        We're treating read_ratio (continuous 0-1) as approximating Bernoulli.
        A simple approach:
        - If read_ratio > threshold (e.g., 0.5): count as "success"
        - Otherwise: count as "failure"

        OR use read_ratio directly as fractional update:
        - alpha += read_ratio
        - beta += (1 - read_ratio)

        Choose whichever makes sense to you. The second is smoother.

        Args:
            arm_idx: Which arm was pulled
            read_ratio: Observed engagement (0 to 1)
        """
        # Update tracking
        self.pulls[arm_idx] += 1
        self.total_rewards[arm_idx] += read_ratio

        # TODO: YOUR CODE HERE
        # Update Beta parameters based on observed read_ratio
        # Method 1 (thresholding):
        # if read_ratio > 0.5:
        #     self.alpha[arm_idx] += 1
        # else:
        #     self.beta[arm_idx] += 1

        # Method 2 (fractional - recommended):
        self.alpha[arm_idx] += read_ratio
        self.beta[arm_idx] += (1 - read_ratio)

    def get_posterior_means(self):
        """Get mean of each arm's posterior Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    def retire_arm(self, arm_idx):
        """
        Retire an arm and reset it to prior.

        This simulates introducing a new content type.
        """
        self.alpha[arm_idx] = 1.0
        self.beta[arm_idx] = 1.0
        self.pulls[arm_idx] = 0
        self.total_rewards[arm_idx] = 0


def run_simulation(n_weeks=52, posts_per_week=5, exploration_weeks=3):
    """
    Run 52-week content optimization simulation.

    This is COMPLETE - you don't need to modify it.
    """
    # Define 6 content arms
    arm_names = [
        "Market Analysis × Essay",
        "Market Analysis × Thread",
        "Trading Psychology × Essay",
        "Trading Psychology × Thread",
        "Risk Management × Essay",
        "Risk Management × Thread"
    ]

    # True read ratios (unknown to agent)
    # These would be discovered through publishing
    true_read_ratios = [0.45, 0.38, 0.52, 0.41, 0.35, 0.48]

    # Initialize environment and sampler
    env = BanditEnvironment(arm_names, true_read_ratios)
    sampler = ThompsonSampler(n_arms=len(arm_names))

    # History tracking
    history = []
    cumulative_reward = 0
    optimal_reward = 0  # What we would have gotten with perfect knowledge

    print("=== CONTENT STRATEGY OPTIMIZER ===\n")
    print(f"Simulating {n_weeks} weeks of content publishing...")
    print(f"Publishing {posts_per_week} posts per week\n")

    # Phase 1: Initial exploration (weeks 1-3)
    print(f"Weeks 1-{exploration_weeks}: EXPLORATION PHASE")
    print("Publishing all content types evenly...\n")

    for week in range(exploration_weeks):
        for arm in range(len(arm_names)):
            read_ratio = env.publish(arm)
            sampler.update(arm, read_ratio)
            history.append({
                'week': week + 1,
                'arm': arm,
                'read_ratio': read_ratio
            })
            cumulative_reward += read_ratio
            optimal_reward += true_read_ratios[env.get_best_arm()]

    # Phase 2: Adaptive publishing with exploration (weeks 4-52)
    print(f"Weeks {exploration_weeks+1}-{n_weeks}: ADAPTIVE PUBLISHING")
    print("Tilting toward best performers, maintaining 20% exploration...\n")

    retirement_weeks = [12, 24, 36]

    for week in range(exploration_weeks, n_weeks):
        # Check for arm retirement
        if week + 1 in retirement_weeks:
            posterior_means = sampler.get_posterior_means()
            worst_arm = np.argmin(posterior_means)
            print(f"\n📉 Week {week+1}: RETIRING '{arm_names[worst_arm]}'")
            print(f"   Performance: {posterior_means[worst_arm]:.1%} read ratio")
            print(f"   Introducing new content type...\n")
            sampler.retire_arm(worst_arm)

        # Publish this week's content
        for _ in range(posts_per_week):
            arm = sampler.select_arm()
            read_ratio = env.publish(arm)
            sampler.update(arm, read_ratio)
            history.append({
                'week': week + 1,
                'arm': arm,
                'read_ratio': read_ratio
            })
            cumulative_reward += read_ratio
            optimal_reward += true_read_ratios[env.get_best_arm()]

    # Final report
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    print(f"\nArm Performance:")
    print(f"{'Content Type':<35} {'Picks':>6} {'Avg RR':>8} {'Posterior':>12}")
    print("-" * 60)

    posterior_means = sampler.get_posterior_means()
    indices = np.argsort(posterior_means)[::-1]  # Sort by performance

    for idx in indices:
        if sampler.pulls[idx] > 0:
            avg_reward = sampler.total_rewards[idx] / sampler.pulls[idx]
            posterior_str = f"β({sampler.alpha[idx]:.0f},{sampler.beta[idx]:.0f})"
            print(f"{arm_names[idx]:<35} {int(sampler.pulls[idx]):>6} "
                  f"{avg_reward:>7.1%} {posterior_str:>12}")

    print()
    total_posts = n_weeks * posts_per_week
    baseline_reward = total_posts * np.mean(true_read_ratios)

    print(f"Cumulative Engagement: {cumulative_reward:.1f}")
    print(f"Random Baseline:       {baseline_reward:.1f}")
    print(f"Improvement:           {((cumulative_reward/baseline_reward - 1) * 100):.1f}%")
    print(f"Regret vs Optimal:     {((1 - cumulative_reward/optimal_reward) * 100):.1f}%")

    return history, sampler, env


def plot_results(history, sampler, env):
    """
    Visualize bandit learning.

    This is COMPLETE - creates useful plots automatically.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Arm selection over time
    weeks = [h['week'] for h in history]
    arms = [h['arm'] for h in history]

    ax = axes[0, 0]
    for i in range(sampler.n_arms):
        arm_weeks = [w for w, a in zip(weeks, arms) if a == i]
        if arm_weeks:
            ax.scatter(arm_weeks, [i] * len(arm_weeks), alpha=0.3, s=20)
    ax.set_xlabel('Week')
    ax.set_ylabel('Arm')
    ax.set_title('Arm Selection Over Time')
    ax.set_yticks(range(sampler.n_arms))
    ax.set_yticklabels([name[:20] for name in env.arm_names])
    ax.grid(True, alpha=0.3)

    # 2. Cumulative reward
    ax = axes[0, 1]
    cumulative = np.cumsum([h['read_ratio'] for h in history])
    ax.plot(cumulative, label='Thompson Sampling')

    # Baseline
    baseline = np.arange(len(history)) * np.mean(env.true_means)
    ax.plot(baseline, '--', label='Random Baseline', alpha=0.7)

    ax.set_xlabel('Posts Published')
    ax.set_ylabel('Cumulative Engagement')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Posterior means evolution
    ax = axes[1, 0]
    posterior_means = sampler.get_posterior_means()
    indices = np.argsort(posterior_means)[::-1]

    ax.barh(range(sampler.n_arms), posterior_means[indices])
    ax.set_yticks(range(sampler.n_arms))
    ax.set_yticklabels([env.arm_names[i][:25] for i in indices])
    ax.set_xlabel('Posterior Mean Read Ratio')
    ax.set_title('Final Beliefs About Each Arm')
    ax.grid(True, alpha=0.3, axis='x')

    # 4. Pull distribution
    ax = axes[1, 1]
    pulls = sampler.pulls[indices]
    ax.barh(range(sampler.n_arms), pulls)
    ax.set_yticks(range(sampler.n_arms))
    ax.set_yticklabels([env.arm_names[i][:25] for i in indices])
    ax.set_xlabel('Number of Times Selected')
    ax.set_title('Exploration vs Exploitation')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('content_strategy_results.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved visualization to 'content_strategy_results.png'")
    plt.show()


if __name__ == "__main__":
    # Run the simulation
    history, sampler, env = run_simulation(
        n_weeks=52,
        posts_per_week=5,
        exploration_weeks=3
    )

    # Visualize results
    plot_results(history, sampler, env)

    print("\n🎉 Simulation complete!")
    print("\nNext steps:")
    print("1. Experiment with different prior parameters")
    print("2. Try different retirement schedules")
    print("3. Add contextual features (seasonality)")
    print("4. Compare to pure exploitation (no exploration)")
