"""
Content Strategy Optimizer - Complete Solution

This is the reference implementation. Only look at this if you're stuck!
Try to complete starter_code.py on your own first.
"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


class BanditEnvironment:
    """Simulates audience response to different content types."""

    def __init__(self, arm_names, true_read_ratios, noise_std=0.1):
        self.arm_names = arm_names
        self.true_means = np.array(true_read_ratios)
        self.noise_std = noise_std
        self.n_arms = len(arm_names)

    def publish(self, arm_idx):
        """Simulate publishing content and observing engagement."""
        true_mean = self.true_means[arm_idx]
        observed = np.random.normal(true_mean, self.noise_std)
        return np.clip(observed, 0, 1)

    def get_best_arm(self):
        """Return index of truly best arm."""
        return np.argmax(self.true_means)


class ThompsonSampler:
    """Thompson Sampling for content optimization - COMPLETE SOLUTION."""

    def __init__(self, n_arms, prior_alpha=1.0, prior_beta=1.0):
        self.n_arms = n_arms
        self.alpha = np.full(n_arms, prior_alpha)
        self.beta = np.full(n_arms, prior_beta)
        self.pulls = np.zeros(n_arms)
        self.total_rewards = np.zeros(n_arms)

    def select_arm(self):
        """
        Thompson Sampling arm selection.

        Sample from each arm's Beta posterior and pick the best sample.
        """
        samples = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            samples[i] = beta.rvs(self.alpha[i], self.beta[i])
        return np.argmax(samples)

    def update(self, arm_idx, read_ratio):
        """
        Update Beta posterior based on observed read ratio.

        Uses fractional updating: treat read_ratio as fractional success.
        """
        self.pulls[arm_idx] += 1
        self.total_rewards[arm_idx] += read_ratio

        # Fractional update (smooth)
        self.alpha[arm_idx] += read_ratio
        self.beta[arm_idx] += (1 - read_ratio)

    def get_posterior_means(self):
        """Mean of each arm's posterior."""
        return self.alpha / (self.alpha + self.beta)

    def retire_arm(self, arm_idx):
        """Reset arm to prior (simulate new content type)."""
        self.alpha[arm_idx] = 1.0
        self.beta[arm_idx] = 1.0
        self.pulls[arm_idx] = 0
        self.total_rewards[arm_idx] = 0


def run_simulation(n_weeks=52, posts_per_week=5, exploration_weeks=3):
    """Run complete 52-week simulation."""

    arm_names = [
        "Market Analysis × Essay",
        "Market Analysis × Thread",
        "Trading Psychology × Essay",
        "Trading Psychology × Thread",
        "Risk Management × Essay",
        "Risk Management × Thread"
    ]

    # True unknown read ratios
    true_read_ratios = [0.45, 0.38, 0.52, 0.41, 0.35, 0.48]

    env = BanditEnvironment(arm_names, true_read_ratios)
    sampler = ThompsonSampler(n_arms=len(arm_names))

    history = []
    cumulative_reward = 0
    optimal_reward = 0

    print("=== CONTENT STRATEGY OPTIMIZER ===\n")
    print(f"Simulating {n_weeks} weeks of content publishing...")
    print(f"Publishing {posts_per_week} posts per week\n")

    # Phase 1: Exploration
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

    # Phase 2: Adaptive publishing
    print(f"Weeks {exploration_weeks+1}-{n_weeks}: ADAPTIVE PUBLISHING")
    print("Tilting toward best performers, maintaining exploration...\n")

    retirement_weeks = [12, 24, 36]

    for week in range(exploration_weeks, n_weeks):
        # Arm retirement
        if week + 1 in retirement_weeks:
            posterior_means = sampler.get_posterior_means()
            worst_arm = np.argmin(posterior_means)
            print(f"\n📉 Week {week+1}: RETIRING '{arm_names[worst_arm]}'")
            print(f"   Performance: {posterior_means[worst_arm]:.1%} read ratio")
            print(f"   Introducing new content type...\n")
            sampler.retire_arm(worst_arm)

        # Publish
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

    # Results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    print(f"\nArm Performance:")
    print(f"{'Content Type':<35} {'Picks':>6} {'Avg RR':>8} {'Posterior':>12}")
    print("-" * 60)

    posterior_means = sampler.get_posterior_means()
    indices = np.argsort(posterior_means)[::-1]

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


def compare_strategies():
    """
    Compare Thompson Sampling vs Random vs Pure Exploit.

    BONUS: Shows why exploration matters.
    """
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60 + "\n")

    arm_names = ["Arm A", "Arm B", "Arm C", "Arm D", "Arm E", "Arm F"]
    true_means = [0.45, 0.38, 0.52, 0.41, 0.35, 0.48]

    n_rounds = 260  # 52 weeks × 5 posts

    # Random strategy
    env_random = BanditEnvironment(arm_names, true_means)
    random_reward = 0
    for _ in range(n_rounds):
        arm = np.random.randint(len(arm_names))
        reward = env_random.publish(arm)
        random_reward += reward

    # Pure exploit (greedy with initial exploration)
    env_exploit = BanditEnvironment(arm_names, true_means)
    exploit_rewards = np.zeros(len(arm_names))
    exploit_counts = np.zeros(len(arm_names))

    # Initial exploration (10 rounds per arm)
    for arm in range(len(arm_names)):
        for _ in range(10):
            reward = env_exploit.publish(arm)
            exploit_rewards[arm] += reward
            exploit_counts[arm] += 1

    # Pure exploitation
    exploit_reward = exploit_rewards.sum()
    for _ in range(n_rounds - 60):  # Remaining rounds
        best_arm = np.argmax(exploit_rewards / exploit_counts)
        reward = env_exploit.publish(best_arm)
        exploit_reward += reward

    # Thompson Sampling
    env_ts = BanditEnvironment(arm_names, true_means)
    sampler = ThompsonSampler(len(arm_names))
    ts_reward = 0
    for _ in range(n_rounds):
        arm = sampler.select_arm()
        reward = env_ts.publish(arm)
        sampler.update(arm, reward)
        ts_reward += reward

    # Optimal (cheating - know true best)
    optimal_reward = n_rounds * max(true_means)

    print("Strategy Performance:")
    print(f"  Random:             {random_reward:.1f} ({random_reward/optimal_reward:.1%} of optimal)")
    print(f"  Pure Exploit:       {exploit_reward:.1f} ({exploit_reward/optimal_reward:.1%} of optimal)")
    print(f"  Thompson Sampling:  {ts_reward:.1f} ({ts_reward/optimal_reward:.1%} of optimal)")
    print(f"  Optimal (oracle):   {optimal_reward:.1f} (100%)")

    print(f"\nThompson Sampling improvement:")
    print(f"  vs Random:  {((ts_reward/random_reward - 1) * 100):+.1f}%")
    print(f"  vs Exploit: {((ts_reward/exploit_reward - 1) * 100):+.1f}%")


def plot_results(history, sampler, env):
    """Visualize bandit learning."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    weeks = [h['week'] for h in history]
    arms = [h['arm'] for h in history]

    # 1. Arm selection timeline
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
    ax.plot(cumulative, label='Thompson Sampling', linewidth=2)
    baseline = np.arange(len(history)) * np.mean(env.true_means)
    ax.plot(baseline, '--', label='Random Baseline', alpha=0.7)
    ax.set_xlabel('Posts Published')
    ax.set_ylabel('Cumulative Engagement')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Final posterior beliefs
    ax = axes[1, 0]
    posterior_means = sampler.get_posterior_means()
    indices = np.argsort(posterior_means)[::-1]
    colors = ['green' if env.true_means[i] == max(env.true_means) else 'steelblue'
              for i in indices]
    ax.barh(range(sampler.n_arms), posterior_means[indices], color=colors)
    ax.set_yticks(range(sampler.n_arms))
    ax.set_yticklabels([env.arm_names[i][:25] for i in indices])
    ax.set_xlabel('Posterior Mean Read Ratio')
    ax.set_title('Final Beliefs (green = true best)')
    ax.grid(True, alpha=0.3, axis='x')

    # 4. Exploration distribution
    ax = axes[1, 1]
    pulls = sampler.pulls[indices]
    ax.barh(range(sampler.n_arms), pulls, color=colors)
    ax.set_yticks(range(sampler.n_arms))
    ax.set_yticklabels([env.arm_names[i][:25] for i in indices])
    ax.set_xlabel('Number of Times Selected')
    ax.set_title('Exploration Distribution')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('content_strategy_solution.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved visualization to 'content_strategy_solution.png'")
    plt.show()


if __name__ == "__main__":
    # Run main simulation
    history, sampler, env = run_simulation(
        n_weeks=52,
        posts_per_week=5,
        exploration_weeks=3
    )

    # Compare strategies
    compare_strategies()

    # Visualize
    plot_results(history, sampler, env)

    print("\n🎉 Complete solution demonstrated!")
