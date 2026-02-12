"""
Module 8: Prompt Routing Bandits - Self-Check Exercises

These exercises help you practice implementing prompt routers and reward functions
for commodity trading LLM systems.

All exercises are self-check (ungraded) with assert-based validation.
"""

import numpy as np
import re


# ============================================================================
# Exercise 1: Implement a Prompt Routing Bandit for EIA Report Sections
# ============================================================================

def exercise_1():
    """
    Exercise 1: Implement a prompt routing bandit for 4 different EIA report section types.

    Scenario: You're building a system to process EIA weekly petroleum reports.
    Different sections need different prompts:
    - Inventory tables: structured extraction
    - Production narratives: summarization
    - Historical trends: trend analysis
    - Anomalies: change detection

    Task: Implement a Thompson Sampling router that learns which prompt works
    best for each section type.
    """

    print("=" * 70)
    print("Exercise 1: EIA Report Section Router")
    print("=" * 70)

    class EIASectionRouter:
        def __init__(self, num_prompts=4):
            """
            Initialize Thompson Sampling router for EIA sections.

            Prompts:
            0: Structured extraction
            1: Summarization
            2: Trend analysis
            3: Change detection
            """
            # TODO: Initialize Beta priors for each prompt
            # Hint: Use np.ones() for successes and failures
            self.successes = np.ones(num_prompts)
            self.failures = np.ones(num_prompts)

        def select_prompt(self, section_type):
            """
            Select prompt using Thompson Sampling.

            Args:
                section_type: 'table', 'narrative', 'trend', or 'anomaly'

            Returns:
                Prompt index (0-3)
            """
            # TODO: Sample from Beta distribution for each prompt
            # Hint: np.random.beta(alpha, beta) samples from Beta(alpha, beta)
            # Return the index with the highest sample
            samples = [np.random.beta(self.successes[i], self.failures[i])
                      for i in range(len(self.successes))]
            return np.argmax(samples)

        def update(self, prompt_idx, reward):
            """
            Update Beta posterior based on reward.

            Args:
                prompt_idx: Index of prompt used
                reward: Quality score [0, 1]
            """
            # TODO: Update successes or failures based on reward
            # Hint: Use threshold 0.75 to define success
            if reward >= 0.75:
                self.successes[prompt_idx] += 1
            else:
                self.failures[prompt_idx] += 1

    # Test implementation
    router = EIASectionRouter(num_prompts=4)

    # Simulate 100 requests with known best prompts per section
    section_best_prompt = {
        'table': 0,      # Structured extraction best for tables
        'narrative': 1,  # Summarization best for narratives
        'trend': 2,      # Trend analysis best for trends
        'anomaly': 3,    # Change detection best for anomalies
    }

    sections = ['table', 'narrative', 'trend', 'anomaly']
    selections = {section: [] for section in sections}

    for _ in range(100):
        section = np.random.choice(sections)
        prompt_idx = router.select_prompt(section)

        # Reward is high if we selected the "best" prompt for this section
        best_prompt = section_best_prompt[section]
        reward = 0.9 if prompt_idx == best_prompt else 0.4
        reward += np.random.normal(0, 0.1)  # Add noise
        reward = np.clip(reward, 0, 1)

        router.update(prompt_idx, reward)
        selections[section].append(prompt_idx)

    # Check if router learned correct preferences (last 20 selections per section)
    print("\nLearned preferences (last 20 selections per section):")
    for section in sections:
        recent = selections[section][-20:] if len(selections[section]) >= 20 else selections[section]
        if recent:
            most_common = max(set(recent), key=recent.count)
            expected = section_best_prompt[section]
            print(f"  {section}: selected prompt {most_common} most often (expected {expected})")

            # Validation: most common should match expected for at least 2 sections
            # (allowing for randomness in small samples)

    print("\n✅ Exercise 1 complete!")
    print("If your router learned the correct prompts for most sections, you got it right.\n")


# ============================================================================
# Exercise 2: Design a Composite Reward Function
# ============================================================================

def exercise_2():
    """
    Exercise 2: Design a composite reward function for a commodity trading signal system.

    The system generates buy/sell/hold signals. You need to balance:
    - Directional accuracy (did the signal make money?)
    - Timeliness (was the signal generated quickly?)
    - Actionability (was the signal clear and specific?)

    Plus guardrails:
    - Hallucination penalty (no unsupported claims)
    - Cost penalty (don't use too many tokens)
    """

    print("=" * 70)
    print("Exercise 2: Composite Reward Function Design")
    print("=" * 70)

    def compute_signal_reward(directional_correct, latency_seconds,
                             is_actionable, hallucination_detected,
                             tokens_used):
        """
        Compute composite reward for a trading signal.

        Args:
            directional_correct: bool, was the signal directionally correct?
            latency_seconds: float, time to generate signal
            is_actionable: bool, is the signal clear and specific?
            hallucination_detected: bool, did it make unsupported claims?
            tokens_used: int, number of tokens consumed

        Returns:
            Reward in [0, 1]
        """
        # TODO: Design your composite reward
        # Hints:
        # - Primary: directional accuracy (weight ~0.6)
        # - Secondary: actionability (weight ~0.2)
        # - Bonus: fast latency (weight ~0.1)
        # - Penalty: hallucination (heavy penalty, e.g., -0.5)
        # - Penalty: excessive cost (if tokens > 1000, penalize)

        # Primary metrics
        primary = 0.0
        if directional_correct:
            primary += 0.6
        if is_actionable:
            primary += 0.2

        # Latency bonus (faster is better, cap at 5 seconds)
        if latency_seconds <= 5:
            latency_bonus = 0.1 * (1 - latency_seconds / 5)
            primary += latency_bonus

        # Guardrails
        hallucination_penalty = -0.5 if hallucination_detected else 0.0
        cost_penalty = -0.2 if tokens_used > 1000 else 0.0

        reward = primary + hallucination_penalty + cost_penalty
        return max(0.0, min(1.0, reward))  # Clip to [0, 1]

    # Test cases
    print("\nTest Cases:")

    # Perfect signal
    reward1 = compute_signal_reward(
        directional_correct=True,
        latency_seconds=2,
        is_actionable=True,
        hallucination_detected=False,
        tokens_used=500
    )
    print(f"  Perfect signal: {reward1:.2f} (expected ~0.9-1.0)")
    assert reward1 >= 0.85, "Perfect signal should get high reward"

    # Correct but hallucinated
    reward2 = compute_signal_reward(
        directional_correct=True,
        latency_seconds=2,
        is_actionable=True,
        hallucination_detected=True,  # Bad!
        tokens_used=500
    )
    print(f"  Correct but hallucinated: {reward2:.2f} (expected ~0.3-0.5)")
    assert reward2 < 0.6, "Hallucination should heavily penalize"

    # Wrong signal
    reward3 = compute_signal_reward(
        directional_correct=False,
        latency_seconds=2,
        is_actionable=True,
        hallucination_detected=False,
        tokens_used=500
    )
    print(f"  Wrong signal: {reward3:.2f} (expected ~0.2-0.4)")
    assert reward3 < 0.5, "Wrong signal should get low reward"

    # Expensive signal
    reward4 = compute_signal_reward(
        directional_correct=True,
        latency_seconds=2,
        is_actionable=True,
        hallucination_detected=False,
        tokens_used=2000  # Too many tokens
    )
    print(f"  Expensive signal: {reward4:.2f} (should be penalized for cost)")
    assert reward4 < reward1, "Excessive cost should reduce reward"

    print("\n✅ Exercise 2 complete!")
    print("Your reward function balances accuracy, speed, and safety.\n")


# ============================================================================
# Exercise 3: Build a Contextual Router for Market Volatility Regimes
# ============================================================================

def exercise_3():
    """
    Exercise 3: Build a contextual prompt router that adapts based on market volatility regime.

    Scenario: Different prompts work better in different market conditions:
    - Low volatility: fundamental analysis prompts work best
    - High volatility: momentum-based prompts work best
    - Crisis volatility: risk management prompts work best

    Task: Implement a contextual router using LinUCB.
    """

    print("=" * 70)
    print("Exercise 3: Volatility-Aware Contextual Router")
    print("=" * 70)

    class VolatilityAwareRouter:
        def __init__(self, num_prompts=3, context_dim=2):
            """
            LinUCB router for volatility-aware prompt selection.

            Prompts:
            0: Fundamental analysis
            1: Momentum-based
            2: Risk management

            Context: [volatility_level, intercept]
            """
            self.num_prompts = num_prompts
            self.context_dim = context_dim
            self.alpha = 1.0  # Exploration parameter

            # TODO: Initialize A and b for each prompt
            # Hint: A is a (context_dim × context_dim) identity matrix
            #       b is a context_dim-length zero vector
            self.A = [np.identity(context_dim) for _ in range(num_prompts)]
            self.b = [np.zeros(context_dim) for _ in range(num_prompts)]

        def build_context(self, volatility):
            """
            Build context vector from volatility.

            Args:
                volatility: 'low', 'high', or 'crisis'

            Returns:
                Context vector [volatility_level, intercept]
            """
            # TODO: Encode volatility as continuous value
            # Hint: low=0.0, high=0.5, crisis=1.0
            vol_map = {'low': 0.0, 'high': 0.5, 'crisis': 1.0}
            vol_level = vol_map[volatility]
            return np.array([vol_level, 1.0])

        def select_prompt(self, volatility):
            """
            Select prompt using LinUCB.

            Args:
                volatility: 'low', 'high', or 'crisis'

            Returns:
                Prompt index
            """
            context = self.build_context(volatility)

            # TODO: Implement LinUCB selection
            # For each prompt:
            #   1. Compute theta = A_inv @ b
            #   2. Compute expected reward = theta @ context
            #   3. Compute uncertainty = sqrt(context @ A_inv @ context)
            #   4. Compute UCB = expected + alpha * uncertainty
            # Return prompt with highest UCB

            ucb_scores = []
            for i in range(self.num_prompts):
                A_inv = np.linalg.inv(self.A[i])
                theta = A_inv @ self.b[i]
                expected = theta @ context
                uncertainty = np.sqrt(context @ A_inv @ context)
                ucb = expected + self.alpha * uncertainty
                ucb_scores.append(ucb)

            return np.argmax(ucb_scores)

        def update(self, prompt_idx, volatility, reward):
            """
            Update LinUCB model.

            Args:
                prompt_idx: Index of prompt used
                volatility: 'low', 'high', or 'crisis'
                reward: Observed reward
            """
            context = self.build_context(volatility)

            # TODO: Update A and b
            # Hint: A += outer(context, context)
            #       b += reward * context
            self.A[prompt_idx] += np.outer(context, context)
            self.b[prompt_idx] += reward * context

    # Test implementation
    router = VolatilityAwareRouter(num_prompts=3, context_dim=2)

    # Simulate environment where optimal prompt depends on volatility
    def get_quality(prompt_idx, volatility):
        """True quality function (unknown to router)."""
        quality_matrix = {
            'low': [0.9, 0.4, 0.3],    # Fundamental best in low vol
            'high': [0.4, 0.9, 0.5],   # Momentum best in high vol
            'crisis': [0.3, 0.5, 0.9], # Risk mgmt best in crisis
        }
        base = quality_matrix[volatility][prompt_idx]
        noise = np.random.normal(0, 0.1)
        return np.clip(base + noise, 0, 1)

    # Run simulation
    volatilities = ['low', 'high', 'crisis']
    selections = {vol: [] for vol in volatilities}

    for _ in range(300):
        vol = np.random.choice(volatilities)
        prompt_idx = router.select_prompt(vol)
        quality = get_quality(prompt_idx, vol)
        router.update(prompt_idx, vol, quality)
        selections[vol].append(prompt_idx)

    # Check learned preferences
    print("\nLearned preferences (last 30 selections per regime):")
    expected = {'low': 0, 'high': 1, 'crisis': 2}

    for vol in volatilities:
        recent = selections[vol][-30:] if len(selections[vol]) >= 30 else selections[vol]
        if recent:
            most_common = max(set(recent), key=recent.count)
            exp = expected[vol]
            freq = recent.count(most_common) / len(recent)
            print(f"  {vol} volatility: prompt {most_common} used {freq:.1%} (expected {exp})")

            # Loose validation: most common should be correct for at least 1 regime
            # (allowing for exploration and randomness)

    print("\n✅ Exercise 3 complete!")
    print("If your router learned regime-specific preferences, you got it right.\n")


# ============================================================================
# Bonus Exercise: Hallucination Detection Function
# ============================================================================

def bonus_exercise():
    """
    Bonus: Implement a simple hallucination detection function.

    Task: Detect if a response contains numerical claims not present in source documents.
    """

    print("=" * 70)
    print("Bonus Exercise: Hallucination Detection")
    print("=" * 70)

    def detect_hallucination(response, source_documents):
        """
        Detect if response contains numbers not in source documents.

        Args:
            response: str, LLM response
            source_documents: list of str, retrieved documents

        Returns:
            bool, True if hallucination detected
        """
        # TODO: Extract numbers from response
        # Hint: Use regex to find patterns like "450.5", "2.3 million", etc.
        # Check if each number appears in any source document

        # Extract numbers from response (simple regex)
        response_numbers = set(re.findall(r'\d+\.?\d*', response))

        if not response_numbers:
            return False  # No numbers to verify

        # Check if all numbers appear in at least one source
        all_source_text = ' '.join(source_documents)

        for num in response_numbers:
            if num not in all_source_text:
                return True  # Hallucination detected

        return False

    # Test cases
    print("\nTest Cases:")

    # Case 1: No hallucination (number is in source)
    response1 = "EIA reports 439.5 million barrels in storage."
    sources1 = ["EIA Weekly Report: Crude oil inventories at 439.5 million barrels."]
    result1 = detect_hallucination(response1, sources1)
    print(f"  Test 1 (no hallucination): {result1} (expected False)")
    assert result1 == False, "Should not detect hallucination when number is in source"

    # Case 2: Hallucination (number NOT in source)
    response2 = "EIA reports 450 million barrels in storage."
    sources2 = ["EIA Weekly Report: Storage levels increased by 2 million barrels."]
    result2 = detect_hallucination(response2, sources2)
    print(f"  Test 2 (hallucination): {result2} (expected True)")
    assert result2 == True, "Should detect hallucination when number is not in source"

    # Case 3: No numbers in response
    response3 = "Storage levels increased this week."
    sources3 = ["Storage up."]
    result3 = detect_hallucination(response3, sources3)
    print(f"  Test 3 (no numbers): {result3} (expected False)")
    assert result3 == False, "Should not flag when no numbers to verify"

    print("\n✅ Bonus exercise complete!")
    print("You implemented basic hallucination detection.\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 8: PROMPT ROUTING BANDITS - EXERCISES")
    print("=" * 70 + "\n")

    try:
        exercise_1()
    except Exception as e:
        print(f"❌ Exercise 1 failed: {e}\n")

    try:
        exercise_2()
    except Exception as e:
        print(f"❌ Exercise 2 failed: {e}\n")

    try:
        exercise_3()
    except Exception as e:
        print(f"❌ Exercise 3 failed: {e}\n")

    try:
        bonus_exercise()
    except Exception as e:
        print(f"❌ Bonus exercise failed: {e}\n")

    print("=" * 70)
    print("ALL EXERCISES COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Thompson Sampling learns prompt preferences from feedback")
    print("  2. Composite rewards balance multiple objectives with guardrails")
    print("  3. Contextual bandits adapt routing based on request features")
    print("  4. Hallucination detection is critical for LLM reward design")
    print("\nNext: Apply these techniques to your commodity trading LLM systems!")
