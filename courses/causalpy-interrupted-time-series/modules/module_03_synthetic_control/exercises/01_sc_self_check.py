"""
Module 03 Self-Check Exercises: Synthetic Control Methods

Run each check function to verify your understanding.
All exercises are self-graded — no submission required.

Usage:
    python 01_sc_self_check.py
    # or interactively:
    from exercises.sc_self_check import *
"""

import numpy as np
import textwrap


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_result(correct: bool, feedback_correct: str, feedback_wrong: str) -> None:
    if correct:
        print(f"  CORRECT. {feedback_correct}")
    else:
        print(f"  INCORRECT. {feedback_wrong}")


# ---------------------------------------------------------------------------
# Exercise 1: When to use synthetic control vs ITS
# ---------------------------------------------------------------------------

SC_VS_ITS_SCENARIOS = {
    1: {
        "description": (
            "A state implements a new texting-while-driving ban. You have 5 years of monthly\n"
            "accident data for this state only. No other states implemented similar bans.\n"
            "Which method is appropriate?"
        ),
        "answer": "its",
        "explanation": (
            "You have only one unit (the state), so synthetic control is not possible — it requires\n"
            "multiple untreated donor units. ITS is the correct choice: use the state's own\n"
            "pre-intervention trend as the counterfactual."
        ),
    },
    2: {
        "description": (
            "One city implements a congestion pricing scheme. You have monthly traffic data for\n"
            "20 other comparable cities that did not implement congestion pricing. The pricing\n"
            "was implemented during an unusually hot summer that may have independently reduced\n"
            "driving. Which method is more appropriate?"
        ),
        "answer": "synthetic_control",
        "explanation": (
            "Synthetic control is better here for two reasons: (1) you have 20 donor cities,\n"
            "enabling a valid synthetic counterpart; (2) the concurrent event (hot summer) would\n"
            "bias an ITS estimate, but synthetic control differences it out if the donor cities\n"
            "also experienced the hot summer. The concurrent event threat strongly favors SC."
        ),
    },
    3: {
        "description": (
            "A firm implements a new HR policy. You have 36 months of employee turnover data\n"
            "for this firm. You have 50 peer firms that did not change HR policy. The treated\n"
            "firm's pre-period turnover is very similar to only 3 of the 50 peer firms.\n"
            "Should you use synthetic control?"
        ),
        "answer": "synthetic_control_with_caution",
        "explanation": (
            "Synthetic control is feasible (50 donors available), but the poor comparability of\n"
            "most donors is a concern. The SC weights will likely concentrate on the 3 similar\n"
            "firms. Inspect the pre-period fit carefully (RMSPE). If the fit is good despite\n"
            "only 3 usable donors, SC is valid. If the pre-period RMSPE is large, ITS may be\n"
            "more credible since the parallel-trends assumption may actually hold better here."
        ),
    },
    4: {
        "description": (
            "You want to evaluate a national minimum wage increase. The entire country (one\n"
            "treated unit) implemented the policy simultaneously. You have data for 30 other\n"
            "countries as potential donors."
        ),
        "answer": "synthetic_control",
        "explanation": (
            "This is a classic synthetic control design — one treated unit (the country),\n"
            "multiple donor countries, no same-time treatment for donors. SC is the appropriate\n"
            "method. The key assumption is that donor countries share macroeconomic trends with\n"
            "the treated country and did not implement similar policies during the study period."
        ),
    },
}


def check_method_choice(scenario_number: int, answer: str) -> None:
    """
    Check whether synthetic control or ITS is more appropriate for a scenario.

    Parameters
    ----------
    scenario_number : int
        1 to 4
    answer : str
        'its', 'synthetic_control', or 'synthetic_control_with_caution'

    Examples
    --------
    >>> check_method_choice(1, 'its')   # correct
    >>> check_method_choice(2, 'its')   # incorrect
    """
    if scenario_number not in SC_VS_ITS_SCENARIOS:
        print(f"  Scenario {scenario_number} not found. Choose 1–4.")
        return

    scenario = SC_VS_ITS_SCENARIOS[scenario_number]
    print(f"\nScenario {scenario_number}: {scenario['description']}")

    correct = answer.lower().strip() == scenario["answer"]
    _print_result(
        correct,
        f"Explanation: {scenario['explanation']}",
        f"The correct answer is '{scenario['answer']}'. {scenario['explanation']}",
    )


# ---------------------------------------------------------------------------
# Exercise 2: Interpreting pre-period fit
# ---------------------------------------------------------------------------

PREFIT_CASES = {
    1: {
        "description": (
            "Pre-period RMSPE = 18.5 for the treated unit. Pre-period standard deviation = 22.0.\n"
            "Is the pre-period fit adequate?"
        ),
        "answer": "poor",
        "explanation": (
            "RMSPE of 18.5 is 84% of the pre-period standard deviation — far above the 20%\n"
            "guideline. The synthetic control does not closely match the treated unit in the\n"
            "pre-period. This could indicate: (1) the treated unit is not in the convex hull\n"
            "of donors; (2) the donor pool is poorly chosen; (3) more predictors are needed.\n"
            "Do not trust the post-period causal estimate with this pre-period fit."
        ),
    },
    2: {
        "description": (
            "Pre-period RMSPE = 1.2. Pre-period standard deviation = 14.8.\n"
            "Is the pre-period fit adequate?"
        ),
        "answer": "good",
        "explanation": (
            "RMSPE of 1.2 is 8% of the pre-period standard deviation — well below the 20%\n"
            "guideline. The synthetic control closely tracks the treated unit in the pre-period.\n"
            "This is a credible synthetic control analysis. The post-period gap is unlikely to\n"
            "be an artifact of poor pre-period fit."
        ),
    },
    3: {
        "description": (
            "Pre-period RMSPE = 4.5 for California. Donor A has pre-period RMSPE = 3.2.\n"
            "Donor B has pre-period RMSPE = 15.7. Should Donor B be excluded from the placebo test?"
        ),
        "answer": "exclude",
        "explanation": (
            "Donor B's pre-period RMSPE (15.7) is 3.5x California's pre-period RMSPE (4.5).\n"
            "This exceeds the standard 2x threshold for exclusion. Units with poor pre-period\n"
            "fit generate large post-period gaps from noise, inflating the placebo null\n"
            "distribution and making it harder to detect California's true effect. Exclude\n"
            "Donor B and report both the full and filtered p-values."
        ),
    },
}


def check_prefit_interpretation(case_number: int, answer: str) -> None:
    """
    Interpret a pre-period fit result.

    Parameters
    ----------
    case_number : int
        1 to 3
    answer : str
        'good', 'poor', 'exclude', or 'include'

    Examples
    --------
    >>> check_prefit_interpretation(1, 'poor')   # correct
    >>> check_prefit_interpretation(2, 'good')   # correct
    >>> check_prefit_interpretation(3, 'exclude')  # correct
    """
    if case_number not in PREFIT_CASES:
        print(f"  Case {case_number} not found. Choose 1–3.")
        return

    case = PREFIT_CASES[case_number]
    print(f"\nCase {case_number}: {case['description']}")

    correct = answer.lower().strip() == case["answer"]
    _print_result(
        correct,
        f"Explanation: {case['explanation']}",
        f"The correct answer is '{case['answer']}'. {case['explanation']}",
    )


# ---------------------------------------------------------------------------
# Exercise 3: Permutation p-value
# ---------------------------------------------------------------------------

PVALUE_CASES = {
    1: {
        "description": (
            "You have 1 treated unit and 8 donor units (9 total). The treated unit has the\n"
            "largest RMSPE ratio among all 9 units. What is the permutation p-value?"
        ),
        "answer": 1.0 / 9,
        "tolerance": 0.001,
        "explanation": (
            "P-value = 1 / number of units = 1/9 ≈ 0.111. The treated unit has the largest\n"
            "ratio, so 1 out of 9 units has a ratio >= the treated unit. This is the minimum\n"
            "achievable p-value with 8 donors. It is not significant at the 5% level — you\n"
            "need at least 19 donors to achieve 5% significance."
        ),
    },
    2: {
        "description": (
            "You have 39 units (1 treated + 38 donors). The treated unit has the 3rd largest\n"
            "RMSPE ratio. What is the permutation p-value?"
        ),
        "answer": 3.0 / 39,
        "tolerance": 0.001,
        "explanation": (
            "P-value = 3/39 ≈ 0.077. Three units (the treated unit + 2 donors) have RMSPE\n"
            "ratios >= the treated unit's ratio. This is marginal — not significant at 5%\n"
            "but suggestive at 10%."
        ),
    },
    3: {
        "description": (
            "You have 20 good-fit units (1 treated + 19 donors after excluding poor-fit donors).\n"
            "The treated unit has the largest RMSPE ratio. What is the filtered p-value?"
        ),
        "answer": 1.0 / 20,
        "tolerance": 0.001,
        "explanation": (
            "Filtered p-value = 1/20 = 0.05. The treated unit has the largest ratio among all\n"
            "20 good-fit units. This is exactly 5% — just barely significant. Report this as\n"
            "p = 0.05 and note the minimum achievable p-value given the donor pool size."
        ),
    },
}


def check_pvalue_computation(case_number: int, your_pvalue: float) -> None:
    """
    Verify your permutation p-value calculation.

    Parameters
    ----------
    case_number : int
        1 to 3
    your_pvalue : float
        Your computed p-value (e.g., 0.111, 0.077)

    Examples
    --------
    >>> check_pvalue_computation(1, 1/9)   # correct
    >>> check_pvalue_computation(2, 1/39)  # incorrect
    """
    if case_number not in PVALUE_CASES:
        print(f"  Case {case_number} not found. Choose 1–3.")
        return

    case = PVALUE_CASES[case_number]
    print(f"\nCase {case_number}: {case['description']}")

    correct = abs(your_pvalue - case["answer"]) <= case["tolerance"]
    _print_result(
        correct,
        f"Explanation: {case['explanation']}",
        f"The correct p-value is {case['answer']:.4f}. {case['explanation']}",
    )


# ---------------------------------------------------------------------------
# Exercise 4: Weight interpretation
# ---------------------------------------------------------------------------

WEIGHT_CASES = {
    1: {
        "description": (
            "The Bayesian SC weight posterior for 'Utah' has mean=0.002 and 94% HDI=[0.000, 0.008].\n"
            "What does this mean?"
        ),
        "answer": "not_contributing",
        "explanation": (
            "Utah's weight is effectively zero — the synthetic control does not need Utah to match\n"
            "the treated unit's pre-period trajectory. Including Utah in the donor pool is harmless\n"
            "(it just gets zero weight) but it is not driving the result."
        ),
    },
    2: {
        "description": (
            "The Bayesian SC weight posterior for 'Colorado' has mean=0.43 and 94% HDI=[0.31, 0.55].\n"
            "What does this mean?"
        ),
        "answer": "major_donor",
        "explanation": (
            "Colorado is the primary driver of the synthetic California — it contributes\n"
            "approximately 43% of the counterfactual. The HDI [0.31, 0.55] means the posterior\n"
            "is concentrated well away from zero, so Colorado's contribution is statistically\n"
            "well-determined. This is not just noise — Colorado genuinely matches California's\n"
            "pre-period trajectory."
        ),
    },
    3: {
        "description": (
            "The Bayesian SC weight posteriors show that 15 out of 20 donors have mean weight > 0.01.\n"
            "Is this a good sign or a concern?"
        ),
        "answer": "concern",
        "explanation": (
            "When many donors have non-trivial weights (especially in a model with many predictors),\n"
            "this often indicates the weights are driven by in-sample overfitting rather than genuine\n"
            "comparability. A good synthetic control should have sparse weights — only a few donors\n"
            "that genuinely match the treated unit's trajectory. 15 active donors out of 20 is a\n"
            "red flag. Consider restricting to a smaller donor pool or using stronger regularization."
        ),
    },
}


def check_weight_interpretation(case_number: int, answer: str) -> None:
    """
    Interpret donor weights from a Bayesian synthetic control.

    Parameters
    ----------
    case_number : int
        1 to 3
    answer : str
        'not_contributing', 'major_donor', 'concern', or 'good_sign'

    Examples
    --------
    >>> check_weight_interpretation(1, 'not_contributing')   # correct
    >>> check_weight_interpretation(2, 'major_donor')        # correct
    >>> check_weight_interpretation(3, 'concern')            # correct
    """
    if case_number not in WEIGHT_CASES:
        print(f"  Case {case_number} not found. Choose 1–3.")
        return

    case = WEIGHT_CASES[case_number]
    print(f"\nCase {case_number}: {case['description']}")

    correct = answer.lower().strip() == case["answer"]
    _print_result(
        correct,
        f"Explanation: {case['explanation']}",
        f"The correct answer is '{case['answer']}'. {case['explanation']}",
    )


# ---------------------------------------------------------------------------
# Exercise 5: Coding exercise — spot the bug
# ---------------------------------------------------------------------------

SC_BUGS = {
    1: {
        "description": "What is wrong with this donor pool construction?",
        "code": textwrap.dedent("""
            # Select donor states for a California tobacco analysis
            all_states = list(state_data.keys())

            # Remove California (treated)
            donor_pool = [s for s in all_states if s != \"California\"]

            # Fit synthetic control using all other states as donors
            sc = cp.SyntheticControl(
                data=panel_df,
                treatment_time=1989,
                formula=\"cigsale ~ 1\",
                group_variable_name=\"state\",
                treated_group=\"California\",
                model=...,
            )
        """),
        "bug": "includes_treated_donors",
        "fix": (
            "Several states (e.g., Massachusetts, Michigan) implemented similar tobacco control\n"
            "programs during the study period. Including them in the donor pool contaminates\n"
            "the counterfactual. Exclude any state that received a similar intervention:\n"
            "donor_pool = [s for s in all_states if s not in ['California', 'MA', 'MI', ...]]"
        ),
        "explanation": (
            "The donor pool should exclude any unit that received the same or a similar treatment\n"
            "during the study period. If a donor state also implemented a tobacco tax, its post-1989\n"
            "trajectory will reflect its own treatment effect, not the pure untreated counterfactual.\n"
            "This contaminates the synthetic California and biases the causal estimate toward zero."
        ),
    },
    2: {
        "description": "What is wrong with this placebo p-value interpretation?",
        "code": textwrap.dedent("""
            # Results from placebo test
            n_units = 5  # 1 treated + 4 donors
            ca_rank = 1  # California has the largest RMSPE ratio

            p_value = 1 / n_units  # = 0.20

            if p_value < 0.05:
                print(\"Significant at 5% level\")
            else:
                print(\"NOT significant — the tobacco tax had no effect\")
        """),
        "bug": "incorrect_null_interpretation",
        "fix": (
            "Change the else branch to: print('P-value = 0.20, not significant at 5%.'\n"
            "' This reflects the small donor pool (4 donors), not absence of effect.')\n"
            "With only 4 donors, 5% significance is impossible. The minimum achievable\n"
            "p-value is 1/5 = 0.20. Report the rank and explain the power limitation."
        ),
        "explanation": (
            "Failure to achieve p < 0.05 does NOT mean the treatment had no effect — it may\n"
            "mean the donor pool is too small to detect the effect. With 4 donors, the minimum\n"
            "achievable p-value is 0.20. Always state the minimum achievable p-value alongside\n"
            "the actual p-value so the reader understands the power constraint."
        ),
    },
    3: {
        "description": "What is the statistical error in this SC confidence set construction?",
        "code": textwrap.dedent("""
            # Bootstrap 95% CI for the cumulative SC effect
            bootstrap_effects = []
            for _ in range(500):
                boot_idx = np.random.choice(n_post, size=n_post, replace=True)
                boot_gap = gap_post[boot_idx]
                bootstrap_effects.append(boot_gap.sum())

            lower, upper = np.percentile(bootstrap_effects, [2.5, 97.5])
            print(f\"95% CI: [{lower:.1f}, {upper:.1f}]\")
        """),
        "bug": "wrong_bootstrap_approach",
        "fix": (
            "The bootstrap should resample over units (donors), not over time periods.\n"
            "The valid inference approach for SC is permutation tests, not time-period bootstrap.\n"
            "Use test inversion (permutation confidence sets) or the Bayesian posterior HDI\n"
            "from cp.SyntheticControl for valid uncertainty quantification."
        ),
        "explanation": (
            "Bootstrapping over time periods treats each post-period observation as an independent\n"
            "draw, which ignores the time-series structure (autocorrelation). This produces\n"
            "artificially narrow CIs. The valid approaches are: (1) permutation test inversion\n"
            "for a confidence set, or (2) Bayesian posterior HDI from CausalPy's Bayesian SC,\n"
            "which properly accounts for weight uncertainty and temporal structure."
        ),
    },
}


def check_sc_bug(snippet_number: int, your_diagnosis: str) -> None:
    """
    Identify the bug in a synthetic control code snippet.

    Parameters
    ----------
    snippet_number : int
        1 to 3
    your_diagnosis : str
        One of: 'includes_treated_donors', 'incorrect_null_interpretation',
                'wrong_bootstrap_approach', 'no_bug', 'wrong_optimization'

    Examples
    --------
    >>> check_sc_bug(1, 'includes_treated_donors')     # correct
    >>> check_sc_bug(2, 'incorrect_null_interpretation')  # correct
    """
    if snippet_number not in SC_BUGS:
        print(f"  Snippet {snippet_number} not found. Choose 1–3.")
        return

    snippet = SC_BUGS[snippet_number]
    print(f"\nSnippet {snippet_number}: {snippet['description']}")
    print(snippet["code"])

    correct = your_diagnosis.lower().strip() == snippet["bug"]
    if correct:
        print(f"  CORRECT. Bug: {snippet['bug']}")
        print(f"  Fix: {snippet['fix']}")
        print(f"  Explanation: {snippet['explanation']}")
    else:
        print(f"  INCORRECT. The bug is: '{snippet['bug']}'")
        print(f"  Fix: {snippet['fix']}")
        print(f"  Explanation: {snippet['explanation']}")


# ---------------------------------------------------------------------------
# Exercise 6: Hands-on optimization verification
# ---------------------------------------------------------------------------

def verify_weight_optimization() -> None:
    """
    Verify that you understand the weight optimization by manually checking
    that the optimal weights minimize the pre-period MSE.
    """
    print("\n--- Exercise 6: Weight Optimization Verification ---\n")

    np.random.seed(7)
    n_pre = 15
    n_donors = 4

    # True weights: unit 0 dominates
    true_w = np.array([0.6, 0.3, 0.1, 0.0])
    Y_donors = np.random.normal(50, 5, (n_pre, n_donors))
    y_treated = Y_donors @ true_w + np.random.normal(0, 0.5, n_pre)

    print(f"Pre-period data: {n_pre} periods, {n_donors} donors")
    print(f"True weights: {true_w}")

    # Test three candidate weight vectors
    candidates = {
        "Equal weights": np.array([0.25, 0.25, 0.25, 0.25]),
        "True weights":  true_w,
        "Wrong weights": np.array([0.0, 0.0, 0.5, 0.5]),
    }

    print()
    print(f"{'Weight vector':<20} {'Pre-period MSE':>18}")
    print("-" * 42)
    for name, w in candidates.items():
        y_syn = Y_donors @ w
        mse = np.mean((y_treated - y_syn) ** 2)
        print(f"  {name:<18}  {mse:>16.4f}")

    print()
    print("Observation: The true weights should give the lowest MSE.")
    print("This is exactly what the scipy optimizer finds.")
    print()

    # Quick verification with scipy
    from scipy.optimize import minimize as sp_minimize
    w0 = np.ones(n_donors) / n_donors
    result = sp_minimize(
        lambda w: np.mean((y_treated - Y_donors @ w) ** 2),
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    print(f"Scipy optimal weights: {result.x.round(3)}")
    print(f"Scipy optimal MSE:     {result.fun:.4f}")
    print()
    print("The optimizer should find weights close to [0.60, 0.30, 0.10, 0.00].")


# ---------------------------------------------------------------------------
# Main: run all exercises
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Module 03 Self-Check: Synthetic Control Methods")
    print("=" * 70)
    print()
    print("Sample answers — replace with your own to test your understanding.")
    print()

    print("--- Method choice: ITS vs Synthetic Control ---")
    check_method_choice(1, "its")
    check_method_choice(2, "synthetic_control")

    print()
    print("--- Pre-period fit interpretation ---")
    check_prefit_interpretation(1, "poor")
    check_prefit_interpretation(2, "good")
    check_prefit_interpretation(3, "exclude")

    print()
    print("--- Permutation p-value ---")
    check_pvalue_computation(1, 1 / 9)
    check_pvalue_computation(2, 3 / 39)

    print()
    print("--- Weight interpretation ---")
    check_weight_interpretation(1, "not_contributing")
    check_weight_interpretation(2, "major_donor")
    check_weight_interpretation(3, "concern")

    print()
    print("--- Code bugs ---")
    check_sc_bug(1, "includes_treated_donors")
    check_sc_bug(2, "incorrect_null_interpretation")

    print()
    verify_weight_optimization()
