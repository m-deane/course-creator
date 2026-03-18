"""
Module 01 — Exercise 1: ITS Self-Check
=======================================

These exercises test your understanding of:
1. Identifying when ITS is appropriate vs. inappropriate
2. Interpreting ITS model output (level change, slope change, counterfactual)
3. Recognizing validity threats in ITS designs
4. Building correct ITS design matrices

No grading — these are for your own understanding.
"""

import numpy as np
import pandas as pd
from typing import List


# ---------------------------------------------------------------------------
# PART 1: Is ITS Appropriate?
# ---------------------------------------------------------------------------
# For each scenario, decide whether ITS is appropriate (Yes/No) and explain.

SCENARIOS = {
    1: {
        "description": (
            "A food safety agency wants to evaluate whether a new restaurant grading "
            "system (A/B/C letter grades posted in windows) reduced foodborne illness cases. "
            "The grading system was rolled out city-wide in January 2019. Monthly foodborne "
            "illness data is available from 2015 to 2022. The policy was announced by "
            "the mayor in November 2018 and implemented in January 2019."
        ),
        "appropriate": True,
        "caveats": [
            "Anticipation effect risk: Some restaurants may have improved practices "
            "between November 2018 and January 2019 (pre-announcement preparation). "
            "Check for a pre-trend break in November 2018.",
            "ITS is appropriate: 4 years of pre-data, clear intervention date, "
            "exogenous timing (legislative process, not driven by foodborne illness trend).",
        ],
    },
    2: {
        "description": (
            "A hospital wants to evaluate whether a new hand hygiene protocol reduced "
            "hospital-acquired infections. The protocol was introduced in March 2020, "
            "the same month that COVID-19 protocols also began requiring masks and "
            "enhanced cleaning in all patient areas."
        ),
        "appropriate": False,
        "caveats": [
            "Concurrent event problem: COVID-19 protocols (masking, enhanced cleaning) "
            "were implemented simultaneously in March 2020. Any reduction in hospital-acquired "
            "infections cannot be attributed to the hand hygiene protocol alone.",
            "This is a fundamental threat to ITS validity. The design cannot separate "
            "the two interventions without additional data or comparison groups.",
        ],
    },
    3: {
        "description": (
            "A city's transit authority had ridership data for 6 months before implementing "
            "free weekend bus service. They want to use ITS to estimate the effect on ridership."
        ),
        "appropriate": False,
        "caveats": [
            "Insufficient pre-period data: 6 months is too few to reliably estimate "
            "the pre-intervention trend. The minimum recommendation is 12 months; "
            "ideally 24+ months.",
            "With only 6 months, the confidence intervals will be very wide and the "
            "trend estimate will be highly unstable.",
        ],
    },
    4: {
        "description": (
            "A manufacturing company's safety officer noticed that workplace accidents "
            "had been rising for 6 months. In response, they implemented a new safety "
            "training program. Monthly accident data is available for 3 years before "
            "and 2 years after."
        ),
        "appropriate": "Caution",
        "caveats": [
            "Regression to the mean risk: The training was triggered by a rising trend "
            "(endogenous timing). Even without the program, accidents might have reverted "
            "toward their mean. Any observed decline post-intervention may be partly or "
            "entirely due to regression to the mean.",
            "ITS may still provide useful evidence if: (1) a comparison group without "
            "the training can be identified, or (2) the pre-trend analysis shows the "
            "rising trend was part of a longer structural increase (not a random spike).",
        ],
    },
    5: {
        "description": (
            "A government wants to evaluate whether a 10-cent tax on sugary beverages "
            "reduced purchases. Quarterly retail sales data for taxed and untaxed beverages "
            "is available for 5 years before and 3 years after the tax implementation. "
            "The tax was legislated 18 months before implementation."
        ),
        "appropriate": True,
        "caveats": [
            "Strong design: 5 years pre-data (20 quarters), legislative timing (exogenous), "
            "comparison series available (untaxed beverages as control).",
            "Anticipation effect: 18-month advance notice may have caused stockpiling "
            "in the quarters before implementation. Check for pre-trend change near "
            "the announcement date.",
            "Consider using untaxed beverages as a control series (Controlled ITS / DiD-ITS).",
        ],
    },
}


def check_its_appropriateness(scenario_number: int, your_answer: str) -> None:
    """
    Evaluate your assessment of whether ITS is appropriate for a scenario.

    Parameters
    ----------
    scenario_number : int
        Scenario number (1-5)
    your_answer : str
        "Yes", "No", or "Caution"
    """
    if scenario_number not in SCENARIOS:
        print(f"Scenario {scenario_number} does not exist.")
        return

    s = SCENARIOS[scenario_number]
    correct = str(s["appropriate"])

    print(f"Scenario {scenario_number}:")
    print(s["description"])
    print()
    print(f"Your answer: {your_answer}")
    print(f"Assessment: {s['appropriate']}")
    print()
    print("Key considerations:")
    for i, caveat in enumerate(s["caveats"], 1):
        print(f"  {i}. {caveat}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# PART 2: Interpret ITS Output
# ---------------------------------------------------------------------------

def interpret_its_output():
    """
    Practice interpreting ITS model output.
    Generates and displays a realistic set of posterior estimates.
    """
    print("ITS Model Output Interpretation Exercise")
    print("=" * 60)
    print()
    print("Context: ITS analysis of a speed camera installation program.")
    print("Outcome: Monthly traffic accident deaths per 1,000,000 vehicle-miles")
    print("Data: 36 pre-intervention months, 24 post-intervention months")
    print()
    print("Posterior Estimates (mean and 94% HDI):")
    print("-" * 60)
    print(f"  Intercept (α):       42.3  [38.1, 46.8]")
    print(f"  Pre-trend (β₁):      -0.08 [-0.19, +0.04]")
    print(f"  Level change (β₂):   -4.2  [-6.8, -1.7]   P(<0): 99.2%")
    print(f"  Slope change (β₃):   -0.05 [-0.12, +0.02] P(<0): 87.1%")
    print()
    print("QUESTIONS:")
    print()
    print("Q1. Was the speed camera program associated with an immediate reduction in fatalities?")
    print("    (Check beta_2 and its credible interval)")
    print()
    print("Q2. Did the fatality rate continue to fall faster after installation?")
    print("    (Check beta_3)")
    print()
    print("Q3. The pre-trend beta_1 has a credible interval that includes zero. What does this mean?")
    print()
    print("Q4. Calculate the estimated causal effect at month 12 post-installation.")
    print("    (Effect at month k: beta_2 + beta_3 * k)")
    print()
    print("ANSWERS:")
    print("-" * 60)
    print()
    print("A1. YES — beta_2 = -4.2 with 94% HDI entirely below zero, P(<0) = 99.2%.")
    print("    Strong evidence of an immediate reduction of ~4.2 deaths/million vehicle-miles.")
    print()
    print("A2. WEAKLY — beta_3 = -0.05 but HDI spans zero, P(<0) = 87.1%.")
    print("    There is modest evidence of a continued monthly decline, but the evidence")
    print("    is not as strong as for the immediate effect.")
    print()
    print("A3. The pre-trend was NOT significantly different from zero.")
    print("    There was no strong secular trend in fatalities before the cameras.")
    print("    This is actually reassuring for the ITS validity: the pre-period was stable,")
    print("    making the counterfactual 'stable trajectory' assumption more plausible.")
    print()
    print("A4. Effect at month 12: beta_2 + beta_3 * 12 = -4.2 + (-0.05 * 12) = -4.8")
    print("    The estimated reduction at month 12 is ~4.8 deaths/million vehicle-miles.")
    print("    This is slightly larger than the immediate effect due to the slope change.")


# ---------------------------------------------------------------------------
# PART 3: Build the Design Matrix
# ---------------------------------------------------------------------------

def build_design_matrix_check():
    """
    Exercise: Build the ITS design matrix and verify it is correct.
    """
    print("\nDesign Matrix Exercise")
    print("=" * 60)
    print()
    print("Given: 8 observations, intervention at month 5 (0-indexed)")
    print("Task: Build the design matrix with columns [t, treated, t_post]")
    print()

    # Correct design matrix
    n = 8
    t_star = 5
    t = np.arange(n)
    treated = (t >= t_star).astype(float)
    t_post = np.maximum(t - t_star, 0).astype(float)

    correct_dm = pd.DataFrame({"t": t, "treated": treated, "t_post": t_post})

    print("Correct design matrix:")
    print(correct_dm.to_string(index=False))
    print()

    # Common mistakes to check
    print("COMMON MISTAKES:")
    print()

    # Mistake 1: t_post = 1 at intervention point
    t_post_wrong1 = (t > t_star).astype(float) * (t - t_star)
    dm_wrong1 = pd.DataFrame({"t": t, "treated": treated, "t_post": t_post_wrong1})
    print("Mistake 1: Using t > t_star instead of t >= t_star for t_post")
    print("  (t_post = 1 at intervention point instead of 0)")
    print(dm_wrong1[dm_wrong1["treated"] == 1].to_string(index=False))
    print("  This biases beta_3 by mixing the level change and slope change.")
    print()

    # Mistake 2: treated uses strict inequality
    treated_wrong2 = (t > t_star).astype(float)
    dm_wrong2 = pd.DataFrame({"t": t, "treated": treated_wrong2, "t_post": t_post})
    print("Mistake 2: Using t > t_star instead of t >= t_star for treated")
    print("  (Intervention point has treated=0 instead of treated=1)")
    print(dm_wrong2[dm_wrong2["t"].isin([4, 5, 6])].to_string(index=False))
    print("  This moves the effective intervention one period later than intended.")
    print()


# ---------------------------------------------------------------------------
# PART 4: Identify the Correct Estimand
# ---------------------------------------------------------------------------

ITS_ESTIMAND_QUESTIONS = {
    1: {
        "question": (
            "An ITS analysis estimates the effect of a city-wide smoking ban "
            "on hospital admissions in that city. What estimand is being estimated?"
        ),
        "answer": "ATT",
        "explanation": (
            "ATT (Average Treatment Effect on the Treated). ITS estimates the effect "
            "on the units that actually received the treatment — in this case, the city "
            "where the ban was implemented. The counterfactual is: what would admissions "
            "have been in THIS city without the ban? Not: what would happen if we applied "
            "the ban to all cities in the country (that would be closer to ATE)."
        ),
    },
    2: {
        "question": (
            "A researcher runs ITS with a comparison group (city that did not get the ban). "
            "The researcher computes: [post-period change in treated city] - "
            "[post-period change in control city]. What estimand is this?"
        ),
        "answer": "ATT (in a Diff-in-ITS design)",
        "explanation": (
            "This is still ATT but estimated more credibly using a Difference-in-ITS design. "
            "The control city provides an estimate of the counterfactual trend for the treated city. "
            "By subtracting the control city's trend, we remove time-varying confounders "
            "common to both cities (e.g., flu season, national health trends)."
        ),
    },
}


def check_estimand_question(question_number: int) -> None:
    """Print the question, answer, and explanation."""
    if question_number not in ITS_ESTIMAND_QUESTIONS:
        print(f"Question {question_number} does not exist.")
        return

    q = ITS_ESTIMAND_QUESTIONS[question_number]
    print(f"Question {question_number}: {q['question']}")
    print()
    print(f"Answer: {q['answer']}")
    print()
    print(f"Explanation: {q['explanation']}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 01 — ITS Self-Check Exercises")
    print("=" * 60)
    print()

    print("PART 1: ITS Appropriateness")
    print("-" * 40)
    # Try all scenarios
    for i in [1, 2, 3, 4, 5]:
        answer_map = {1: "Yes", 2: "No", 3: "No", 4: "Caution", 5: "Yes"}
        check_its_appropriateness(i, answer_map[i])
        print()

    print("PART 2: Interpret ITS Output")
    print("-" * 40)
    interpret_its_output()
    print()

    print("PART 3: Design Matrix")
    print("-" * 40)
    build_design_matrix_check()

    print("PART 4: Estimand Identification")
    print("-" * 40)
    for i in [1, 2]:
        check_estimand_question(i)
        print()
