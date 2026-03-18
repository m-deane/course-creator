"""
Module 00 — Exercise 2: DAG Reasoning Self-Check
=================================================

These exercises test your ability to:
1. Identify confounders, mediators, and colliders in a DAG
2. Determine the correct adjustment set for causal identification
3. Spot collider bias when conditioning on the wrong variables
4. Draw and reason about DAGs for ITS scenarios

No grading — these are for your own understanding.
Run this file directly to see the answers and explanations.
"""

from typing import List, Set, Dict, Tuple
import textwrap


# ---------------------------------------------------------------------------
# PART 1: Identify Node Type in a DAG
# ---------------------------------------------------------------------------
# Given a DAG described in text, classify each node as:
# - Confounder (C): causes both treatment and outcome
# - Mediator (M): on the causal path between treatment and outcome
# - Collider (L): caused by both treatment and outcome
# - Instrumental Variable (IV): causes treatment but not outcome directly
# - Pure covariate (X): affects outcome but not treatment


DAGS = {
    1: {
        "description": (
            "Education (T) affects Earnings (Y). "
            "Family Wealth (W) affects both Education and Earnings. "
            "Job Skills (S) is caused by Education and also causes Earnings."
        ),
        "nodes": {
            "Family Wealth (W)": "Confounder",
            "Job Skills (S)": "Mediator",
        },
        "explanation": {
            "Family Wealth (W)": (
                "Confounder — Family wealth causes people to get more education (can afford tuition) "
                "AND also directly causes higher earnings (inheritance, connections). "
                "Failing to control for family wealth will overestimate education's causal effect on earnings."
            ),
            "Job Skills (S)": (
                "Mediator — Education causes skill acquisition, and skills cause higher earnings. "
                "Job Skills is on the causal pathway from education to earnings. "
                "Controlling for skills would block the mechanism and underestimate the total effect of education."
            ),
        },
    },
    2: {
        "description": (
            "Smoking (T) causes Lung Cancer (Y). "
            "Genetic Variant (G) causes Smoking tendency. "
            "Hospitalization (H) is caused by both Smoking and Lung Cancer. "
            "Age (A) causes Smoking."
        ),
        "nodes": {
            "Genetic Variant (G)": "Instrumental Variable",
            "Hospitalization (H)": "Collider",
            "Age (A)": "Confounder",
        },
        "explanation": {
            "Genetic Variant (G)": (
                "Instrumental Variable — The genetic variant affects smoking tendency but has no direct "
                "path to lung cancer (other than through smoking). "
                "This is Mendelian randomization — a real technique that uses genetic variants as IVs "
                "to study causal effects of lifestyle factors."
            ),
            "Hospitalization (H)": (
                "Collider — Both smoking and lung cancer independently cause hospitalization. "
                "Conditioning on hospitalized patients (e.g., studying only hospital patients) "
                "creates a spurious negative correlation between smoking and other causes of "
                "hospitalization. This is Berkson's paradox."
            ),
            "Age (A)": (
                "Confounder — Older people both smoke more (historical patterns) and develop "
                "cancer more. Age is a backdoor path: Smoking ← Age → Cancer. "
                "Must control for age when estimating the smoking → cancer effect."
            ),
        },
    },
    3: {
        "description": (
            "Job Training Program (T) affects Employment (Y). "
            "Unemployment Duration (D) affects selection into training AND affects employment. "
            "Interview Performance (I) is caused by job training AND by unobserved ability. "
            "Unobserved Ability (A) affects both Interview Performance and Employment."
        ),
        "nodes": {
            "Unemployment Duration (D)": "Confounder",
            "Interview Performance (I)": "Collider",
        },
        "explanation": {
            "Unemployment Duration (D)": (
                "Confounder — People unemployed for longer are more likely to seek training "
                "AND less likely to find employment (scarring effects). "
                "Backdoor path: Training ← Duration → Employment. Must control for duration."
            ),
            "Interview Performance (I)": (
                "Collider — Job training improves interview skills, AND unobserved ability "
                "affects performance. Both T and unobserved A point into I. "
                "Conditioning on interview performance would open a path from training to "
                "ability, creating collider bias. Do NOT include interview performance as a control."
            ),
        },
    },
}


def check_node_type(dag_number: int, node_name: str, your_classification: str) -> None:
    """
    Check your classification of a node in a DAG.

    Parameters
    ----------
    dag_number : int
        DAG scenario number (1-3)
    node_name : str
        The name of the node (must match the key in DAGS)
    your_classification : str
        Your answer: "Confounder", "Mediator", "Collider", "IV", or "Covariate"
    """
    if dag_number not in DAGS:
        print(f"DAG {dag_number} does not exist.")
        return

    dag = DAGS[dag_number]
    print(f"DAG {dag_number}: {dag['description']}")
    print()

    # Find the node (case-insensitive partial match)
    matched_node = None
    for n in dag["nodes"]:
        if node_name.lower() in n.lower():
            matched_node = n
            break

    if matched_node is None:
        print(f"Node '{node_name}' not found in DAG {dag_number}.")
        print(f"Available nodes: {list(dag['nodes'].keys())}")
        return

    correct = dag["nodes"][matched_node]
    explanation = dag["explanation"][matched_node]

    if your_classification.strip().lower() == correct.lower():
        print(f"Correct! {matched_node} is a {correct}.")
    else:
        print(f"Not quite. You said {your_classification}, correct is {correct}.")

    print(f"\nExplanation: {explanation}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# PART 2: Adjustment Sets
# ---------------------------------------------------------------------------
# For each DAG, determine the correct minimal adjustment set to identify
# the causal effect of Treatment on Outcome.

ADJUSTMENT_SCENARIOS = {
    1: {
        "description": (
            "Nodes: Age (A), Exercise (T), Health (Y), BMI (B), Diet (D)\n"
            "Edges:\n"
            "  A -> T (older people exercise less)\n"
            "  A -> Y (age affects health)\n"
            "  T -> B (exercise reduces BMI)\n"
            "  B -> Y (BMI affects health)\n"
            "  D -> T (healthier diet correlates with exercise)\n"
            "  D -> Y (diet affects health)\n"
            "  T -> Y (direct effect of exercise on health)"
        ),
        "question": "What is the minimal adjustment set to identify the TOTAL effect of Exercise on Health?",
        "correct_adjustments": {"Age", "Diet"},
        "wrong_adjustments_explanation": {
            "BMI": (
                "BMI is a MEDIATOR — exercise reduces BMI, which reduces disease risk. "
                "Controlling for BMI blocks this pathway and gives only the direct effect "
                "of exercise (not through BMI), underestimating the total effect."
            ),
        },
        "explanation": (
            "Adjust for Age and Diet.\n"
            "- Age is a confounder: backdoor path Exercise ← Age → Health.\n"
            "- Diet is a confounder: backdoor path Exercise ← Diet → Health.\n"
            "- BMI is a mediator on the front-door path Exercise → BMI → Health.\n"
            "  Do NOT adjust for BMI if you want the total causal effect.\n"
            "- No other backdoor paths exist after adjusting for Age and Diet."
        ),
    },
    2: {
        "description": (
            "ITS scenario: Policy Intervention (T) affects Crime Rate (Y).\n"
            "Nodes: Time (time), Economic Conditions (E), Police Staffing (P), "
            "Crime Rate (Y), Intervention (T), Demographic Change (D)\n"
            "Edges:\n"
            "  time -> T (intervention occurs at t*)\n"
            "  time -> Y (secular trend in crime)\n"
            "  E -> Y (economy affects crime)\n"
            "  E -> T (economic crisis might trigger policy)\n"
            "  P -> Y (more police reduces crime)\n"
            "  T -> P (intervention includes more police)\n"
            "  D -> Y (demographics affect crime)\n"
            "  T -> Y (direct effect of intervention)"
        ),
        "question": (
            "What should be included in the ITS model to identify the effect of "
            "the intervention on crime rate? What should NOT be included?"
        ),
        "correct_adjustments": {"Time trend", "Economic Conditions", "Demographic Change"},
        "wrong_adjustments_explanation": {
            "Police Staffing": (
                "Police Staffing is a MEDIATOR — the intervention increases police staffing, "
                "which reduces crime. Controlling for police staffing would block this "
                "mechanism and give only the direct effect of the intervention not through policing."
            ),
        },
        "explanation": (
            "Include: Time trend, Economic Conditions, Demographic Change.\n"
            "- Time trend: controls for the secular trend in crime (standard in all ITS models)\n"
            "- Economic Conditions: confounder if the intervention was triggered by an economic crisis\n"
            "- Demographic Change: confounder that affects crime independently\n\n"
            "Do NOT include: Police Staffing (mediator — part of the intervention mechanism).\n"
            "If you control for police staffing, you lose the policing mechanism and "
            "underestimate the total intervention effect."
        ),
    },
}


def check_adjustment_set(
    scenario_number: int,
    your_adjustments: List[str],
) -> None:
    """
    Check your proposed adjustment set.

    Parameters
    ----------
    scenario_number : int
        Scenario number (1-2)
    your_adjustments : list of str
        Variables you would adjust for
    """
    if scenario_number not in ADJUSTMENT_SCENARIOS:
        print(f"Scenario {scenario_number} does not exist.")
        return

    s = ADJUSTMENT_SCENARIOS[scenario_number]
    correct = s["correct_adjustments"]
    your_set = {v.strip() for v in your_adjustments}

    print(f"Scenario {scenario_number}:")
    print(s["description"])
    print(f"\nQuestion: {s['question']}")
    print()
    print(f"Your adjustments:   {sorted(your_set)}")
    print(f"Correct adjustments: {sorted(correct)}")
    print()

    missing = correct - your_set
    extra = your_set - correct

    if missing:
        print(f"Missing: {missing}")
        print("  These are confounders creating backdoor paths you left open.")

    for var, explanation in s["wrong_adjustments_explanation"].items():
        if var in your_set:
            print(f"\nWARNING: You included '{var}'")
            print(f"  {explanation}")

    if not missing and not extra:
        print("Correct adjustment set!")

    print(f"\nFull explanation:\n{s['explanation']}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# PART 3: Collider Bias Detection
# ---------------------------------------------------------------------------

COLLIDER_SCENARIOS = {
    1: {
        "setup": (
            "A researcher studies the relationship between talent and effort among "
            "Nobel Prize winners. They find that within this group, talent and effort "
            "are negatively correlated: the most talented winners did the least work, "
            "and the hardest-working winners showed moderate talent."
        ),
        "question": "Is this a real negative relationship between talent and effort?",
        "answer": "No — collider bias",
        "explanation": (
            "Collider bias. Nobel Prizes are caused by both talent and effort. "
            "Conditioning on Nobel Prize winners (collider) opens a path between "
            "talent and effort that does not exist in the general population.\n\n"
            "In the general population, talent and effort may be uncorrelated or even "
            "positively correlated (both reflect motivation and circumstance).\n\n"
            "Within Nobel winners, high talent + low effort can achieve the prize, "
            "and high effort + moderate talent can also achieve it — creating the illusion "
            "of a trade-off."
        ),
    },
    2: {
        "setup": (
            "A hospital researcher conditions their study on patients who were admitted "
            "to the ICU. They find that patients with respiratory disease have better "
            "survival rates than patients with other conditions in the ICU."
        ),
        "question": (
            "Should the researcher conclude that respiratory disease protects survival "
            "in the ICU compared to other conditions?"
        ),
        "answer": "No — likely collider bias (and selection into ICU)",
        "explanation": (
            "ICU admission is a collider: it is caused by disease severity AND by the "
            "specific disease type. Respiratory disease patients often reach the ICU "
            "at earlier stages (before other conditions become severe enough for ICU admission). "
            "Conditioning on ICU admission (collider) creates a spurious favorable comparison "
            "for respiratory disease patients, who are systematically less severe within ICU than "
            "patients with other conditions who only arrive in ICU at extreme severity."
        ),
    },
}


def check_collider_scenario(scenario_number: int) -> None:
    """
    Print the collider scenario, question, and full explanation.

    Parameters
    ----------
    scenario_number : int
        Scenario number (1-2)
    """
    if scenario_number not in COLLIDER_SCENARIOS:
        print(f"Scenario {scenario_number} does not exist.")
        return

    s = COLLIDER_SCENARIOS[scenario_number]
    print(f"Scenario {scenario_number}:")
    print(textwrap.fill(s["setup"], 70))
    print(f"\nQuestion: {s['question']}")
    print(f"\nAnswer: {s['answer']}")
    print(f"\nExplanation:\n{textwrap.fill(s['explanation'], 70)}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 00 — DAG Reasoning Self-Check Exercises")
    print("=" * 60)
    print()

    print("PART 1: Identify Node Types")
    print("-" * 40)
    check_node_type(1, "Family Wealth", "Confounder")
    print()
    check_node_type(2, "Hospitalization", "Mediator")  # Wrong — see feedback
    print()

    print("PART 2: Adjustment Sets")
    print("-" * 40)
    check_adjustment_set(1, ["Age", "Diet", "BMI"])  # BMI is wrong
    print()
    check_adjustment_set(2, ["Time trend", "Economic Conditions", "Demographic Change"])  # Correct
    print()

    print("PART 3: Collider Bias Detection")
    print("-" * 40)
    check_collider_scenario(1)
    print()
    check_collider_scenario(2)
