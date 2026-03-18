"""
Module 00 — Exercise 1: Causal Thinking Self-Check
===================================================

These exercises test your ability to distinguish causal from correlational
claims, identify confounders, and recognize when causal inference is needed.

Instructions
------------
Work through each problem. For the multiple-choice questions, check your
answer by calling the corresponding check function. For written exercises,
compare your reasoning to the provided discussion.

No grading — these are for your own understanding.
"""

# ---------------------------------------------------------------------------
# PART 1: Causal vs Correlational Claims
# ---------------------------------------------------------------------------
# For each claim below, decide:
# (A) Causal — this claim is about the effect of an intervention
# (B) Predictive / correlational — this claim is about association only
#
# After deciding, call check_claim(question_number, your_answer) to verify.

CLAIMS = {
    1: "People who drink more coffee have lower rates of Parkinson's disease.",
    2: "If we increase the minimum wage, unemployment will rise.",
    3: "Regions with more police officers have higher crime rates.",
    4: "Students who attend tutoring sessions score higher on exams.",
    5: "Hospitals that perform more surgeries have higher in-hospital mortality.",
    6: "Countries that banned leaded gasoline saw drops in violent crime 20 years later.",
    7: "Zip codes with more Starbucks locations have higher average income.",
    8: "Children who read more books have larger vocabularies.",
    9: "If we reduce carbon emissions, global average temperature will stabilize.",
    10: "Firms that adopt remote work policies have lower employee turnover.",
}

ANSWERS = {
    1: "B",  # Correlational: confounders (diet, genetics, lifestyle) drive both coffee consumption and Parkinson's risk
    2: "A",  # Causal: asking about the effect of an intervention (raising the minimum wage)
    3: "B",  # Correlational: reverse causation — high-crime areas get more police deployed, not the other way around
    4: "B",  # Correlational: selection bias — motivated students self-select into tutoring
    5: "B",  # Correlational: sicker patients go to specialized hospitals (confounding by severity)
    6: "A",  # Causal: claims the ban caused the later crime drop (the 20-year lag is the developmental mechanism)
    7: "B",  # Correlational: Starbucks locates in wealthy areas, not the other way around
    8: "B",  # Correlational: higher-SES families both buy more books and develop larger vocabularies (confounder)
    9: "A",  # Causal: asking about the effect of an emissions intervention
    10: "B",  # Correlational: selection bias — employee-friendly firms both adopt remote work AND have lower turnover
}

EXPLANATIONS = {
    1: ("B — Correlational. Coffee consumption correlates with many lifestyle factors. "
        "A causal claim would be: 'Drinking coffee reduces Parkinson's risk.' "
        "The correlational version just describes the association without claiming direction."),
    2: ("A — Causal. 'If we increase' is the intervention language. "
        "This is asking what would happen if we set the minimum wage to a higher value."),
    3: ("B — Correlational, with reverse causation. High-crime areas get more police. "
        "The causal arrow runs from crime to police deployment, not vice versa. "
        "A naive predictive model would wrongly suggest deploying fewer police reduces crime."),
    4: ("B — Correlational. Self-selection: students who attend tutoring are more motivated "
        "and would score higher regardless. A causal claim would require random assignment to tutoring."),
    5: ("B — Correlational. This is confounding by illness severity (a collider/confounder). "
        "Sicker patients are referred to high-volume specialized hospitals. "
        "The higher mortality reflects patient severity, not hospital quality."),
    6: ("A — Causal. The claim asserts that the policy intervention (ban) caused the later reduction. "
        "This is the classic 'lead-crime hypothesis' studied by Rick Nevin and others. "
        "Note: the causal mechanism (lead exposure affecting developing brains) justifies the lag."),
    7: ("B — Correlational. Starbucks selects locations based on existing wealth. "
        "Adding a Starbucks does not cause income to rise. Common cause: neighborhood wealth."),
    8: ("B — Correlational. Socioeconomic status affects both book access and vocabulary development. "
        "A causal study would randomly assign books to children (some RCTs have done this)."),
    9: ("A — Causal. 'If we reduce emissions' is an intervention. "
        "Asking about the effect of a human action on global temperature is causal inference."),
    10: ("B — Correlational. Companies with better cultures both adopt flexible policies "
         "AND retain employees. Selection bias: good employers do many good things simultaneously."),
}


def check_claim(question_number: int, your_answer: str) -> None:
    """
    Check your answer for a causal/correlational identification question.

    Parameters
    ----------
    question_number : int
        Question number (1-10)
    your_answer : str
        Your answer: "A" for causal, "B" for correlational/predictive
    """
    if question_number not in ANSWERS:
        print(f"Question {question_number} does not exist. Choose from 1-10.")
        return

    your_answer = your_answer.strip().upper()
    correct = ANSWERS[question_number]
    claim = CLAIMS[question_number]
    explanation = EXPLANATIONS[question_number]

    print(f"Question {question_number}: \"{claim}\"")
    print()

    if your_answer == correct:
        print(f"Correct! Answer: ({correct})")
    else:
        print(f"Not quite. You answered ({your_answer}), correct is ({correct})")

    print(f"\nExplanation: {explanation}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# PART 2: Identifying the Estimand
# ---------------------------------------------------------------------------
# For each research question below, identify which causal estimand is most
# appropriate: ATE, ATT, or ATU (Average Treatment Effect on Untreated)

ESTIMAND_QUESTIONS = {
    1: {
        "question": ("A city implements a new job training program for unemployed residents. "
                     "The city council wants to know: did the program help the people who participated?"),
        "answer": "ATT",
        "explanation": (
            "ATT — The question asks about the effect on those who participated (the treated). "
            "The council wants to evaluate the program as applied. "
            "The ATE would require speculating about what would happen if non-participants "
            "(who are employed) were forced to take the training."
        ),
    },
    2: {
        "question": ("A pharmaceutical company wants to know the average effect of a new "
                     "antidepressant on depression symptoms across all patients who might receive it."),
        "answer": "ATE",
        "explanation": (
            "ATE — The company wants the population average effect for all patients who could "
            "potentially receive the drug (not just those who happened to participate in the trial). "
            "This is the standard FDA approval estimand."
        ),
    },
    3: {
        "question": ("A school district implemented a reading intervention in struggling schools. "
                     "A policy analyst wants to know: would expanding the program to all schools help?"),
        "answer": "ATE",
        "explanation": (
            "ATE (or a related quantity) — Expanding to all schools means giving the treatment "
            "to currently untreated (non-struggling) schools. The ATT only tells us about effects "
            "in the original struggling schools. The analyst needs to extrapolate to a broader "
            "population, which requires ATE or a specific subgroup estimand."
        ),
    },
    4: {
        "question": ("An ITS study finds that a pollution regulation reduced hospital admissions "
                     "in the cities where it was implemented. The analyst reports the effect."),
        "answer": "ATT",
        "explanation": (
            "ATT — ITS estimates the effect on the treated units (the cities that got the regulation). "
            "We observe what happened to those cities with and without treatment in the counterfactual sense. "
            "Whether this effect would generalize to other cities (ATE) is a separate question."
        ),
    },
}


def check_estimand(question_number: int, your_answer: str) -> None:
    """
    Check your estimand identification.

    Parameters
    ----------
    question_number : int
        Question number (1-4)
    your_answer : str
        Your answer: "ATE", "ATT", or "ATU"
    """
    if question_number not in ESTIMAND_QUESTIONS:
        print(f"Question {question_number} does not exist. Choose from 1-4.")
        return

    q = ESTIMAND_QUESTIONS[question_number]
    your_answer = your_answer.strip().upper()
    correct = q["answer"]

    print(f"Question {question_number}: {q['question']}")
    print()

    if your_answer == correct:
        print(f"Correct! Estimand: {correct}")
    else:
        print(f"Not quite. You answered {your_answer}, correct is {correct}")

    print(f"\nExplanation: {q['explanation']}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# PART 3: Identifying Threats to Causal Validity
# ---------------------------------------------------------------------------
# For each ITS scenario, identify the primary threat to causal validity.

ITS_SCENARIOS = {
    1: {
        "scenario": (
            "A city implements a public transit fare reduction in January 2020. "
            "An ITS analysis shows ridership jumped dramatically after January 2020. "
            "However, January 2020 was also when a new stadium opened in the city center."
        ),
        "threat": "Concurrent event",
        "explanation": (
            "Concurrent event threat: The new stadium opening at the same time as the fare reduction "
            "makes it impossible to attribute the ridership increase to the fare reduction alone. "
            "The DAG would show: Stadium Opening → Ridership, at the same time as Fare Reduction → Ridership. "
            "Fix: Use a placebo test on ridership to non-stadium routes."
        ),
    },
    2: {
        "scenario": (
            "A government announces a major subsidy for electric vehicles in Q1. "
            "The subsidy goes into effect in Q3. An ITS analysis uses Q3 as the intervention date. "
            "The data shows a gradual pre-trend change starting in Q1."
        ),
        "threat": "Anticipation effect",
        "explanation": (
            "Anticipation effect: Buyers adjusted their purchase timing in response to the announcement "
            "(Q1), creating a pre-intervention trend break before the formal implementation (Q3). "
            "The pre-period is contaminated — the ITS assumption that the pre-trend reflects the "
            "counterfactual is violated. "
            "Fix: Use Q1 as the intervention date, or model the announcement and implementation separately."
        ),
    },
    3: {
        "scenario": (
            "An HR department notices that overtime hours spiked in March 2023. "
            "They immediately implement a mandatory overtime cap in April 2023. "
            "An ITS analysis shows overtime hours fell sharply after April 2023."
        ),
        "threat": "Regression to the mean",
        "explanation": (
            "Regression to the mean: The intervention was triggered by an extreme value (the spike). "
            "Extreme values naturally revert toward the mean over time regardless of any intervention. "
            "Part of the observed decline is mechanical reversion, not the effect of the overtime cap. "
            "Fix: Compare the post-intervention trend to a control group with similar pre-period spikes "
            "but no intervention."
        ),
    },
}


def check_its_threat(scenario_number: int, your_answer: str) -> None:
    """
    Check your threat identification for an ITS scenario.

    Parameters
    ----------
    scenario_number : int
        Scenario number (1-3)
    your_answer : str
        Your identified threat (free-text — compared for key terms)
    """
    if scenario_number not in ITS_SCENARIOS:
        print(f"Scenario {scenario_number} does not exist. Choose from 1-3.")
        return

    s = ITS_SCENARIOS[scenario_number]

    print(f"Scenario {scenario_number}: {s['scenario']}")
    print()
    print(f"Primary threat: {s['threat']}")
    print(f"\nExplanation: {s['explanation']}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# USAGE EXAMPLE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 00 — Causal Thinking Self-Check Exercises")
    print("=" * 60)
    print()
    print("PART 1: Causal vs Correlational")
    print("Call check_claim(n, 'A') or check_claim(n, 'B') for questions 1-10")
    print()

    # Example: Try question 3 (police officers and crime)
    check_claim(3, "A")  # Wrong answer — see what the feedback says
    print()
    check_claim(3, "B")  # Correct answer
    print()

    print("PART 2: Identifying the Estimand")
    print("Call check_estimand(n, 'ATE') or check_estimand(n, 'ATT') for questions 1-4")
    print()

    # Example
    check_estimand(1, "ATT")
    print()

    print("PART 3: ITS Threats to Validity")
    print("Call check_its_threat(n, 'your answer') for scenarios 1-3")
    print()

    # Example
    check_its_threat(2, "anticipation")
