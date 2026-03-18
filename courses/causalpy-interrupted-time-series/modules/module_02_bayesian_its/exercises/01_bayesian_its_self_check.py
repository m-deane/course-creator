"""
Module 02 Self-Check Exercises: Bayesian ITS with PyMC

Run each check function to verify your understanding.
All exercises are self-graded — no submission required.

Usage:
    python 01_bayesian_its_self_check.py
    # or interactively:
    from exercises.01_bayesian_its_self_check import *
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
# Exercise 1: Bayesian p-value interpretation
# ---------------------------------------------------------------------------

BAYESIAN_PVALUE_SCENARIOS = {
    1: {
        "description": (
            "A posterior predictive check for the mean of y returns a Bayesian p-value of 0.48. "
            "What does this indicate?"
        ),
        "options": {
            "a": "The model significantly overestimates the mean",
            "b": "The model captures the mean well",
            "c": "The model significantly underestimates the mean",
            "d": "The prior is too tight",
        },
        "answer": "b",
        "explanation": (
            "A Bayesian p-value near 0.5 means the observed statistic falls near the center of the "
            "posterior predictive distribution. The model reproduces this statistic well. Values near "
            "0 mean the model overestimates (replicated data are almost always higher), values near "
            "1 mean the model underestimates."
        ),
    },
    2: {
        "description": (
            "A lag-1 autocorrelation PPC returns a Bayesian p-value of 0.02. "
            "Which model extension would best address this?"
        ),
        "options": {
            "a": "Add polynomial trend terms",
            "b": "Add an AR(1) error structure",
            "c": "Use a heavier-tailed prior on sigma",
            "d": "Increase the number of MCMC draws",
        },
        "answer": "b",
        "explanation": (
            "A Bayesian p-value near 0 for lag-1 autocorrelation means the observed data have much "
            "more autocorrelation than the model generates. The model's residuals are correlated. "
            "Adding AR(1) errors directly models this correlation structure."
        ),
    },
    3: {
        "description": (
            "The maximum-value PPC returns a Bayesian p-value of 0.97. "
            "Which statement best describes the situation?"
        ),
        "options": {
            "a": "The model correctly captures extreme values",
            "b": "The model generates larger maxima than observed — tail distribution may be too heavy",
            "c": "The model does not generate values as large as observed — tails are too thin",
            "d": "The sampler has not converged",
        },
        "answer": "b",
        "explanation": (
            "Bayesian p-value of 0.97 for the maximum means 97% of replicated datasets have a "
            "larger maximum than observed. The model's likelihood assigns too much probability to "
            "extreme values. This can happen when sigma is overestimated or when the prior on sigma "
            "is too diffuse."
        ),
    },
    4: {
        "description": (
            "You run PPCs on two models. Model A has all Bayesian p-values between 0.1 and 0.9. "
            "Model B has several p-values below 0.05. Which model should you prefer for causal "
            "inference and why?"
        ),
        "options": {
            "a": "Model A, because all statistics are well-captured",
            "b": "Model B, because small p-values indicate a more precise model",
            "c": "Either model — Bayesian p-values do not affect causal estimates",
            "d": "Model B, because a good model should have extreme p-values",
        },
        "answer": "a",
        "explanation": (
            "Model A has all statistics within the central range of the predictive distribution, "
            "meaning the model is well-calibrated across multiple dimensions. Model B's flagged "
            "statistics indicate systematic misspecification that can bias causal estimates — for "
            "example, underestimated variance inflates precision of the causal effect."
        ),
    },
}


def check_bayesian_pvalue(question_number: int, answer: str) -> None:
    """
    Check your answer to a Bayesian p-value interpretation question.

    Parameters
    ----------
    question_number : int
        1 to 4
    answer : str
        Your answer: 'a', 'b', 'c', or 'd'

    Examples
    --------
    >>> check_bayesian_pvalue(1, 'b')   # should be correct
    >>> check_bayesian_pvalue(2, 'a')   # should be incorrect
    """
    if question_number not in BAYESIAN_PVALUE_SCENARIOS:
        print(f"  Question {question_number} not found. Choose 1–4.")
        return

    scenario = BAYESIAN_PVALUE_SCENARIOS[question_number]
    print(f"\nQuestion {question_number}: {scenario['description']}")
    for key, text in scenario["options"].items():
        print(f"  ({key}) {text}")

    correct = answer.lower().strip() == scenario["answer"]
    _print_result(
        correct,
        f"Explanation: {scenario['explanation']}",
        f"The correct answer is ({scenario['answer']}). {scenario['explanation']}",
    )


# ---------------------------------------------------------------------------
# Exercise 2: Prior specification
# ---------------------------------------------------------------------------

PRIOR_SCENARIOS = {
    1: {
        "description": (
            "The pre-intervention outcome has mean 85 and standard deviation 10. "
            "Which prior for the level change (β₂) is most appropriate as a weakly "
            "informative default?"
        ),
        "options": {
            "a": "Normal(0, 0.01)",
            "b": "Normal(0, 10)",
            "c": "Normal(0, 1000)",
            "d": "Normal(-12, 3)",
        },
        "answer": "b",
        "explanation": (
            "Normal(0, σ_Y) = Normal(0, 10) is the weakly informative default. It is centered at "
            "zero (no assumed direction) and spans a range of ±20 units — plausible for most "
            "interventions. Normal(0, 0.01) is too tight and will prevent the model from detecting "
            "the effect. Normal(0, 1000) allows physically impossible effects. Normal(-12, 3) is "
            "informative and requires prior evidence to justify."
        ),
    },
    2: {
        "description": (
            "A prior predictive check shows trajectories frequently going below zero "
            "for an outcome that must be non-negative (e.g., hospital visits). "
            "What should you change?"
        ),
        "options": {
            "a": "Increase the prior sigma for beta_2",
            "b": "Tighten the intercept prior or switch to a log-normal outcome model",
            "c": "Use a flat (uniform) prior instead",
            "d": "Increase the number of MCMC chains",
        },
        "answer": "b",
        "explanation": (
            "Negative trajectories arise when the intercept prior allows the baseline level to be "
            "near zero and the slope prior allows large negative trends, together producing outcomes "
            "below zero. The fix is either (1) tighten the intercept prior by centering it at the "
            "pre-period mean with smaller sigma, or (2) use a log-normal or Poisson likelihood that "
            "guarantees non-negative predictions."
        ),
    },
    3: {
        "description": (
            "A prior sensitivity analysis shows that the tight prior gives β₂ ∈ [−3.1, +0.5] "
            "and the diffuse prior gives β₂ ∈ [−18.4, −2.3]. The posteriors do not overlap. "
            "What is the correct interpretation?"
        ),
        "options": {
            "a": "The result is prior-robust — both priors give negative effects",
            "b": "The result is prior-sensitive — conclusions depend on prior choice; more data are needed",
            "c": "Use the diffuse prior since it spans a wider range",
            "d": "The analysis is invalid and must be discarded",
        },
        "answer": "b",
        "explanation": (
            "The posteriors differ substantially: the tight prior suggests the effect may be near "
            "zero, while the diffuse prior suggests a large negative effect. This means the data "
            "alone cannot distinguish between these hypotheses — the prior is determining the "
            "conclusion. The honest response is to report the sensitivity and acknowledge that more "
            "post-intervention observations are needed for a robust conclusion."
        ),
    },
    4: {
        "description": (
            "You have access to a meta-analysis of 12 studies with mean effect −10.5 and "
            "cross-study SD of 3.2. What prior for β₂ formally incorporates this evidence?"
        ),
        "options": {
            "a": "Normal(0, 3.2)",
            "b": "Normal(−10.5, 3.2)",
            "c": "HalfNormal(3.2)",
            "d": "Normal(0, 10.5)",
        },
        "answer": "b",
        "explanation": (
            "Normal(−10.5, 3.2) directly encodes the meta-analytic mean (−10.5) and the "
            "cross-study heterogeneity (3.2). This is a principled informative prior that uses all "
            "available external evidence. The prior says: the effect in this new study is probably "
            "similar to effects seen in the literature, within roughly one SD of the meta-analytic "
            "mean."
        ),
    },
    5: {
        "description": (
            "Which prior for σ (the residual noise) is recommended for ITS models, and why?"
        ),
        "options": {
            "a": "Normal(0, 100) — completely uninformative",
            "b": "HalfNormal(σ_Y) — constrained to be positive, scaled to pre-intervention variation",
            "c": "Uniform(0, 1000) — covers all plausible noise levels equally",
            "d": "HalfCauchy(1) — heavy tails prevent underestimating noise",
        },
        "answer": "b",
        "explanation": (
            "HalfNormal(σ_Y) is the recommended prior for σ. It is constrained to positive values "
            "(noise must be positive), and the scale σ_Y means the prior says 'noise is probably "
            "similar to pre-intervention variation.' This puts 95% of prior mass below 2×σ_Y — "
            "reasonable for most datasets. Uniform(0, 1000) can cause sampling problems and "
            "implicitly downweights low-noise solutions. HalfCauchy is acceptable but can allow "
            "very large noise values that slow sampling."
        ),
    },
}


def check_prior_specification(question_number: int, answer: str) -> None:
    """
    Check your answer to a prior specification question.

    Parameters
    ----------
    question_number : int
        1 to 5
    answer : str
        Your answer: 'a', 'b', 'c', or 'd'

    Examples
    --------
    >>> check_prior_specification(1, 'b')   # correct
    >>> check_prior_specification(3, 'a')   # incorrect
    """
    if question_number not in PRIOR_SCENARIOS:
        print(f"  Question {question_number} not found. Choose 1–5.")
        return

    scenario = PRIOR_SCENARIOS[question_number]
    print(f"\nQuestion {question_number}: {scenario['description']}")
    for key, text in scenario["options"].items():
        print(f"  ({key}) {text}")

    correct = answer.lower().strip() == scenario["answer"]
    _print_result(
        correct,
        f"Explanation: {scenario['explanation']}",
        f"The correct answer is ({scenario['answer']}). {scenario['explanation']}",
    )


# ---------------------------------------------------------------------------
# Exercise 3: Convergence diagnostics
# ---------------------------------------------------------------------------

CONVERGENCE_CASES = {
    1: {
        "description": "R-hat = 1.62 for the 'treated' parameter after 4 chains × 1000 draws.",
        "diagnosis": "not_converged",
        "action": "increase_tune",
        "explanation": (
            "R-hat > 1.1 (and especially > 1.2) indicates the chains have not mixed. The 'treated' "
            "coefficient is likely in a poorly-constrained posterior region. Actions: (1) increase "
            "target_accept to 0.9, (2) increase tune steps to 2000, (3) check for highly correlated "
            "parameters with az.plot_pair()."
        ),
    },
    2: {
        "description": "R-hat = 1.003 for all parameters. Bulk ESS = 180 for 'treated' (out of 4000 total draws).",
        "diagnosis": "low_ess",
        "action": "increase_draws",
        "explanation": (
            "R-hat is fine (< 1.01), but ESS = 180 is low — roughly 4.5% efficiency. The posterior "
            "is being sampled slowly due to high autocorrelation in the chain. Solutions: (1) increase "
            "draws to 4000, (2) use reparameterization if possible, (3) increase target_accept. "
            "For reliable HDI estimates, aim for ESS > 400."
        ),
    },
    3: {
        "description": "R-hat = 1.001. Bulk ESS = 3800. Tail ESS = 220 for sigma.",
        "diagnosis": "low_tail_ess",
        "action": "more_draws_or_reparameterize",
        "explanation": (
            "Tail ESS < 400 means the tails of the posterior are not well-sampled even though the "
            "bulk is fine. This affects HDI bounds. For sigma (a scale parameter), this often "
            "indicates the HalfNormal prior is too tight. Try HalfNormal(2 * y_pre_std) or "
            "increase draws to 4000."
        ),
    },
    4: {
        "description": "Divergences: 847 out of 4000 transitions. R-hat looks fine.",
        "diagnosis": "geometry_problem",
        "action": "increase_target_accept_and_reparameterize",
        "explanation": (
            "Many divergences indicate a geometry problem — the posterior has a funnel or highly "
            "curved region that the NUTS integrator cannot traverse accurately. R-hat can look fine "
            "even with divergences if all chains are stuck in the same region. Actions: "
            "(1) increase target_accept to 0.95, (2) look for funnel geometry with az.plot_pair(), "
            "(3) use non-centered parameterization for hierarchical terms."
        ),
    },
}


def diagnose_convergence(case_number: int, diagnosis: str) -> None:
    """
    Diagnose a convergence issue described in the scenario.

    Parameters
    ----------
    case_number : int
        1 to 4
    diagnosis : str
        One of: 'converged', 'not_converged', 'low_ess', 'low_tail_ess', 'geometry_problem'

    Examples
    --------
    >>> diagnose_convergence(1, 'not_converged')  # correct
    >>> diagnose_convergence(2, 'converged')       # incorrect
    """
    if case_number not in CONVERGENCE_CASES:
        print(f"  Case {case_number} not found. Choose 1–4.")
        return

    case = CONVERGENCE_CASES[case_number]
    print(f"\nCase {case_number}: {case['description']}")

    correct = diagnosis.lower().strip() == case["diagnosis"]
    _print_result(
        correct,
        f"Explanation: {case['explanation']}",
        f"The correct diagnosis is '{case['diagnosis']}'. {case['explanation']}",
    )


# ---------------------------------------------------------------------------
# Exercise 4: HDI interpretation
# ---------------------------------------------------------------------------

HDI_SCENARIOS = {
    1: {
        "description": (
            "The 94% HDI for the level change β₂ is [−15.3, −8.7]. "
            "What is the correct interpretation?"
        ),
        "correct": (
            "There is a 94% probability that the true level change lies between −15.3 and −8.7, "
            "given the model and data."
        ),
        "incorrect_options": [
            "If this study were repeated 100 times, 94 of the CIs would contain the true value.",
            "The effect is statistically significant at the 6% level.",
            "We reject the null hypothesis with 94% power.",
        ],
        "explanation": (
            "The HDI is a direct probability statement about the parameter: 94% of the posterior "
            "probability mass falls within these bounds. This is the key advantage over frequentist "
            "confidence intervals, which are statements about long-run procedure coverage, not "
            "about any specific interval."
        ),
    },
    2: {
        "description": (
            "The 94% HDI for β₂ is [−2.1, +1.4]. A colleague says 'the effect is not significant.' "
            "What is the Bayesian response?"
        ),
        "correct": (
            "The HDI includes zero, so the data are consistent with both positive and negative "
            "effects. We cannot make a strong directional claim. P(β₂ < 0) should be reported."
        ),
        "incorrect_options": [
            "We accept the null hypothesis that β₂ = 0.",
            "The posterior is invalid because it includes zero.",
            "More MCMC draws would resolve this.",
        ],
        "explanation": (
            "In Bayesian analysis, we never 'accept the null.' An HDI that includes zero means the "
            "posterior is compatible with zero, but it also gives mass to non-zero values. The right "
            "summary is P(β₂ < 0) — if it's 65%, the effect may lean negative but there is "
            "substantial uncertainty. This is more informative than a binary significance decision."
        ),
    },
}


def check_hdi_interpretation(question_number: int, answer: str) -> None:
    """
    Check your interpretation of a Bayesian HDI.

    Parameters
    ----------
    question_number : int
        1 or 2
    answer : str
        'correct' if you think the stated interpretation is the right Bayesian interpretation,
        or 'incorrect' if you think it is wrong.

    The question presents a scenario — read it and decide if the
    'correct' interpretation given in the scenario is valid.

    Examples
    --------
    >>> check_hdi_interpretation(1, 'correct')   # confirm the correct interpretation
    """
    if question_number not in HDI_SCENARIOS:
        print(f"  Question {question_number} not found. Choose 1 or 2.")
        return

    scenario = HDI_SCENARIOS[question_number]
    print(f"\nQuestion {question_number}: {scenario['description']}")
    print(f"\n  Proposed interpretation: {scenario['correct']}")

    # Both answer choices are always "correct" since we're asking them to confirm
    # the Bayesian interpretation and reason through it
    if answer.lower().strip() == "correct":
        print(f"  CORRECT. {scenario['explanation']}")
    else:
        print(
            f"  The proposed interpretation IS the correct Bayesian interpretation. "
            f"{scenario['explanation']}"
        )


# ---------------------------------------------------------------------------
# Exercise 5: Model comparison with LOO
# ---------------------------------------------------------------------------

LOO_SCENARIOS = {
    1: {
        "description": (
            "LOO comparison gives:\n"
            "  seasonal model: elpd_loo = −142.3, p_loo = 5.1\n"
            "  naive model:    elpd_loo = −171.8, p_loo = 4.2\n"
            "Which model should you use for causal inference?"
        ),
        "answer": "seasonal",
        "explanation": (
            "Higher elpd_loo = better out-of-sample predictive accuracy. The seasonal model has "
            "elpd_loo = −142.3 vs −171.8 for the naive model — a difference of 29.5 ELPD units. "
            "This is a large, meaningful difference (differences > 4 are typically considered "
            "substantial). The seasonal model also has a reasonable p_loo (effective number of "
            "parameters) relative to sample size, so it is not overfitting."
        ),
    },
    2: {
        "description": (
            "LOO comparison gives:\n"
            "  fourier model:  elpd_loo = −143.1, p_loo = 8.2, weight = 0.63\n"
            "  monthly dummies: elpd_loo = −144.8, p_loo = 15.3, weight = 0.37\n"
            "Both models are similar. What does the weight column represent?"
        ),
        "answer": "stacking_weight",
        "explanation": (
            "The weight column in az.compare() shows Bayesian model stacking weights — the optimal "
            "mixture of the two models for out-of-sample prediction. A weight of 0.63 for the "
            "Fourier model means a prediction ensemble giving 63% weight to Fourier and 37% to "
            "monthly dummies would maximize predictive accuracy. When models are close in ELPD, "
            "stacking weights reflect the predictive advantage more accurately than a discrete "
            "model selection."
        ),
    },
}


def check_loo_interpretation(question_number: int, answer: str) -> None:
    """
    Check your interpretation of a LOO cross-validation result.

    Parameters
    ----------
    question_number : int
        1 or 2
    answer : str
        Question 1: 'seasonal' or 'naive'
        Question 2: 'stacking_weight', 'model_probability', or 'r_squared'

    Examples
    --------
    >>> check_loo_interpretation(1, 'seasonal')
    >>> check_loo_interpretation(2, 'stacking_weight')
    """
    if question_number not in LOO_SCENARIOS:
        print(f"  Question {question_number} not found. Choose 1 or 2.")
        return

    scenario = LOO_SCENARIOS[question_number]
    print(f"\nQuestion {question_number}: {scenario['description']}")

    correct = answer.lower().strip() == scenario["answer"]
    _print_result(
        correct,
        f"Explanation: {scenario['explanation']}",
        f"The correct answer is '{scenario['answer']}'. {scenario['explanation']}",
    )


# ---------------------------------------------------------------------------
# Exercise 6: PyMC model building — spot the error
# ---------------------------------------------------------------------------

BUGGY_SNIPPETS = {
    1: {
        "description": "What is wrong with this PyMC ITS model?",
        "code": textwrap.dedent("""
            with pm.Model() as its_model:
                alpha = pm.Normal("alpha", mu=0, sigma=1000)
                beta = pm.Normal("beta", mu=0, sigma=1000, shape=3)
                sigma = pm.Normal("sigma", mu=0, sigma=100)  # <-- problematic line

                mu = alpha + beta[0] * t + beta[1] * treated + beta[2] * t_post
                y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y)
        """),
        "bug": "sigma_can_be_negative",
        "fix": "Use pm.HalfNormal('sigma', sigma=y_pre_std) to constrain sigma to be positive.",
        "explanation": (
            "pm.Normal for sigma allows negative values, which is physically impossible for a "
            "standard deviation. This causes divergences and incorrect posteriors. Always use "
            "pm.HalfNormal, pm.HalfCauchy, or pm.Exponential for scale parameters."
        ),
    },
    2: {
        "description": "What will go wrong when this model tries to generate a counterfactual?",
        "code": textwrap.dedent("""
            with pm.Model() as its_model:
                X_data = pm.Data("X", X_obs)
                alpha = pm.Normal("alpha", mu=y_pre_mean, sigma=2*y_pre_std)
                beta = pm.Normal("beta", mu=0, sigma=y_pre_std, shape=X_obs.shape[1])
                sigma = pm.HalfNormal("sigma", sigma=y_pre_std)
                mu = pm.Deterministic("mu", alpha + pm.math.dot(X_data, beta))
                y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_obs)

            # Later: generate counterfactual
            with its_model:
                pm.set_data({"X": X_counterfactual})  # <-- will this work?
                cf_samples = pm.sample_posterior_predictive(idata)
        """),
        "bug": "missing_mutable_true",
        "fix": "Add mutable=True to pm.Data: pm.Data('X', X_obs, mutable=True)",
        "explanation": (
            "In PyMC >= 5, pm.Data() creates immutable shared variables by default. To later "
            "update the data with pm.set_data(), the variable must be declared mutable=True. "
            "Without this, pm.set_data() raises an error."
        ),
    },
    3: {
        "description": "What convergence problem will this sampling configuration cause?",
        "code": textwrap.dedent("""
            idata = pm.sample(
                draws=100,
                tune=50,
                chains=2,
                target_accept=0.65,
                random_seed=42,
            )
        """),
        "bug": "insufficient_tuning",
        "fix": (
            "Use draws=1000, tune=1000, target_accept=0.9 as a safe default for ITS models. "
            "target_accept=0.65 is too low and will produce many divergences."
        ),
        "explanation": (
            "tune=50 is far too few tuning steps for NUTS to adapt its step size and mass matrix. "
            "ITS models with correlated parameters (e.g., intercept and slope) need at least "
            "1000 tuning steps. target_accept=0.65 is the NUTS default but ITS posteriors often "
            "benefit from 0.9, which reduces divergences by taking smaller, more careful steps."
        ),
    },
}


def check_pymc_bug(snippet_number: int, your_diagnosis: str) -> None:
    """
    Identify the bug in a PyMC code snippet.

    Parameters
    ----------
    snippet_number : int
        1 to 3
    your_diagnosis : str
        One of: 'sigma_can_be_negative', 'missing_mutable_true',
                'insufficient_tuning', 'wrong_prior', 'missing_observed'

    Examples
    --------
    >>> check_pymc_bug(1, 'sigma_can_be_negative')   # correct
    >>> check_pymc_bug(2, 'wrong_prior')              # incorrect
    """
    if snippet_number not in BUGGY_SNIPPETS:
        print(f"  Snippet {snippet_number} not found. Choose 1–3.")
        return

    snippet = BUGGY_SNIPPETS[snippet_number]
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
# Exercise 7: Posterior predictive p-value computation
# ---------------------------------------------------------------------------

def compute_ppc_exercise() -> None:
    """
    Hands-on exercise: compute a Bayesian p-value from scratch.

    Complete the function below, then run the verification.
    """
    print("\n--- Exercise 7: Compute a Bayesian p-value ---\n")

    np.random.seed(42)
    y_obs = np.array([82, 79, 75, 88, 71, 77, 83, 76, 72, 80])

    # Simulated posterior predictive samples: shape (500, 10)
    y_ppc = np.random.normal(loc=78, scale=5, size=(500, 10))

    print("y_obs:", y_obs)
    print("y_ppc shape:", y_ppc.shape)
    print()
    print("Task: Compute the Bayesian p-value for the STANDARD DEVIATION.")
    print("      P(std(y_rep) >= std(y_obs) | y_obs)")
    print()
    print("Steps:")
    print("  1. Compute t_obs = y_obs.std()")
    print("  2. Compute t_rep = y_ppc.std(axis=1)  # std for each replicated dataset")
    print("  3. p_value = (t_rep >= t_obs).mean()")
    print()

    # Solution
    t_obs = y_obs.std()
    t_rep = y_ppc.std(axis=1)
    p_value = (t_rep >= t_obs).mean()

    print(f"  t_obs (observed std):          {t_obs:.3f}")
    print(f"  t_rep mean (PPC std mean):     {t_rep.mean():.3f}")
    print(f"  Bayesian p-value:              {p_value:.3f}")
    print()

    if 0.05 <= p_value <= 0.95:
        print("  INTERPRETATION: The model captures the standard deviation well (p in [0.05, 0.95]).")
    elif p_value < 0.05:
        print("  INTERPRETATION: The model OVERESTIMATES variance (observed std is low relative to PPC).")
    else:
        print("  INTERPRETATION: The model UNDERESTIMATES variance (observed std is high relative to PPC).")


# ---------------------------------------------------------------------------
# Main: run all exercises in sequence
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Module 02 Self-Check: Bayesian ITS with PyMC")
    print("=" * 70)
    print()
    print("Sample answers — replace with your own to test your understanding.")
    print()

    print("--- Bayesian p-value interpretation ---")
    check_bayesian_pvalue(1, "b")
    check_bayesian_pvalue(2, "b")
    check_bayesian_pvalue(3, "b")

    print()
    print("--- Prior specification ---")
    check_prior_specification(1, "b")
    check_prior_specification(2, "b")
    check_prior_specification(3, "b")

    print()
    print("--- Convergence diagnostics ---")
    diagnose_convergence(1, "not_converged")
    diagnose_convergence(2, "low_ess")
    diagnose_convergence(4, "geometry_problem")

    print()
    print("--- HDI interpretation ---")
    check_hdi_interpretation(1, "correct")

    print()
    print("--- LOO model comparison ---")
    check_loo_interpretation(1, "seasonal")

    print()
    print("--- PyMC bug detection ---")
    check_pymc_bug(1, "sigma_can_be_negative")
    check_pymc_bug(2, "missing_mutable_true")

    print()
    compute_ppc_exercise()
