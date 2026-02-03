# Module 1: Bayesian Fundamentals - Quiz

**Time Limit:** 30 minutes
**Total Points:** 100 (15 questions)
**Passing Score:** 80%

**Instructions:**
- Answer all questions
- Select the BEST answer for multiple choice
- Show work for calculation problems
- Closed book (but you may use a calculator)

---

## Section 1: Bayes' Theorem (30 points)

### Question 1 (10 points)

A commodity trading desk uses a model that correctly identifies price increases 80% of the time (true positive rate). The model has a 15% false positive rate. Historically, prices increase 40% of days.

If the model predicts a price increase today, what is the probability that the price actually increases?

**A)** 0.68
**B)** 0.74
**C)** 0.80
**D)** 0.84

<details>
<summary>Answer</summary>

**Correct Answer: B (0.74)**

**Solution:**
Let I = price increases, M = model predicts increase

Given:
- P(M|I) = 0.80 (true positive rate)
- P(M|¬I) = 0.15 (false positive rate)
- P(I) = 0.40

By Bayes' theorem:
$$P(I|M) = \frac{P(M|I) P(I)}{P(M)}$$

First calculate P(M) using law of total probability:
$$P(M) = P(M|I)P(I) + P(M|\neg I)P(\neg I)$$
$$P(M) = 0.80 \times 0.40 + 0.15 \times 0.60 = 0.32 + 0.09 = 0.41$$

Then:
$$P(I|M) = \frac{0.80 \times 0.40}{0.41} = \frac{0.32}{0.41} \approx 0.78$$

**Closest answer: B (0.74)**

**Points:** Full credit for B, partial for correct process with arithmetic error
</details>

---

### Question 2 (10 points)

In Bayesian inference, the posterior distribution is:

**A)** Always narrower than the prior
**B)** Proportional to the likelihood times the prior
**C)** The same as the maximum likelihood estimate
**D)** Independent of the prior choice for large datasets

Select ALL correct answers.

<details>
<summary>Answer</summary>

**Correct Answers: B and D**

**Explanation:**

**B is correct:** $$p(\theta|y) \propto p(y|\theta) p(\theta)$$
This is the fundamental definition of Bayesian inference.

**D is correct:** As n → ∞, data overwhelms prior and posterior → likelihood-based estimate (asymptotic agreement with MLE).

**A is wrong:** Posterior can be wider than prior in rare cases (though typically narrower).

**C is wrong:** Posterior is a distribution, MLE is a point estimate. The posterior mode may equal MLE with flat prior.

**Grading:**
- Both correct: 10 points
- One correct: 5 points
- Neither: 0 points
</details>

---

### Question 3 (10 points)

Which statement about the prior distribution is TRUE?

**A)** It must always be non-informative to avoid bias
**B)** It represents your belief about parameters before seeing current data
**C)** It has no effect on the posterior with enough data
**D)** It should be chosen to maximize model fit

<details>
<summary>Answer</summary>

**Correct Answer: B**

**Explanation:**

**B is correct:** The prior explicitly encodes pre-data beliefs or assumptions.

**A is wrong:** Informative priors are often beneficial, encoding domain knowledge or providing regularization.

**C is somewhat true but misleading:** While priors have diminishing effect as n → ∞, they still matter for finite data.

**D is wrong:** Priors should reflect genuine beliefs or regularization goals, not be chosen to fit data better (that's data snooping).

**Points:** 10 for B, 0 for others
</details>

---

## Section 2: Conjugate Priors (20 points)

### Question 4 (10 points)

You observe y ~ Binomial(n, θ) and use a Beta(α, β) prior for θ.

What is the posterior distribution?

**A)** Beta(α + y, β + n - y)
**B)** Beta(α + n, β + y)
**C)** Normal(α/(α+β), √(αβ/n))
**D)** Binomial(n + α, θ)

<details>
<summary>Answer</summary>

**Correct Answer: A**

**Explanation:**

Beta is conjugate to Binomial:
- Prior: Beta(α, β)
- Likelihood: Binomial(n, θ) with y successes
- Posterior: Beta(α + y, β + n - y)

**Interpretation:**
- α represents prior pseudo-successes
- β represents prior pseudo-failures
- Posterior adds actual successes (y) to α
- Posterior adds actual failures (n-y) to β

**Points:** 10 for A, 0 for others
</details>

---

### Question 5 (10 points)

A commodity analyst models daily returns μ with prior μ ~ Normal(0, σ²). They observe n returns with sample mean ȳ and known variance σ².

What is the posterior mean?

**A)** ȳ
**B)** 0
**C)** (ȳ + 0) / 2
**D)** Weighted average of 0 and ȳ based on relative precisions

<details>
<summary>Answer</summary>

**Correct Answer: D**

**Explanation:**

For Normal-Normal conjugacy:
$$\mu_{posterior} = \frac{\frac{1}{\sigma_0^2} \mu_0 + \frac{n}{\sigma^2} \bar{y}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}$$

This is a precision-weighted average:
- Prior precision: 1/σ₀²
- Data precision: n/σ²

**Interpretation:**
- More data (large n) → posterior closer to ȳ
- Tight prior (small σ₀) → posterior closer to prior mean
- This is NOT a simple average (C is wrong)

**Points:** 10 for D, partial for recognizing weighted average concept
</details>

---

## Section 3: Bayesian Regression (30 points)

### Question 6 (10 points)

In Bayesian linear regression y = Xβ + ε, what does a tight prior on β accomplish?

**A)** It increases the likelihood of the data
**B)** It acts as regularization, shrinking coefficients toward the prior mean
**C)** It speeds up MCMC sampling
**D)** It guarantees posterior normality

<details>
<summary>Answer</summary>

**Correct Answer: B**

**Explanation:**

A tight prior (small variance) constrains β to be near the prior mean. This is equivalent to:
- **Ridge regression:** Prior β ~ N(0, σ²) ↔ L2 penalty
- **Lasso:** Prior β ~ Laplace(0, b) ↔ L1 penalty

**Why others are wrong:**
- **A:** Prior affects posterior, not likelihood directly
- **C:** Tight priors can actually slow sampling (narrow distribution harder to explore)
- **D:** Normality depends on likelihood and prior family, not tightness

**Application to commodities:**
If you know inventory coefficient should be near -2.5, use β ~ N(-2.5, 0.5) to regularize.

**Points:** 10 for B
</details>

---

### Question 7 (10 points)

What is a 95% credible interval?

**A)** An interval that contains 95% of the data
**B)** An interval that will contain the true parameter 95% of the time in repeated samples
**C)** An interval containing 95% of the posterior probability mass
**D)** The range of values within 1.96 standard deviations of the mean

<details>
<summary>Answer</summary>

**Correct Answer: C**

**Explanation:**

**Credible interval (Bayesian):** Given the data, there's a 95% probability the parameter is in this interval.

$$P(\theta \in [L, U] | y) = 0.95$$

**Confidence interval (Frequentist):** If we repeated the experiment infinitely, 95% of constructed intervals would contain θ.

**Key difference:**
- Credible: Makes probabilistic statement about parameter (what we want!)
- Confidence: Makes statement about procedure, not parameter

**Why D is wrong:** That's specifically for Normal distributions; credible intervals apply to any posterior.

**Points:** 10 for C
</details>

---

### Question 8 (10 points)

You fit two Bayesian regression models for crude oil prices:
- **Model A:** WAIC = 450.2
- **Model B:** WAIC = 448.7

Which statement is correct?

**A)** Model A fits better (lower WAIC is worse)
**B)** Model B fits better (lower WAIC is better)
**C)** The models are statistically equivalent
**D)** Need p-value to decide

<details>
<summary>Answer</summary>

**Correct Answer: B**

**Explanation:**

**WAIC (Widely Applicable Information Criterion):**
- Bayesian model selection criterion
- **Lower is better** (like AIC, BIC)
- Balances fit and complexity
- Difference of 1.5 suggests Model B is preferred

**Formula:**
$$WAIC = -2 \times (lppd - p_{WAIC})$$

Where:
- lppd: log pointwise predictive density (fit)
- p_WAIC: effective number of parameters (complexity penalty)

**Why others are wrong:**
- **A:** Backward interpretation
- **C:** Difference > 1 suggests meaningful distinction
- **D:** Bayesian model comparison uses WAIC/LOO, not p-values

**Points:** 10 for B
</details>

---

## Section 4: Interpretation & Application (20 points)

### Question 9 (5 points)

A posterior distribution for β_inventory has mean -2.3 and 95% CI [-3.1, -1.5].

What can you conclude?

**A)** On average, a 1-unit inventory increase decreases price by $2.30
**B)** We're 95% certain β is between -3.1 and -1.5
**C)** The effect is statistically significant (interval doesn't include 0)
**D)** All of the above

<details>
<summary>Answer</summary>

**Correct Answer: D**

All three statements are valid interpretations:

**A:** Posterior mean is best point estimate for effect size

**B:** Bayesian credible intervals allow probabilistic statements about parameters

**C:** Since 0 is not in the interval, there's strong evidence for a negative effect

**Points:** 5 for D, 3 for recognizing any two, 1 for one
</details>

---

### Question 10 (5 points)

Why use Bayesian regression for commodity forecasting instead of OLS?

**A)** It's always more accurate
**B)** It provides uncertainty quantification for predictions
**C)** It doesn't require assumptions
**D)** It's faster to compute

<details>
<summary>Answer</summary>

**Correct Answer: B**

**Explanation:**

Bayesian regression provides:
1. **Full posterior distribution** for parameters (not just point estimates)
2. **Posterior predictive distribution** with natural prediction intervals
3. **Principled regularization** via priors
4. **Incorporation of domain knowledge**

**Why others are wrong:**
- **A:** Not always more accurate; depends on prior quality and data
- **C:** Still requires model assumptions (linearity, error distribution)
- **D:** Actually slower (MCMC vs closed-form OLS)

**For commodities:**
Trading decisions require risk assessment → Need full predictive distribution, not just E[y].

**Points:** 5 for B
</details>

---

### Question 11 (10 points)

**Short Answer:** Explain in 2-3 sentences why a trader would prefer Bayesian regression over frequentist regression when forecasting commodity prices for risk management.

<details>
<summary>Sample Answer</summary>

**Strong answer (10 points):**

"Bayesian regression provides a full predictive distribution for future prices, not just point forecasts. This allows traders to compute risk metrics like Value-at-Risk (VaR) and quantify position sizing based on forecast uncertainty. The credible intervals directly inform hedging decisions, whereas frequentist confidence intervals don't provide probabilistic statements about future prices."

**Key elements:**
- Mentions uncertainty quantification (5 pts)
- Connects to risk management application (3 pts)
- Contrasts with frequentist limitations (2 pts)

**Grading rubric:**
- 10: All key elements clearly explained
- 7: Mentions uncertainty but misses application
- 4: Vague reference to "better uncertainty"
- 0: No substantive answer
</details>

---

## Bonus Questions (10 points extra credit)

### Bonus 1 (5 points)

Derive the posterior distribution for θ given:
- Prior: θ ~ Beta(2, 2)
- Data: 7 successes in 10 Bernoulli trials

Show your work.

<details>
<summary>Answer</summary>

**Solution:**

Beta-Binomial conjugacy:
- Prior: Beta(α=2, β=2)
- Likelihood: Binomial(n=10, y=7)
- Posterior: Beta(α + y, β + n - y) = Beta(2+7, 2+10-7) = **Beta(9, 5)**

**Interpretation:**
- Prior mean: 2/(2+2) = 0.5 (neutral)
- Posterior mean: 9/(9+5) = 0.643
- Data shifted belief toward higher success probability

**Points:**
- 5: Correct answer with clear steps
- 3: Correct formula but arithmetic error
- 1: Recognizes conjugacy but wrong application
</details>

---

### Bonus 2 (5 points)

Explain the connection between Bayesian priors and regularization in machine learning (2-3 sentences).

<details>
<summary>Sample Answer</summary>

**Strong answer:**

"A Gaussian prior N(0, σ²) on regression coefficients is equivalent to L2 (ridge) regularization, where 1/σ² plays the role of the regularization parameter λ. Similarly, a Laplace prior corresponds to L1 (Lasso) regularization. This equivalence shows that regularization can be interpreted as encoding prior beliefs about coefficient magnitudes."

**Grading:**
- 5: Clear connection with specific example
- 3: Vague reference to equivalence
- 0: Incorrect or missing
</details>

---

## Answer Key Summary

| Question | Answer | Points |
|----------|--------|--------|
| 1 | B | 10 |
| 2 | B, D | 10 |
| 3 | B | 10 |
| 4 | A | 10 |
| 5 | D | 10 |
| 6 | B | 10 |
| 7 | C | 10 |
| 8 | B | 10 |
| 9 | D | 5 |
| 10 | B | 5 |
| 11 | Rubric | 10 |
| **Total** | | **100** |

**Passing:** 80+ points

---

## Post-Quiz Reflection

After completing the quiz, review:

1. **Missed questions:** Which concepts need more study?
2. **Time management:** Did you finish in 30 minutes?
3. **Confidence:** Were you unsure on questions you got right?

**If you scored:**
- **90-100:** Excellent understanding, ready for Module 2
- **80-89:** Good foundation, review weak areas
- **70-79:** Borderline, spend extra time on Module 1 notebooks
- **< 70:** Retake Module 1 materials before proceeding

**Resources for review:**
- Module 1 guides (detailed explanations)
- Interactive notebooks (hands-on practice)
- Additional readings (deeper theory)
- Office hours (personalized help)

---

*"Assessment is not about proving what you know, but identifying what you need to learn next."*
