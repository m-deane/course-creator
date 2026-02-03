# Module 6 Quiz: Factor-Augmented Models

## Instructions

**Time Estimate:** 45-60 minutes
**Total Points:** 100
**Attempts Allowed:** 2
**Open Resources:** Yes (course materials, documentation)

This quiz assesses your understanding of diffusion indices, Factor-Augmented Vector Autoregression (FAVAR), structural identification in factor models, and applications to macroeconomic forecasting and policy analysis.

**Question Types:**
- Multiple Choice (select the best answer)
- True/False (with brief justification)
- Short Answer (2-4 sentences)

---

## Part 1: Diffusion Indices (20 points)

### Question 1 (5 points) - Historical Understanding
**Learning Objective:** Understand the development of diffusion indices

Diffusion indices were pioneered by Stock and Watson (2002) for macroeconomic forecasting. What was the primary motivation for their development?

A) To replace VAR models with more interpretable alternatives
B) To handle high-dimensional datasets where n >> T
C) To improve short-term nowcasting accuracy
D) To identify structural shocks in monetary policy

**Feedback:**
- A: DI complements rather than replaces VARs
- B: **Correct!** With many economic indicators but limited time observations, traditional methods fail. DI extracts factors from high-dimensional data.
- C: While useful for nowcasting, this wasn't the primary motivation
- D: Structural identification came later with FAVAR

---

### Question 2 (6 points) - Construction Methodology
**Learning Objective:** Understand diffusion index construction

In constructing diffusion indices via principal components analysis:

**Part A (3 points):** The first principal component is:

A) The eigenvector corresponding to the smallest eigenvalue
B) The linear combination of variables that explains maximum variance
C) The average of all standardized variables
D) The variable most correlated with the outcome variable

**Feedback:**
- A: We use the largest eigenvalue, not smallest
- B: **Correct!** PC1 is the direction of maximum variance in the data
- C: This is a simple average, not PCA
- D: This is supervised selection, PCA is unsupervised

**Part B (3 points):** Why do we typically standardize variables before PCA?

**Expected Answer:** Standardization (z-scoring) ensures that variables with different scales or units don't dominate the factor extraction. Without standardization, variables with larger variances (e.g., dollar values vs. percentages) would receive disproportionate weight.

---

### Question 3 (5 points) - Factor Interpretation
**Learning Objective:** Interpret economic factors

You extract 3 diffusion indices from 120 monthly U.S. economic indicators. The first factor loads heavily on employment, industrial production, and income. What does this factor likely represent?

A) Monetary policy stance
B) Real economic activity / business cycle
C) Financial market conditions
D) Inflation expectations

**Feedback:**
- A: Monetary policy would load on interest rates, credit aggregates
- B: **Correct!** High loadings on real variables (employment, production, income) indicate a real activity factor
- C: Financial conditions would load on spreads, stock prices, volatility
- D: Inflation factors load on price indices, wage growth

---

### Question 4 (4 points) - True/False
**Learning Objective:** Recognize diffusion index properties

**Statement:** "Diffusion indices extracted via PCA are unique and invariant to the ordering of variables in the dataset."

**True** / **False**

**Justification (1-2 sentences):**

**Expected Answer:** False. While PCA results are unique, the factors are only identified up to rotation and sign changes. Different software implementations or PCA algorithms may produce factors that are rotations of each other, though they span the same space. Additionally, the sign of factors is arbitrary.

---

## Part 2: Factor-Augmented VAR (FAVAR) (30 points)

### Question 5 (5 points) - Conceptual Understanding
**Learning Objective:** Understand FAVAR framework

The FAVAR model extends standard VAR by:

A) Adding time-varying parameters to VAR coefficients
B) Including extracted factors alongside observed policy variables
C) Using Bayesian priors to shrink coefficients
D) Estimating factors and VAR simultaneously in one step

**Feedback:**
- A: This describes time-varying parameter VAR
- B: **Correct!** FAVAR: $Y_t = [F_t, M_t]'$ where $F_t$ are factors, $M_t$ are policy variables
- C: This describes Bayesian VAR
- D: FAVAR typically uses two-step estimation

---

### Question 6 (8 points) - Model Specification
**Learning Objective:** Specify FAVAR models

A FAVAR model consists of two equations:

**Factor Equation:** $X_t = \Lambda^f F_t + \Lambda^m M_t + e_t$
**VAR Equation:** $\begin{bmatrix} F_t \\ M_t \end{bmatrix} = \Phi(L) \begin{bmatrix} F_{t-1} \\ M_{t-1} \end{bmatrix} + u_t$

**Part A (4 points):** What does $\Lambda^m$ represent and why is it important?

**Expected Answer:** $\Lambda^m$ represents the direct (contemporaneous) effect of observed policy variables $M_t$ on the large set of economic indicators $X_t$. It's important because it allows policy variables to have immediate impact on the economy beyond their effect through the factors, capturing policy-specific transmission channels.

**Part B (4 points):** In the "slow-moving" identification strategy, why do we sometimes restrict $\Lambda^m = 0$?

**Expected Answer:** The restriction assumes that within-period policy shocks don't affect "slow-moving" macroeconomic variables (like GDP, employment) contemporaneously. This provides identifying restrictions for structural shocks without needing additional assumptions. Fast-moving variables (like asset prices) can still respond immediately through factors.

---

### Question 7 (6 points) - Two-Step Estimation
**Learning Objective:** Understand FAVAR estimation

The most common FAVAR estimation approach is two-step:

**Step 1:** Extract factors $\hat{F}_t$ from $X_t$
**Step 2:** Estimate VAR on $[\hat{F}_t, M_t]'$

**Question:** What is the main limitation of two-step estimation, and how does it affect inference?

**Grading (2 points each):**
1. Identification of limitation
2. Impact on standard errors
3. Potential solution or correction

**Expected Answer:** The main limitation is **generated regressor bias** - Step 2 treats estimated factors as observed data, ignoring estimation uncertainty from Step 1. This causes standard errors in Step 2 to be **understated** (too optimistic), making confidence intervals too narrow and hypothesis tests anti-conservative. Solutions include: (a) bootstrap procedures that re-estimate factors in each draw, (b) one-step Bayesian estimation, or (c) asymptotic corrections for generated regressors.

---

### Question 8 (6 points) - Impulse Response Analysis
**Learning Objective:** Interpret FAVAR impulse responses

**Scenario:** You estimate a FAVAR with 3 factors and 1 monetary policy variable (Federal Funds Rate). After identifying a monetary policy shock using short-run restrictions, you generate impulse responses for 120 economic variables.

**Part A (3 points):** How do you obtain impulse responses for the 120 variables when your VAR only includes factors and the policy rate?

**Expected Answer:** Use the factor equation $X_t = \Lambda^f F_t + \Lambda^m M_t + e_t$. The impulse response of variable $i$ is: IRF$_i(h) = \lambda_i^f \cdot$ IRF$_F(h) + \lambda_i^m \cdot$ IRF$_M(h)$, where IRF$_F(h)$ and IRF$_M(h)$ come from the VAR, and $\lambda_i^f$, $\lambda_i^m$ are loadings from the factor equation.

**Part B (3 points):** You observe that core inflation takes 18 months to respond negatively to a tightening shock, but stock prices drop immediately. Is this consistent with economic theory? Why?

**Expected Answer:** Yes, this is consistent with the "price puzzle" resolution and asset price theory. Stock prices are forward-looking and respond immediately to anticipated future conditions. Core inflation is sticky due to nominal rigidities (menu costs, wage contracts), taking time to adjust. This delayed response is evidence of monetary transmission lags and validates the FAVAR approach.

---

### Question 9 (5 points) - Multiple Choice
**Learning Objective:** Compare FAVAR to alternatives

Compared to a standard small-scale VAR (e.g., 5-7 variables), FAVAR has which advantage?

A) Faster estimation with fewer parameters
B) Clearer structural interpretation
C) Reduced omitted variable bias / information set problem
D) Easier to satisfy stationarity conditions

**Feedback:**
- A: FAVAR is computationally more demanding
- B: Both require identification strategies; FAVAR isn't inherently clearer
- C: **Correct!** FAVAR incorporates information from 100+ variables via factors, reducing omitted variable bias that plagues small VARs
- D: Stationarity requirements are similar

---

## Part 3: Structural Identification (30 points)

### Question 10 (6 points) - Identification Strategies
**Learning Objective:** Understand identification approaches

Match each identification strategy to its description:

1. Short-run restrictions
2. Long-run restrictions
3. Sign restrictions
4. High-frequency identification

A) Use external instruments from event studies or market reactions
B) Impose that certain shocks have no permanent effect on some variables
C) Assume some variables don't respond contemporaneously to shocks
D) Restrict the direction of impulse responses without exact magnitude

**Answer:** 1-C, 2-B, 3-D, 4-A (2 points for perfect match, 1 point for 3/4 correct)

**Follow-up (2 points):** Which approach is most robust to misspecification but provides set identification rather than point identification?

**Expected Answer:** Sign restrictions (3). They provide set identification (a region of plausible responses) rather than a unique point estimate, making them robust to precise specification assumptions but less informative.

---

### Question 11 (8 points) - Cholesky Decomposition
**Learning Objective:** Apply short-run restrictions

Consider a FAVAR with 2 factors and the Federal Funds Rate (FFR). You impose a Cholesky ordering: [Factor 1, Factor 2, FFR].

**Part A (4 points):** What economic assumption does this ordering impose?

**Expected Answer:** This assumes that monetary policy (FFR) can respond contemporaneously to both factors (since it's ordered last), but factors cannot respond to monetary policy shocks within the same period. This is a "slow-moving variables" restriction - the economy takes time to react to policy, but policy can react quickly to economic conditions.

**Part B (4 points):** When would this ordering be inappropriate? Give a specific example.

**Expected Answer:** This ordering would be inappropriate if Factor 1 or Factor 2 loads heavily on asset prices or financial variables. For example, if Factor 2 is a financial conditions factor, it should respond immediately to monetary policy shocks (rates affect asset prices instantly). In this case, FFR should be ordered before Factor 2 in the Cholesky decomposition.

---

### Question 12 (8 points) - External Instruments
**Learning Objective:** Understand IV identification in FAVAR

Stock and Watson (2012) and others use high-frequency interest rate surprises around FOMC announcements as instruments for monetary policy shocks.

**Part A (4 points):** What two conditions must the instrument satisfy to validly identify monetary policy shocks?

**Expected Answer:**
1. **Relevance:** The instrument must be strongly correlated with monetary policy surprises (first-stage strength)
2. **Exogeneity:** The instrument must be uncorrelated with other structural shocks affecting the economy (orthogonal to non-policy shocks)

**Part B (4 points):** Why are high-frequency (e.g., 30-minute window) rate changes around FOMC announcements preferred over daily changes?

**Expected Answer:** Narrower windows isolate the pure monetary policy surprise, excluding confounding information that arrives throughout the day. A 30-minute window captures market's reaction to the policy decision while avoiding contamination from other economic news, macroeconomic data releases, or geopolitical events that might occur on the same day.

---

### Question 13 (8 points) - Short Answer
**Learning Objective:** Design identification strategy

**Scenario:** You are estimating a FAVAR to study the effects of:
- Monetary policy shocks (interest rate changes)
- Fiscal policy shocks (government spending changes)
- Oil supply shocks

You have 3 factors extracted from 150 macroeconomic and financial indicators, plus 3 policy/exogenous variables (FFR, government spending, oil prices).

**Question:** Propose an identification strategy and justify your choice. Address:
1. Which method (short-run, long-run, sign restrictions, IV)
2. Key restrictions or instruments you'd use
3. One challenge you anticipate

**Grading Rubric:**
- Method choice with justification (3 points)
- Specific restrictions/instruments (3 points)
- Challenge identified (2 points)

**Model Answer:**

**Method:** Combine multiple approaches - **IV for monetary policy** + **sign restrictions for fiscal and oil shocks**.

**Restrictions/Instruments:**
1. **Monetary:** Use high-frequency FOMC surprises as external instrument (Gertler-Karadi instrument)
2. **Fiscal:** Sign restrictions - positive spending shock increases GDP and inflation, decreases debt/GDP ratio in medium run
3. **Oil:** Identify via heteroskedasticity (oil shocks have different volatility regimes) or use Kilian-style narrative approach (OPEC announcements)

**Challenge:** Ensuring contemporaneous orthogonality between shocks. Monetary policy may react to oil prices, and fiscal policy may respond to economic conditions. Solution: Use information from different frequencies (high-freq for monetary, quarterly for fiscal) or impose block exogeneity restrictions based on timing and decision-making processes.

(Alternative valid answers: pure sign restrictions across all shocks, mix of zero short-run restrictions, or Bayesian approach with informative priors)

---

## Part 4: Applications and Interpretation (20 points)

### Question 14 (6 points) - Forecast Evaluation
**Learning Objective:** Evaluate FAVAR forecast performance

**Scenario:** You compare forecasts from three models:
- Simple AR(4) on GDP growth
- Small VAR(4) with 5 variables
- FAVAR(4) with 3 factors + 2 policy variables (from 100+ indicators)

Results (RMSFE for GDP growth, 4-quarter ahead):
- AR: 1.20
- Small VAR: 1.15
- FAVAR: 0.98

**Part A (3 points):** The FAVAR achieves lower RMSFE. Does this definitively prove it's the best model?

**Expected Answer:** No. We need to conduct formal tests (Diebold-Mariano, Giacomini-White) to determine if the improvement is statistically significant. We should also check robustness across different samples, assess other loss functions (e.g., directional accuracy for recession forecasting), and evaluate forecast combinations that might dominate any single model.

**Part B (3 points):** In what scenario might you prefer the simpler AR model despite its higher RMSFE?

**Expected Answer:** When interpretability and simplicity are paramount, when you have limited data for out-of-sample evaluation, when computational resources are constrained, or when the RMSFE difference is not statistically significant. Also, if the FAVAR overfits in real-time (requires data revisions), the simpler AR might be more robust.

---

### Question 15 (8 points) - Policy Counterfactuals
**Learning Objective:** Construct policy counterfactuals

A key application of FAVAR is constructing counterfactual scenarios (e.g., "What if the Fed had not cut rates in 2008?").

**Question:** Describe the steps to construct a counterfactual where monetary policy remained unchanged during a crisis period. What are two important caveats to interpretation?

**Expected Answer:**

**Steps:**
1. Estimate FAVAR on full sample and identify monetary policy shocks
2. Decompose historical data into contributions from each shock type (historical decomposition)
3. Remove monetary policy shock contributions during the counterfactual period
4. Reconstruct variables using only non-policy shocks and initial conditions
5. Compare actual vs. counterfactual paths

**Caveats:**
1. **Lucas Critique:** Assumes structural parameters (factor loadings, VAR coefficients) remain constant under alternative policy. Agents' behavior might change with different policy.
2. **Partial equilibrium:** Assumes other shocks (fiscal, productivity, etc.) would have been identical in the counterfactual scenario. In reality, shocks may be correlated with policy actions or regime changes.

(Additional acceptable caveats: identification assumptions may not hold out of sample, uncertainty bands are typically very wide, assumes linearity)

---

### Question 16 (6 points) - True/False with Justification
**Learning Objective:** Recognize FAVAR limitations

**Statement:** "FAVAR models can perfectly separate the effects of anticipated vs. unanticipated monetary policy changes."

**True** / **False**

**Justification (3-4 sentences):**

**Expected Answer:** False. Standard FAVAR identifies total policy shocks but doesn't automatically decompose them into anticipated and unanticipated components. This requires additional structure:
1. External instruments (high-frequency identification) help isolate unanticipated shocks
2. Forward guidance and interest rate forecasts can proxy for anticipated changes
3. Recent work uses shadow rates or incorporates expectations data, but full separation remains challenging

Perfect separation would require observing agents' complete information sets and expectations, which is infeasible. Most FAVAR applications identify a mix of anticipated and surprise components unless specifically designed otherwise.

---

## Bonus Question (5 points)

### Question 17 - Research Frontier
**Learning Objective:** Connect to current research

**Question:** Recent papers have criticized standard FAVAR for the "poor man's invertibility" problem. Briefly explain this issue and propose one solution.

**Expected Answer:**

**Problem:** If the econometrician observes fewer variables than economic agents (which is always true), the identified VAR shocks are non-fundamental - they can't fully recover true structural shocks. This is because the true shocks may be a non-invertible function of observables. In FAVAR, even with many factors, we may not span the full information set agents use.

**Solutions (any one acceptable):**
1. **External instruments:** Use instrumental variables that are truly exogenous and informative about specific shocks
2. **Higher-order moments:** Exploit non-Gaussianity for identification (independent component analysis)
3. **Larger information sets:** Include expectations data, forecasts, or forward-looking variables that approximate agents' information
4. **Structural models:** Impose economic theory restrictions from DSGE models to discipline identification
5. **Longer lags:** Use longer lag lengths to better approximate agents' information sets

---

## Quiz Completion

**Total Points: 105 (100 + 5 bonus)**

### Learning Objectives Coverage

- ✓ Diffusion index construction and interpretation
- ✓ FAVAR specification and estimation
- ✓ Structural identification strategies (short-run, long-run, sign, IV)
- ✓ Impulse response analysis in high-dimensional settings
- ✓ Forecasting and policy counterfactual applications
- ✓ Critical assessment of FAVAR limitations

### Submission Instructions

1. Save your answers in a single document (PDF or markdown)
2. Show all work for mathematical questions
3. For scenarios, provide complete reasoning
4. Submit via course platform by due date

### Grading Scale

- 90-100: Excellent understanding of factor-augmented methods
- 80-89: Good grasp with minor gaps
- 70-79: Adequate understanding, review identification
- Below 70: Significant gaps, attend office hours

### Recommended Review Materials

If you scored below 80 on any section:
- **Part 1 (Diffusion Indices):** Re-read Stock & Watson (2002), review PCA lectures
- **Part 2 (FAVAR):** Study Bernanke, Boivin & Eliasz (2005), practice VAR estimation
- **Part 3 (Identification):** Review identification lectures, work through examples
- **Part 4 (Applications):** Study empirical papers, replicate results

**After submission, review the detailed answer key and solutions manual.**
