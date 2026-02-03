# Module 8 Quiz: Advanced Topics in Dynamic Factor Models

## Instructions

**Time Estimate:** 50-65 minutes
**Total Points:** 100
**Attempts Allowed:** 2
**Open Resources:** Yes (course materials, documentation)

This quiz assesses your understanding of advanced topics in dynamic factor models, including time-varying parameters, regime-switching models, non-Gaussian innovations, connections to machine learning, and emerging research directions.

**Question Types:**
- Multiple Choice (select the best answer)
- True/False (with brief justification)
- Short Answer (2-4 sentences)
- Conceptual synthesis

---

## Part 1: Time-Varying Parameters (25 points)

### Question 1 (6 points) - Conceptual Understanding
**Learning Objective:** Understand motivation for time variation

Why might we want factor models with time-varying parameters rather than constant parameters?

A) Time-varying models always fit training data better
B) Economic relationships evolve due to structural changes, policy shifts, and technological progress
C) They are easier to estimate than constant-parameter models
D) They guarantee better out-of-sample forecasts

**Feedback:**
- A: Better fit doesn't justify complexity without economic motivation
- B: **Correct!** Factor loadings and dynamics change with structural shifts (e.g., Great Moderation, financial crises, digitalization)
- C: Time-varying models are more complex to estimate
- D: They don't guarantee better forecasts; must balance fit and overfitting

**Follow-up (2 points):** Give one specific example of a structural change that would affect factor loadings.

**Expected Answer:** The rise of globalization increased co-movement between national economies, strengthening loadings on global factors. Alternatively: Financial deregulation changed how credit variables load on activity factors; pandemic changed remote work's relationship with economic indicators; monetary policy regime changes (inflation targeting) altered how interest rates affect real variables.

---

### Question 2 (8 points) - Time-Varying Parameter Specification
**Learning Objective:** Specify TVP models

A time-varying parameter dynamic factor model can be written as:

$$X_t = \Lambda_t F_t + e_t$$
$$F_t = \Phi_t F_{t-1} + u_t$$

where $\Lambda_t$ and/or $\Phi_t$ evolve over time.

**Part A (4 points):** What are two common specifications for time variation in $\Lambda_t$?

**Expected Answer (any two):**

1. **Random walk:** $\Lambda_t = \Lambda_{t-1} + \eta_t$ where $\eta_t \sim N(0, Q)$ - allows gradual drift
2. **Markov-switching:** $\Lambda_t = \Lambda_{(s_t)}$ where $s_t \in \{1,...,K\}$ follows Markov chain - discrete regime changes
3. **Stochastic volatility in changes:** $\Lambda_t = \Lambda_{t-1} + \sigma_t \eta_t$ where $\log(\sigma_t^2)$ follows AR(1) - time-varying speed of change
4. **Deterministic (smooth):** $\Lambda_t = g(t/T, \theta)$ using splines or polynomials - smooth evolution
5. **Threshold models:** $\Lambda_t = \Lambda^{(1)} \cdot I(z_t \leq \gamma) + \Lambda^{(2)} \cdot I(z_t > \gamma)$ - regime depends on observable $z_t$

**Part B (4 points):** What is the main computational challenge in estimating TVP factor models?

**Expected Answer:** The **curse of dimensionality** in state-space representation. With time-varying parameters, the state vector includes both factors $F_t$ AND parameters $\Lambda_t$, $\Phi_t$. The Kalman filter must track $K$ factors plus $n \times K$ loadings plus $K^2$ dynamics, making the state dimension very high. This requires:
- Computational strategies: particle filters, MCMC, variational inference
- Dimension reduction: structured time variation (e.g., only first r loadings vary)
- Approximate methods: forgetting factors, rolling windows

---

### Question 3 (6 points) - Stochastic Volatility
**Learning Objective:** Understand time-varying volatility

Consider a factor model where factor volatility varies over time:

$$F_t = \Phi F_{t-1} + \sigma_t u_t, \quad u_t \sim N(0, I)$$
$$\log(\sigma_t^2) = \mu + \phi \log(\sigma_{t-1}^2) + v_t, \quad v_t \sim N(0, \tau^2)$$

**Part A (3 points):** How does this differ from a constant volatility model? What economic phenomenon does it capture?

**Expected Answer:** Unlike constant volatility where $\text{Var}(F_t | F_{t-1})$ is fixed, this model allows variance to change over time following an AR(1) process in log-space. It captures **volatility clustering** observed in economic data - periods of high uncertainty (crises) vs. low uncertainty (Great Moderation). The factor can have large innovations during recessions, small innovations during expansions.

**Part B (3 points):** Why is the log-transform used for volatility rather than modeling $\sigma_t^2$ directly?

**Expected Answer:** Log-transform ensures $\sigma_t^2 > 0$ (volatility is always positive) without imposing constraints in estimation. It also captures the multiplicative nature of volatility (percentage changes rather than level changes) and makes the distribution more Gaussian. Direct modeling of $\sigma_t^2$ could produce negative values.

---

### Question 4 (5 points) - Short Answer
**Learning Objective:** Apply TVP models

**Question:** You estimate a TVP factor model on quarterly data from 1960-2023. You find that the first factor's persistence parameter $\phi_t$ (in $F_t = \phi_t F_{t-1} + u_t$) declined from 0.95 in the 1970s to 0.75 in the 2010s. Provide two possible economic interpretations.

**Expected Answer (any two):**

1. **Improved stabilization policy:** More active monetary/fiscal policy in recent decades dampens shocks faster, reducing persistence

2. **Structural change in economy:** Shift from manufacturing (durable goods, high persistence) to services (less persistent) reduces factor memory

3. **Information technology:** Faster information flow and production adjustments allow quicker responses to shocks

4. **Financial deepening:** Better developed financial markets allow smoother consumption/investment, reducing propagation

5. **Globalization:** International trade diversifies shocks, reducing domestic persistence

6. **Measurement and data quality:** Earlier data might have more measurement error or revision, artificially inflating persistence

---

## Part 2: Regime-Switching Models (20 points)

### Question 5 (5 points) - Markov-Switching Framework
**Learning Objective:** Understand regime-switching models

In a Markov-switching factor model, the state $s_t \in \{1, 2, ..., K\}$ follows a first-order Markov chain with transition probabilities:

$$P(s_t = j | s_{t-1} = i) = p_{ij}$$

**Question:** What is the key assumption that makes this a "Markov" chain?

A) Transition probabilities are constant over time
B) Current state depends only on the immediately preceding state
C) All states are equally likely
D) The process eventually returns to any starting state

**Feedback:**
- A: This is often assumed but not the defining property
- B: **Correct!** Markov property: $P(s_t | s_{t-1}, s_{t-2}, ...) = P(s_t | s_{t-1})$ - memoryless
- C: States can have different long-run probabilities
- D: This is ergodicity, not the Markov property

---

### Question 6 (8 points) - Regime Identification
**Learning Objective:** Identify economic regimes

**Scenario:** You estimate a 2-regime Markov-switching factor model on U.S. data (1960-2023). The estimated regimes show:

**Regime 1:** High factor volatility, negative average factor values, short duration (avg 5 quarters)
**Regime 2:** Low factor volatility, positive average factor values, long duration (avg 30 quarters)

**Part A (4 points):** What do these regimes likely represent economically?

**Expected Answer:**
- **Regime 1:** Recessions - high volatility (uncertainty), negative factor (below trend activity), short duration (average recession length ~6 months)
- **Regime 2:** Expansions - low volatility (stable growth), positive factor (above trend), long duration (expansions last years)

This aligns with business cycle asymmetry: recessions are sharp and short, expansions are gradual and long.

**Part B (4 points):** The transition matrix is estimated as:

$$P = \begin{bmatrix} 0.70 & 0.30 \\ 0.05 & 0.95 \end{bmatrix}$$

where rows are "from" states, columns are "to" states. Calculate the expected duration of Regime 1 and interpret.

**Expected Answer:**

Expected duration in regime $i$ is $\frac{1}{1 - p_{ii}}$.

For Regime 1: $\text{Duration} = \frac{1}{1 - 0.70} = \frac{1}{0.30} = 3.33$ quarters

**Interpretation:** On average, Regime 1 (recession) lasts about 3.3 quarters (10 months), after which it transitions to Regime 2 with probability 0.30 each quarter. This matches stylized facts about U.S. recessions being shorter than expansions.

(Regime 2 duration: $1/(1-0.95) = 20$ quarters, about 5 years)

---

### Question 7 (7 points) - Path Dependence
**Learning Objective:** Understand filtering vs. smoothing

In regime-switching models, we distinguish between:
- **Filtered probabilities:** $P(s_t | Y_{1:t})$ - probability of regime at t using data up to t
- **Smoothed probabilities:** $P(s_t | Y_{1:T})$ - probability of regime at t using full sample

**Part A (3 points):** Why are smoothed probabilities typically more accurate than filtered probabilities?

**Expected Answer:** Smoothed probabilities use **future data** (beyond time t) which contains information about the current regime. For example, if you observe a recession in t+1, t+2, you can infer t was likely also in recession even if the signal at t was noisy. Smoothing uses the full sample, reducing uncertainty about regimes in the middle of the sample.

**Part B (4 points):** For real-time recession nowcasting, which probabilities should you use and why?

**Expected Answer:** Use **filtered probabilities** $P(s_t | Y_{1:t})$. Real-time nowcasting must use only information available at time t - future data isn't available yet. Smoothed probabilities would involve look-ahead bias, using data that hasn't been released.

However, there's a trade-off: filtered probabilities have higher uncertainty and may lag regime changes (slow to detect start/end of recessions). In practice, combine filtered probabilities with external indicators to improve real-time detection.

---

## Part 3: Non-Gaussian Innovations (20 points)

### Question 8 (6 points) - Fat Tails and Outliers
**Learning Objective:** Recognize limitations of Gaussianity

Standard factor models assume Gaussian innovations: $e_t \sim N(0, \Sigma)$. However, economic and financial data often exhibit fat tails.

**Part A (3 points):** What is one consequence of assuming Gaussianity when true innovations have fat tails?

**Expected Answer:**
- **Inefficient estimation:** Maximum likelihood under Gaussianity gives undue weight to outliers, producing inefficient estimates
- **Poor uncertainty quantification:** Confidence intervals and forecast distributions understate tail risk (e.g., miss extreme recessions)
- **Suboptimal forecasting:** Gaussian forecasts miss asymmetry and extreme events
- **Incorrect inference:** Standard errors and hypothesis tests are unreliable

**Part B (3 points):** Name two alternative distributional assumptions that accommodate fat tails.

**Expected Answer (any two):**

1. **Student's t-distribution:** Heavier tails than Gaussian, controlled by degrees of freedom parameter
2. **Mixture of normals:** $e_t \sim \sum_{k=1}^K w_k N(\mu_k, \sigma_k^2)$ - can approximate fat tails and multimodality
3. **Generalized error distribution (GED):** Includes Gaussian (shape=2), Laplace (shape=1), and fatter tails
4. **Stable distributions:** Heavy-tailed, but no closed-form MLE (harder to use)
5. **Skewed distributions:** Skew-t, skew-normal - accommodate both fat tails and asymmetry

---

### Question 9 (8 points) - Robust Estimation
**Learning Objective:** Apply robust methods

To handle outliers and non-Gaussianity, robust estimation methods can be used.

**Part A (4 points):** Explain how the Huber loss function provides robustness compared to squared error loss in factor estimation.

**Expected Answer:**

Squared error loss: $\ell(e) = e^2$ - penalizes large errors quadratically, giving extreme values high influence

Huber loss:
$$\ell_\delta(e) = \begin{cases} \frac{1}{2}e^2 & \text{if } |e| \leq \delta \\ \delta(|e| - \frac{1}{2}\delta) & \text{if } |e| > \delta \end{cases}$$

For small errors ($|e| \leq \delta$), behaves like squared error (efficient). For large errors ($|e| > \delta$), switches to absolute error (linear) - downweighting outliers. This makes estimation less sensitive to extreme observations while maintaining efficiency for typical data.

**Part B (4 points):** When would you prefer robust factor estimation over standard MLE? Give a specific example.

**Expected Answer:**

**When to use robust estimation:**
- Data contains outliers from measurement errors, data entry mistakes, or crisis periods
- Heavy-tailed distributions (financial data, commodity prices)
- Structural breaks that create transient outliers
- Small samples where outliers have disproportionate influence

**Example:** Estimating factors from financial returns (stock returns, exchange rates) which have fat tails and occasional extreme movements (crashes, flash crashes). Standard PCA/MLE would be distorted by the 2008 crisis or 2020 pandemic shock. Robust PCA (using Huber or quantile methods) produces factors that represent typical co-movement, less influenced by extreme events.

---

### Question 10 (6 points) - Independent Component Analysis
**Learning Objective:** Understand ICA vs. PCA

Independent Component Analysis (ICA) can be used for factor extraction when innovations are non-Gaussian and independent.

**Question:** How does ICA differ from PCA in terms of (1) objective and (2) identification? Why does ICA require non-Gaussianity?

**Expected Answer:**

**1. Objective:**
- **PCA:** Maximizes variance explained; finds orthogonal (uncorrelated) components
- **ICA:** Maximizes statistical independence; finds components with minimal mutual information

**2. Identification:**
- **PCA:** Factors identified up to orthogonal rotation and sign; not unique
- **ICA:** Under non-Gaussianity, factors are uniquely identified up to permutation and scaling

**Why non-Gaussianity is required:**

Gaussian variables have a critical property: uncorrelated Gaussians are independent. If factors are Gaussian, PCA (which ensures uncorrelatedness) already achieves independence, so ICA has no additional identifying power - any rotation of PCA factors would be equally "independent."

Non-Gaussianity breaks this equivalence. ICA exploits higher-order moments (kurtosis, skewness) to find unique independent components. Intuitively: non-Gaussian distributions have distinctive shapes, allowing ICA to "unmix" them.

---

## Part 4: Machine Learning Connections (20 points)

### Question 11 (6 points) - Autoencoders and Factor Models
**Learning Objective:** Connect neural networks to factor models

An autoencoder neural network consists of:
- **Encoder:** $F_t = g_\text{enc}(X_t; \theta_\text{enc})$ - maps data to low-dimensional representation
- **Decoder:** $\hat{X}_t = g_\text{dec}(F_t; \theta_\text{dec})$ - reconstructs data from representation

Trained to minimize reconstruction error: $\min_{\theta} \sum_t ||X_t - \hat{X}_t||^2$

**Part A (3 points):** How does this relate to a linear factor model?

**Expected Answer:**

A **linear** autoencoder (linear activation functions) is equivalent to Principal Components Analysis / factor model:
- Encoder: $F_t = W_\text{enc} X_t$ (linear projection)
- Decoder: $\hat{X}_t = W_\text{dec} F_t$ (linear reconstruction)

Minimizing reconstruction error finds the same principal components as PCA. The encoder extracts factors, the decoder represents loadings.

**Part B (3 points):** What advantage do nonlinear autoencoders have over standard factor models?

**Expected Answer:**

Nonlinear autoencoders can capture **nonlinear relationships** in data:
- Variables may interact multiplicatively or through threshold effects
- Factor structure may be nonlinear (e.g., factors affecting variables differently in expansions vs. recessions)
- Can discover hierarchical structures (e.g., industry factors loading on sector factors loading on market factor)

This flexibility can improve fit and forecasting when linear assumptions are restrictive, but at the cost of interpretability and risk of overfitting.

---

### Question 12 (8 points) - Deep Learning for Forecasting
**Learning Objective:** Apply ML to economic forecasting

**Scenario:** You are comparing three approaches for forecasting GDP growth:

1. **Classical:** Linear factor model (PCA + regression)
2. **Hybrid:** Factor model + random forest on factors
3. **End-to-end:** Deep neural network (LSTM) directly on raw data

**Part A (4 points):** What is one advantage and one disadvantage of the end-to-end deep learning approach?

**Expected Answer:**

**Advantage:**
- Automatically learns feature representations without manual engineering
- Can capture complex temporal dependencies and nonlinearities
- Jointly optimizes feature extraction and prediction (avoids two-step inefficiency)
- Scales well with large datasets

**Disadvantage:**
- "Black box" - difficult to interpret what the model learned
- Requires large sample size (deep learning needs many observations)
- Prone to overfitting with limited economic data
- No economic interpretation of learned features
- Harder to diagnose failures or understand forecast changes

**Part B (4 points):** You find that in-sample, the LSTM dominates, but out-of-sample, the linear factor model performs better. Explain why and propose a solution.

**Expected Answer:**

**Explanation:** The LSTM is **overfitting** - its flexibility allows it to memorize training data patterns (including noise) rather than learning generalizable relationships. Economic time series have limited sample size (e.g., 60 years of quarterly data = 240 observations), insufficient for training deep networks with thousands of parameters. The in-sample superiority reflects memorization, not true forecasting ability.

**Solutions (any one):**

1. **Regularization:** Add dropout, L1/L2 penalties, early stopping to prevent overfitting
2. **Simpler architecture:** Use shallower networks or fewer parameters
3. **Hybrid approach:** Use factor model for dimension reduction, then LSTM on extracted factors (combines dimension reduction with nonlinearity)
4. **Ensembling:** Average linear factor model and LSTM forecasts to balance bias and variance
5. **Transfer learning:** Pre-train on other countries/variables, fine-tune on target variable
6. **Data augmentation:** If available, pool across related series or use bootstrap

---

### Question 13 (6 points) - Short Answer
**Learning Objective:** Evaluate ML in economics

**Question:** Some researchers argue that machine learning is "fundamentally different" from econometrics, while others view it as a new set of tools for old problems. Taking a stance on this debate, discuss whether ML changes the way we should think about factor models in economics.

**Grading:**
- Clear stance (2 points)
- Economic reasoning (2 points)
- Recognition of trade-offs (2 points)

**Model Answer (one perspective - others acceptable):**

**Stance:** ML is primarily a **new toolkit** rather than fundamentally different - it addresses the same identification and prediction problems with different computational approaches.

**Reasoning:** Both classical factor models and ML aim to reduce dimensionality and extract patterns from data. ML methods (autoencoders, neural networks) can be viewed as **nonlinear extensions** of PCA and factor analysis. The core challenge remains: distinguishing signal from noise, ensuring out-of-sample validity, and interpreting results economically.

**Trade-offs ML introduces:**
- **Pro:** Handles nonlinearity and interactions that linear models miss; scales to massive datasets (text, images)
- **Con:** Sacrifices interpretability and theoretical grounding; requires larger samples; black-box nature conflicts with causal inference needs

**Conclusion:** ML doesn't replace economic theory or causal thinking - we still need structural identification, knowledge of institutional details, and interpretability. ML is most valuable when:
1. Sample size is large (many time periods or cross-sections)
2. Relationships are genuinely nonlinear
3. Pure prediction (not explanation) is the goal

For typical macro applications (small T, need interpretation), classical factor models with targeted ML enhancements (LASSO, random forests for variable selection) may be optimal.

(Alternative stance arguing ML is fundamentally different, emphasizing new objectives like algorithmic fairness or black-box prediction over explanation, is equally valid if well-reasoned)

---

## Part 5: Research Frontiers (15 points)

### Question 14 (5 points) - Text as Data
**Learning Objective:** Understand alternative data sources

Recent research extracts factors from textual data (news articles, Fed speeches, Twitter) using natural language processing.

**Question:** How might text-based factors complement traditional numeric factors in forecasting? Provide two specific examples.

**Expected Answer (any two):**

1. **Sentiment/expectations:** Text captures forward-looking expectations (consumer confidence, business optimism) not reflected in backward-looking hard data. Example: Negative news sentiment predicts consumption drops before retail sales data shows declines.

2. **Policy uncertainty:** Text-based uncertainty indices (Baker et al.) measure political risk, trade policy uncertainty, or geopolitical tensions. Example: Spike in uncertainty language in news predicts investment declines beyond what GDP or financial data show.

3. **Leading indicators:** Earnings call transcripts or Fed minutes reveal forward guidance before policy actions. Example: Hawkish Fed speech text predicts future rate hikes, improving interest rate forecasts.

4. **Granular sectoral information:** Industry-specific news provides sector-level factors. Example: Tech sector sentiment from news predicts tech stock returns and investment patterns.

5. **Real-time availability:** Text is available immediately (no publication lag). Example: Twitter economic sentiment provides nowcasting signal when official data not yet released.

---

### Question 15 (6 points) - High-Frequency Data
**Learning Objective:** Consider measurement frequency

**Scenario:** Traditional factor models use monthly/quarterly data. Now you have access to daily financial data, weekly claims, and some real-time indicators.

**Part A (3 points):** What is one challenge in using high-frequency data for factor extraction?

**Expected Answer:**

- **Noise-to-signal ratio:** High-frequency data has more noise (market microstructure, measurement error) relative to signal. Daily stock returns are 90% noise, 10% information.
- **Mixed frequencies:** Harder to combine daily financial data with monthly real data in a single factor model
- **Curse of dimensionality:** More time points but relationships may differ across frequencies
- **Overfitting:** Easy to find spurious patterns in high-frequency data that don't persist
- **Structural breaks:** High-frequency relationships less stable over time

**Part B (3 points):** How would you design a factor model that leverages high-frequency data without being overwhelmed by noise?

**Expected Answer (approaches):**

1. **Hierarchical factors:** Extract high-frequency factors from daily data, then aggregate (average) to lower frequency to reduce noise before combining with monthly factors

2. **Realized volatility:** Use high-frequency data to construct realized volatility (variance) measures, then use these as inputs to monthly factor model

3. **Mixed-frequency models:** State-space approach with observation equations at different frequencies; use Kalman filter to optimally extract signal

4. **Denoising:** Pre-filter high-frequency data (rolling averages, wavelets) before factor extraction to remove microstructure noise

5. **Targeted extraction:** Use supervised methods (targeted PCA, 3PRF) to extract only components of high-frequency data relevant for lower-frequency outcomes

---

### Question 16 (4 points) - True/False
**Learning Objective:** Recognize research directions

**Statement:** "As machine learning and big data become more prevalent, traditional economic theory and structural models will become less important for forecasting."

**True** / **False**

**Justification (3-4 sentences):**

**Expected Answer:** False (or "False, but with caveats"). While ML improves forecasting in some contexts, economic theory remains essential:

1. **Structural stability:** Theory-based models are more robust to regime changes because they encode causal relationships

2. **Interpretability:** Policy makers need to understand why forecasts changed (new data vs. structural shift)

3. **Data scarcity:** Economic data has limited history; theory provides discipline to avoid overfitting

4. **Counterfactuals:** ML predicts under existing policy; theory is needed for "what-if" scenarios

**Nuance:** Best approach often combines both - use ML for flexible functional forms and dimension reduction, but incorporate theory through choice of variables, timing restrictions, or hybrid models.

(Arguing "True" is acceptable if well-reasoned: e.g., "In pure prediction tasks with abundant data and stable environment, ML can dominate purely theory-based models")

---

## Bonus Question (5 points)

### Question 17 - Synthesis
**Learning Objective:** Integrate module concepts

**Question:** Design a "next-generation" factor model that incorporates at least three advanced techniques from this module. Describe the model, justify each component, and identify one remaining challenge.

**Grading:**
- Three techniques integrated (3 points)
- Coherent justification (1 point)
- Challenge identified (1 point)

**Model Answer (example - many variants acceptable):**

**Model Design: Regime-Switching, Time-Varying Volatility, Text-Augmented Factor Model**

**Specification:**

$$X_t = \Lambda_{s_t} F_t + e_t$$
$$F_t = \Phi_{s_t} F_{t-1} + \sigma_t u_t, \quad u_t \sim t_\nu(0, I)$$
$$\text{Text}_t = \gamma' F_t + v_t$$
$$s_t \sim \text{Markov}(\{1,2\})$$
$$\log(\sigma_t^2) = \mu_{s_t} + \phi \log(\sigma_{t-1}^2) + \eta_t$$

**Components:**

1. **Markov-switching ($s_t$):** Loadings and dynamics differ in expansion vs. recession - captures business cycle asymmetry and structural change

2. **Stochastic volatility ($\sigma_t$):** Time-varying factor volatility captures Great Moderation, crisis periods, and volatility clustering

3. **Text augmentation:** Include news sentiment or uncertainty indices as observable variables loading on factors - adds real-time forward-looking information

**Additional (optional):**
4. **Student-t innovations:** Accommodate fat tails in factor shocks (financial crises, pandemic)

**Justification:** This model addresses key empirical facts:
- Business cycles have discrete regimes (not continuous)
- Volatility varies within regimes (2008 financial crisis vs. 2001 recession)
- Text data provides leading information (expectations, uncertainty)
- Economic shocks are fat-tailed (Student-t more realistic than Gaussian)

**Challenge:** **Computational burden and identification**. Estimating this model requires:
- Filtering over continuous state (factors, volatility) AND discrete state (regime)
- Particle filters or advanced MCMC (computationally intensive)
- Risk of overfitting with many parameters and limited data
- Identification: Is volatility change within-regime or across-regime?

**Solution directions:** Use informative priors (Bayesian), subset of time-varying parameters, or two-step estimation to manage complexity.

---

## Quiz Completion

**Total Points: 105 (100 + 5 bonus)**

### Learning Objectives Coverage

- ✓ Time-varying parameters and stochastic volatility
- ✓ Regime-switching and Markov models
- ✓ Non-Gaussian distributions and robust estimation
- ✓ Machine learning connections (autoencoders, LSTM)
- ✓ Research frontiers (text data, high-frequency data)
- ✓ Integration of advanced techniques

### Submission Instructions

1. Save your answers in a single document (PDF or markdown)
2. Show mathematical derivations where requested
3. For conceptual questions, provide economic intuition
4. Submit via course platform by due date

### Grading Scale

- 90-100: Excellent mastery of advanced topics
- 80-89: Good understanding with minor gaps
- 70-79: Adequate grasp, review extensions
- Below 70: Significant gaps, attend office hours

### Recommended Review Materials

If you scored below 80 on any section:
- **Part 1 (TVP):** Review state-space models, Kalman filter with time-varying parameters
- **Part 2 (Regimes):** Study Hamilton (1989), Markov-switching models
- **Part 3 (Non-Gaussian):** Re-read robust statistics, ICA literature
- **Part 4 (ML):** Work through deep learning tutorials, read Gu et al. (2020)
- **Part 5 (Frontiers):** Survey recent papers in Journal of Econometrics, RFS

### Recommended Papers

Advanced reading for high-performing students:

1. **TVP:** Primiceri (2005) - Time-varying structural VARs
2. **Regime-switching:** Hamilton (1989) - Markov-switching models
3. **Non-Gaussian:** Moneta et al. (2013) - ICA for structural identification
4. **ML connections:** Gu, Kelly, Xiu (2020) - Empirical asset pricing via ML
5. **Text:** Baker, Bloom, Davis (2016) - Economic policy uncertainty
6. **High-frequency:** Todorov & Bollerslev (2010) - Jumps and cojumps

### Computational Resources

- **TVP:** `dlm` (R), `statsmodels` (Python) for state-space models
- **Regime-switching:** `MSwM` (R), `pymc3` (Python)
- **Robust estimation:** `robustbase` (R), `sklearn.covariance` (Python)
- **ML:** `keras`/`tensorflow` (autoencoders), `pytorch` (LSTM)
- **Text:** `quanteda` (R), `nltk`/`spacy` (Python)

### Course Wrap-Up

This module concludes the core content of Dynamic Factor Models. You've progressed from:
- Foundations (static factors, PCA)
- Dynamics (state-space, Kalman filter)
- Extensions (mixed-frequency, FAVAR)
- Sparsity (LASSO, targeted predictors)
- Advanced topics (TVP, regimes, ML)

**Capstone project** now available - apply multiple techniques to a real forecasting problem!

**After submission, review the detailed answer key and reflect on your learning journey.**
