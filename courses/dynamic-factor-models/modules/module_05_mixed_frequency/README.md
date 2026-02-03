# Module 5: Mixed Frequency & Nowcasting

## Overview

This module addresses the challenge of combining data observed at different frequencies—the cornerstone of modern nowcasting applications. You'll learn temporal aggregation schemes, MIDAS regression, state-space mixed-frequency models, techniques for handling ragged-edge data, and implement a complete GDP nowcasting system.

**Estimated Time:** 8-10 hours
**Prerequisites:** Module 2 (state-space), Module 3 or 4 (estimation methods)

## Learning Objectives

By completing this module, you will be able to:

1. **Distinguish** between stock and flow temporal aggregation
2. **Formulate** mixed-frequency DFMs in state-space form
3. **Implement** MIDAS regression for mixed-frequency forecasting
4. **Handle** ragged-edge datasets with publication lags
5. **Construct** a nowcasting model for GDP using monthly indicators
6. **Evaluate** nowcast accuracy across the information release calendar

## Module Contents

### Guides
1. `guides/01_temporal_aggregation.md` - Stock vs flow variables
2. `guides/02_midas_regression.md` - Mixed Data Sampling approach
3. `guides/03_state_space_mixed_freq.md` - Unified DFM framework
4. `guides/04_nowcasting_practice.md` - Real-time forecasting workflow

### Notebooks
1. `notebooks/01_midas_regression.ipynb` - MIDAS for forecasting
2. `notebooks/02_nowcasting_gdp.ipynb` - Complete nowcasting system

### Assessments
- `assessments/quiz_module_05.md` - Conceptual quiz on mixed frequency
- `assessments/mini_project_nowcasting.md` - Build real-time nowcasting model

## Key Concepts

### Temporal Aggregation

**Flow variables** (sum over period):
$$X_t^Q = X_{t,1}^M + X_{t,2}^M + X_{t,3}^M$$

Examples: GDP, retail sales, industrial production

**Stock variables** (end-of-period value):
$$X_t^Q = X_{t,3}^M$$

Examples: Unemployment rate, interest rates, asset prices

**Average variables** (mean over period):
$$X_t^Q = \frac{1}{3}(X_{t,1}^M + X_{t,2}^M + X_{t,3}^M)$$

Examples: Price indices, exchange rates

### MIDAS Regression

**Basic MIDAS specification:**
$$y_t^{(q)} = \beta_0 + \beta_1 \sum_{j=0}^{J-1} w(j; \theta) x_{t-j}^{(m)} + \varepsilon_t$$

where:
- $y_t^{(q)}$: low-frequency target (quarterly GDP)
- $x_t^{(m)}$: high-frequency predictor (monthly indicator)
- $w(j; \theta)$: weighting scheme (e.g., Beta polynomial)

**Beta weighting function:**
$$w(j; \theta_1, \theta_2) = \frac{(1-j/J)^{\theta_1-1}(j/J)^{\theta_2-1}}{\sum_{k=0}^{J-1}(1-k/J)^{\theta_1-1}(k/J)^{\theta_2-1}}$$

Estimate $\theta = (\beta_0, \beta_1, \theta_1, \theta_2)$ by NLS.

### Mixed-Frequency DFM

**State-space representation:**

**Measurement equations:**
$$\begin{aligned}
X_t^{(m)} &= \Lambda^{(m)} F_t + e_t^{(m)} \quad &\text{(monthly)} \\
X_t^{(q)} &= \Lambda^{(q)} \sum_{j=0}^2 F_{t-j} + e_t^{(q)} \quad &\text{(quarterly flow)}
\end{aligned}$$

**Transition equation:**
$$F_t = \Phi_1 F_{t-1} + ... + \Phi_p F_{t-p} + \eta_t$$

**Key insight:** Quarterly flow variable depends on sum of monthly factors within the quarter.

**For stock variables:**
$$X_t^{(q)} = \Lambda^{(q)} F_{t} + e_t^{(q)}$$
(only end-of-quarter factor matters)

### Ragged-Edge Data

**Publication pattern:**
- GDP: Released 30 days after quarter-end
- Monthly indicators: Various lags (0-30 days)
- Creates "ragged edge" of available data

**Kalman filter solution:**
- Missing observations handled naturally
- Incorporates each data release as it arrives
- Updates nowcast in real-time

**Notation:**
$$X_t^{obs} = \{X_{ij}: (i,j) \in \Omega_t\}$$
where $\Omega_t$ = set of available observations at time $t$.

### Nowcasting Framework

**Objective:** Forecast current quarter GDP before official release

**Information set evolution:**
```
Time →  Q1 end    +15 days   +30 days   +45 days
Data:   Feb data  Mar partial Mar final  Q1 GDP
```

**Nowcast updates:**
$$\hat{y}_{Q1|t} = E[y_{Q1} | \Omega_t]$$

As $t$ increases, more monthly indicators become available, improving nowcast accuracy.

### Bridge Equations

**Simple approach:** Regress GDP on same-quarter monthly indicators

$$y_t^Q = \alpha + \sum_{i=1}^N \beta_i \bar{x}_{i,t}^Q + \varepsilon_t$$

where $\bar{x}_{i,t}^Q$ = average of monthly indicator $i$ in quarter $t$.

**Ragged-edge handling:** Replace missing values with forecasts or backfill.

**Limitation:** Ignores factor structure and cross-series information.

## Nowcasting Project: GDP Nowcasting System

**Objective:** Build operational nowcasting model for US GDP

**Data:**
- Target: Real GDP growth (quarterly)
- Predictors: 30 monthly indicators from FRED
  - Hard data: Industrial production, retail sales, employment
  - Soft data: Surveys, sentiment indices
  - Financial: Interest rates, spreads, equity returns

**Requirements:**
1. Download and clean mixed-frequency data
2. Handle ragged-edge publication patterns
3. Estimate mixed-frequency DFM via Kalman filter
4. Generate nowcasts as data arrives within quarter
5. Evaluate out-of-sample accuracy
6. Visualize nowcast evolution over information release calendar

**Deliverable:** Complete nowcasting pipeline with documentation

**Evaluation Rubric:**
- Data handling: 20%
- Model specification: 25%
- Implementation correctness: 25%
- Evaluation methodology: 15%
- Visualization and interpretation: 15%

## Connections to Other Modules

| Module | Connection |
|--------|------------|
| Module 2 | State-space framework handles MF naturally |
| Module 3 | PCA on monthly data for factor initialization |
| Module 4 | ML estimation with MF measurement equations |
| Module 6 | Nowcasts are short-term forecasts |
| Module 7 | Real-time data vintages |

## Key Formulas

### Quarterly Flow Aggregation in State-Space

**Monthly factors:** $F_1, F_2, F_3$ (within quarter)

**Quarterly observation:**
$$X_Q = \lambda_Q (F_1 + F_2 + F_3) + e_Q$$

**State vector augmentation:**
$$\alpha_t = \begin{bmatrix} F_t \\ F_{t-1} \\ F_{t-2} \end{bmatrix}$$

**Measurement matrix for quarterly flow:**
$$Z^Q = [\lambda_Q \quad \lambda_Q \quad \lambda_Q]$$

### MIDAS Forecast

**h-step ahead forecast:**
$$\hat{y}_{t+h}^{(q)} = \hat{\beta}_0 + \hat{\beta}_1 \sum_{j=0}^{J-1} w(j; \hat{\theta}) \hat{x}_{t+h-j}^{(m)}$$

where $\hat{x}_{t+h-j}^{(m)}$ are forecasts from ARIMA model if $t+h-j > t_{now}$.

### Nowcast Revision

**Mean squared error:**
$$MSE_t = E[(y_Q - \hat{y}_{Q|t})^2 | \Omega_t]$$

**Nowcast variance (from Kalman filter):**
$$\text{Var}(\hat{y}_{Q|t}) = \lambda_Q' P_{F|t} \lambda_Q + \sigma_e^2$$

where $P_{F|t}$ = filtered covariance of factors.

### Bridge Equation with Soft Indicators

$$\Delta y_t^Q = \alpha + \beta_1 \Delta IP_t^Q + \beta_2 PMI_t^Q + \beta_3 CS_t^Q + \varepsilon_t$$

where:
- $IP$: Industrial production (hard)
- $PMI$: Purchasing Managers Index (soft)
- $CS$: Consumer sentiment (soft)

## Reading List

### Required
- Mariano, R.S. & Murasawa, Y. (2003). "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *Journal of Applied Econometrics* 18(4), 427-443.
- Bańbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013). "Now-Casting and the Real-Time Data Flow." *Handbook of Economic Forecasting* Vol. 2, 195-237.

### Recommended
- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). "The MIDAS Touch: Mixed Data Sampling Regression Models." *UCLA Working Paper*.
- Foroni, C., Marcellino, M., & Schumacher, C. (2015). "Unrestricted Mixed Data Sampling (MIDAS): MIDAS Regressions with Unrestricted Lag Polynomials." *Journal of the Royal Statistical Society A* 178(1), 57-82.

### Applications
- Giannone, D., Reichlin, L., & Small, D. (2008). "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics* 55(4), 665-676.
- Bok, B., Caratelli, D., Giannone, D., Sbordone, A.M., & Tambalotti, A. (2018). "Macroeconomic Nowcasting and Forecasting with Big Data." *Annual Review of Economics* 10, 615-643.

## Practical Applications

After completing this module, you can:

1. **Build nowcasting models** for GDP, inflation, employment
2. **Evaluate information content** of soft vs hard indicators
3. **Quantify news effects** from each data release
4. **Handle irregular publication schedules** systematically
5. **Construct fan charts** showing nowcast uncertainty evolution
6. **Compare MIDAS vs DFM approaches** empirically

## Real-World Nowcasting Systems

**New York Fed Nowcast:**
- Uses mixed-frequency DFM with 30+ indicators
- Updated continuously as data releases arrive
- Public-facing: nyfed.org/nowcast

**Atlanta Fed GDPNow:**
- Bridge equation approach with detailed GDP components
- Updates daily
- Public-facing: frbatlanta.org/gdpnow

**ECB/Euro Area Nowcast:**
- Mixed-frequency DFM for euro area GDP
- Handles cross-country data

**Your implementation** will follow similar methodology!

## Common Pitfalls

1. **Wrong aggregation:** Using stock formula for flow variables
2. **Ignoring publication lags:** Treating all data as synchronized
3. **Overfitting:** Too many monthly indicators for short GDP sample
4. **Forgetting transformations:** Mixing levels and growth rates
5. **Out-of-sample evaluation errors:** Using future information
6. **MIDAS numerical issues:** Sensitive to starting values

## Computational Considerations

**State augmentation:**
- Monthly model: State dimension $= r \times p$
- With quarterly: State dimension $= r \times (p+2)$ (for flows)
- Kalman filter computational cost increases with state dimension

**Estimation strategies:**
1. Two-step: Estimate monthly DFM, then add quarterly variables
2. Joint ML: Maximize full MF likelihood (more efficient but slower)
3. Bayesian: Prior on aggregation scheme parameters

**Software:**
- statsmodels: `DynamicFactorMQ` (mixed-frequency)
- midaspy: MIDAS regression package
- nowcasting: R package (excellent but needs translation)

## Data Sources

**US Macroeconomic Data:**
- FRED-MD: 128 monthly indicators (fred.stlouisfed.org)
- GDP: BEA quarterly (apps.bea.gov)
- Real-time vintages: ALFRED (alfred.stlouisfed.org)

**Survey Data:**
- ISM PMI (monthly)
- University of Michigan Consumer Sentiment (monthly)
- Conference Board Consumer Confidence (monthly)

**Financial Data:**
- Daily: Convert to monthly average or end-of-month
- Useful for timely signals

## Extension: Daily to Monthly Aggregation

**Challenge:** Use daily financial data to predict monthly macroeconomic variables

**Approach:** Stock-to-flow specification
$$X_t^{(m)} = \sum_{d \in \text{month } t} \lambda_d F_d + e_t^{(m)}$$

**Implementation:** Further state augmentation (20-23 daily factors per month)

**Applications:**
- High-frequency macro forecasting
- Measuring information flow from markets to economy

## Next Steps

After completing this module:
1. Complete GDP nowcasting project
2. Verify understanding with conceptual quiz
3. Explore real-time data vintages (Module 7)
4. Proceed to Module 6: Forecasting with Factors

---

*"Nowcasting is the art of predicting the present. By cleverly combining high-frequency indicators with state-space methods, we can substantially reduce the 30-45 day information lag for GDP—critical for real-time policy decisions." - Adapted from Giannone, Reichlin & Small*
