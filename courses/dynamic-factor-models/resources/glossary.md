# Glossary of Key Terms

## A

### Approximate Factor Model
A factor model where idiosyncratic errors are allowed to be weakly correlated across units, in contrast to exact factor models requiring strict independence. Formalized by Chamberlain & Rothschild (1983) and central to large-N asymptotics.

### Autoregressive (AR) Process
A time series model where current values depend linearly on past values: $y_t = \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \varepsilon_t$.

## B

### Bai-Ng Criteria
Information criteria for determining the number of factors in approximate factor models. Include IC1, IC2, IC3, PC1, PC2, PC3 with different penalty terms. See Bai & Ng (2002).

### Bridge Equation
A regression linking high-frequency indicators to a lower-frequency target variable. Used in nowcasting to translate monthly data into quarterly GDP forecasts.

## C

### Common Component
The portion of variation in a variable explained by common factors: $\chi_{it} = \lambda_i' F_t$. Contrasts with idiosyncratic component.

### Communality
The proportion of a variable's variance explained by common factors. Equals $1 - \psi_i$ where $\psi_i$ is the idiosyncratic variance.

## D

### Diffusion Index
A summary measure combining information from many indicators, typically the first few principal components of a large panel. Used for forecasting in Stock & Watson (2002).

### Dynamic Factor Model (DFM)
A factor model where factors follow a vector autoregressive process: $F_t = \Phi F_{t-1} + \eta_t$. Captures temporal dynamics in latent factors.

## E

### EM Algorithm
Expectation-Maximization algorithm for maximum likelihood estimation with latent variables. E-step computes expected sufficient statistics (via Kalman smoother); M-step updates parameters.

### Exact Factor Model
A factor model where idiosyncratic errors are mutually independent across all units and time periods. Restrictive for large panels.

## F

### Factor Augmented VAR (FAVAR)
A VAR model that includes estimated factors as additional variables: $[F_t', Y_t']' = \Phi(L)[F_{t-1}', Y_{t-1}']' + u_t$. Enables structural analysis with dimension reduction.

### Factor Loadings
The coefficients $\Lambda = [\lambda_1, ..., \lambda_N]'$ relating observed variables to latent factors. Element $\lambda_{ij}$ measures sensitivity of variable $i$ to factor $j$.

### Factor Scores
Estimates of the latent factors $F_t$ for each time period. Can be computed via regression, Bartlett method, or Kalman filtering/smoothing.

### FRED-MD
McCracken-Ng Monthly Database of approximately 127 monthly U.S. macroeconomic time series, designed for factor model analysis. Updated monthly by Federal Reserve Bank of St. Louis.

## G

### Generalized Principal Components
Extension of PCA accounting for heteroskedasticity or correlation in idiosyncratic errors. Weights observations by inverse covariance.

## H

### Heterogeneous Loadings
Factor loadings that vary across units, allowing different variables to respond differently to common factors.

## I

### Identification
The problem of uniquely determining model parameters from data. Factor models require normalization constraints (e.g., $F'F/T = I$, $\Lambda'\Lambda$ diagonal) to resolve rotational indeterminacy.

### Idiosyncratic Component
The portion of variation specific to each variable, not explained by common factors: $e_{it} = X_{it} - \lambda_i' F_t$.

### Information Criteria
Penalized likelihood measures for model selection. In factor models, used to determine number of factors. Include AIC, BIC, and Bai-Ng criteria.

## K

### Kalman Filter
Recursive algorithm for computing filtered state estimates $E[F_t | X_1, ..., X_t]$ in state-space models. Produces one-step-ahead predictions and their variances.

### Kalman Smoother
Algorithm for computing smoothed state estimates $E[F_t | X_1, ..., X_T]$ using all available data. Rauch-Tung-Striebel (RTS) smoother is standard.

## L

### LASSO (Least Absolute Shrinkage and Selection Operator)
Penalized regression with L1 penalty: $\min_\beta \|y - X\beta\|^2 + \lambda \|\beta\|_1$. Produces sparse solutions for variable selection.

### Latent Variable
An unobserved variable inferred from observed data. In factor models, the factors $F_t$ are latent.

### Loading Matrix
The matrix $\Lambda$ of factor loadings relating all observed variables to all factors.

## M

### Maximum Likelihood Estimation (MLE)
Estimation by maximizing the likelihood function. For DFMs, computed via Kalman filter prediction error decomposition.

### MIDAS (Mixed Data Sampling)
Regression framework for relating variables observed at different frequencies. Uses distributed lag polynomials with restricted weights.

### Missing at Random (MAR)
Data are missing at random if missingness depends only on observed data, not on the missing values themselves. Required assumption for EM-based imputation.

### Mixed Frequency
Data where different series are observed at different frequencies (e.g., monthly indicators, quarterly GDP). Requires special modeling approaches.

## N

### Nowcasting
Real-time prediction of current-period values before official data release. Combines high-frequency indicators to estimate current economic conditions.

### Number of Factors
The dimension $r$ of the factor space. Determining $r$ is crucial; common approaches include scree plots, information criteria, and eigenvalue ratios.

## O

### Orthogonal Factors
Factors that are uncorrelated with each other: $E[F_t F_t'] = I$. Standard normalization in many factor models.

## P

### Partial Pooling
Bayesian approach where parameters are drawn from common distributions, sharing information across units while allowing heterogeneity.

### Prediction Error Decomposition
Expressing the likelihood as product of conditional densities: $\mathcal{L} = \prod_t p(X_t | X_{t-1}, ..., X_1)$. Kalman filter provides these conditional distributions.

### Principal Components (PC)
Orthogonal linear combinations of variables maximizing explained variance. First PC explains most variance, second PC (orthogonal to first) explains most remaining variance, etc.

## Q

### Quasi-Maximum Likelihood (QML)
Maximum likelihood estimation treating estimated factors as known in second-stage regression. May require standard error corrections.

## R

### R² Ratio
Measure of factor model fit: proportion of total variance explained by common factors.

### Ragged Edge
The pattern of missing observations at the end of a dataset due to different publication lags across series. Common in real-time nowcasting.

### Rotation
Transformation of factors and loadings that preserves fit: $(\Lambda, F) \to (\Lambda R, R^{-1} F)$ for any invertible $R$. Varimax, promax are common rotation methods.

## S

### Scree Plot
Graph of eigenvalues in descending order. "Elbow" in plot suggests number of factors.

### State-Space Model
Model with latent state evolving over time: measurement equation $X_t = Z F_t + e_t$, transition equation $F_t = T F_{t-1} + \eta_t$.

### Stock-Watson Estimator
Two-step estimator: (1) extract factors via PCA, (2) use factors in forecasting regression. Consistent as $N, T \to \infty$.

### Strong Factor
A factor with loadings of order $O(1)$ affecting most variables in the panel. Pervasive factors in Bai-Ng terminology.

## T

### Targeted Predictors
Variable selection approach choosing predictors with strong relationship to target variable before factor extraction. See Bai & Ng (2008).

### Three-Pass Regression Filter
Method combining factor extraction with targeted selection: (1) regress each predictor on target, (2) form factor from selected predictors, (3) forecast with factors.

### Time-Varying Parameters
Model parameters (loadings, factor dynamics) that change over time. Captures structural change and evolving relationships.

## U

### Unbalanced Panel
Panel data where not all series are observed for all time periods. Common in macroeconomic applications due to different series start dates.

## V

### Varimax Rotation
Orthogonal rotation maximizing variance of squared loadings within factors. Produces "simple structure" with loadings close to 0 or 1.

### Vintage Data
Historical data as it was available at a specific point in time, before revisions. Essential for proper real-time forecast evaluation.

## W

### Weak Factor
A factor with loadings of order $O(N^{-1/2})$ affecting few variables. May not be consistently estimable with PCA.

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $X_t$ | $N \times 1$ vector of observed variables at time $t$ |
| $F_t$ | $r \times 1$ vector of latent factors at time $t$ |
| $\Lambda$ | $N \times r$ matrix of factor loadings |
| $e_t$ | $N \times 1$ vector of idiosyncratic errors |
| $\Phi$ | $r \times r$ factor dynamics matrix (VAR coefficients) |
| $\Sigma_e$ | $N \times N$ idiosyncratic covariance matrix |
| $\Sigma_\eta$ | $r \times r$ factor innovation covariance |
| $T$ | Number of time periods |
| $N$ | Number of observed series |
| $r$ | Number of factors |

---

## Key References

- Bai, J. & Ng, S. (2002). "Determining the Number of Factors in Approximate Factor Models." *Econometrica* 70(1): 191-221.
- Bai, J. & Ng, S. (2008). "Forecasting Economic Time Series Using Targeted Predictors." *Journal of Econometrics* 146(2): 304-317.
- Stock, J.H. & Watson, M.W. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *JASA* 97(460): 1167-1179.
- Giannone, D., Reichlin, L. & Small, D. (2008). "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics* 55(4): 665-676.
