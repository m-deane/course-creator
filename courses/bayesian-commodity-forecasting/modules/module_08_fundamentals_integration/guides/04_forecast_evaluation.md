# Forecast Evaluation for Bayesian Models

> **Reading time:** ~11 min | **Module:** 8 — Fundamentals Integration | **Prerequisites:** Modules 1-7


## In Brief

Forecast evaluation assesses prediction quality using both point forecast accuracy (MAE, RMSE) and probabilistic calibration (coverage, sharpness, proper scoring rules). Bayesian models provide full predictive distributions, enabling evaluation beyond simple point forecasts to measure whether uncertainty quantification is accurate.

<div class="callout-insight">
<strong>Insight:</strong> A forecast is not just a number—it's a probability distribution. Evaluating only the mean (MAE, RMSE) ignores uncertainty quantification. A good Bayesian forecast has: (1) accurate central tendency (low MAE), (2) well-calibrated uncertainty (95% intervals contain 95% of outcomes), (3) sharp distributions (not overly conservative). Proper scoring rules like CRPS evaluate the entire predictive distribution.
</div>

## Formal Definition

### Point Forecast Metrics

Given true values $\{y_t\}$ and forecasts $\{\hat{y}_t\}$ for $t = 1, ..., T$:
<div class="callout-key">
<strong>Key Point:</strong> Given true values $\{y_t\}$ and forecasts $\{\hat{y}_t\}$ for $t = 1, ..., T$:
</div>


**Mean Absolute Error (MAE):**
$$\text{MAE} = \frac{1}{T} \sum_{t=1}^T |y_t - \hat{y}_t|$$

**Root Mean Squared Error (RMSE):**
$$\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^T (y_t - \hat{y}_t)^2}$$

**Mean Absolute Percentage Error (MAPE):**
$$\text{MAPE} = \frac{100}{T} \sum_{t=1}^T \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$

Properties:
- MAE: Robust to outliers, units of original data
- RMSE: Penalizes large errors more, units of original data
- MAPE: Scale-free but undefined when $y_t = 0$

### Probabilistic Forecast Metrics

**Predictive distribution:** $p(y_t | \mathcal{D}_{1:t-1})$

**Coverage (Calibration):**
For α-level prediction interval $[L_t, U_t]$:
$$\text{Coverage} = \frac{1}{T} \sum_{t=1}^T \mathbb{1}(L_t \leq y_t \leq U_t)$$

Should equal $1 - \alpha$ (e.g., 95% intervals should contain 95% of observations).

**Sharpness (Precision):**
$$\text{Sharpness} = \frac{1}{T} \sum_{t=1}^T (U_t - L_t)$$

Narrower intervals are sharper (more informative), but only good if well-calibrated.

**Continuous Ranked Probability Score (CRPS):**
$$\text{CRPS}(F_t, y_t) = \int_{-\infty}^{\infty} (F_t(x) - \mathbb{1}(x \geq y_t))^2 dx$$

Where $F_t$ is the predictive CDF.

For Gaussian predictive: $\mathcal{N}(\mu_t, \sigma_t^2)$:
$$\text{CRPS} = \sigma_t \left[\frac{1}{\sqrt{\pi}} - 2\phi(z) - z(2\Phi(z) - 1)\right]$$

where $z = (y_t - \mu_t)/\sigma_t$, $\phi$ is standard normal PDF, $\Phi$ is CDF.

**Log Predictive Density (LPD):**
$$\text{LPD} = \frac{1}{T} \sum_{t=1}^T \log p(y_t | \mathcal{D}_{1:t-1})$$

Higher is better. Proper scoring rule rewarding both accuracy and calibration.

### Diebold-Mariano Test

Test if forecast 1 is significantly better than forecast 2:

**Null hypothesis:** $\mathbb{E}[L(e_t^{(1)}) - L(e_t^{(2)})] = 0$

Where $L$ is loss function (e.g., squared error), $e_t^{(i)}$ is forecast error.

**Test statistic:**
$$\text{DM} = \frac{\bar{d}}{\sqrt{\text{Var}(\bar{d})/T}} \sim t_{T-1}$$

where $\bar{d} = \frac{1}{T}\sum_t (L(e_t^{(1)}) - L(e_t^{(2)}))$

## Intuitive Explanation

Think of weather forecasts:

**Point forecast only:**
- Forecast: "65°F tomorrow"
- Actual: 63°F
- Error: 2°F (good!)
- But: Was 65°F certain or uncertain?

**Probabilistic forecast:**
- Forecast: "65°F ± 10°F (95% interval)"
- Actual: 63°F
- Check: Within interval? ✓
- Check: Is interval reasonable width? ✓

**Bad probabilistic forecast:**
- Forecast: "65°F ± 50°F (95% interval)"
- Actual: 63°F
- Within interval? ✓
- But: Too wide! Not useful. (Poor sharpness)

For commodity forecasts:

**Good forecast:**
- Mean: $75/barrel
- 95% interval: [$70, $80]
- Actual: $76
- ✓ Within interval
- ✓ Interval reasonably narrow
- ✓ CRPS low

**Overconfident forecast:**
- Mean: $75
- 95% interval: [$74, $76] (too narrow!)
- Actual: $78
- ✗ Outside interval
- Uncertainty underestimated

**Evaluation checks:**
1. Is mean forecast accurate? (MAE, RMSE)
2. Are intervals well-calibrated? (Coverage)
3. Are intervals informative? (Sharpness)
4. Is full distribution correct? (CRPS, LPD)

## Code Implementation

### Comprehensive Forecast Evaluation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import erf
import arviz as az

class ForecastEvaluator:
    """
    Comprehensive evaluation of Bayesian forecasts.
    """

    def __init__(self, y_true, predictive_samples=None, predictive_mean=None,
                 predictive_std=None, intervals=None):
        """
        Args:
            y_true: True values [T]
            predictive_samples: Posterior predictive samples [n_samples, T]
                               (if available)
            predictive_mean: Predictive mean [T] (if samples not available)
            predictive_std: Predictive std [T] (if samples not available)
            intervals: Dict of prediction intervals {alpha: (lower, upper)}
        """
        self.y_true = np.array(y_true)
        self.T = len(y_true)

        if predictive_samples is not None:
            self.samples = predictive_samples
            self.mean = predictive_samples.mean(axis=0)
            self.std = predictive_samples.std(axis=0)
        else:
            self.samples = None
            self.mean = np.array(predictive_mean)
            self.std = np.array(predictive_std) if predictive_std is not None else None

        self.intervals = intervals if intervals is not None else {}

    def point_forecast_metrics(self):
        """
        Compute point forecast accuracy metrics.
        """
        errors = self.y_true - self.mean
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2

        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(squared_errors))
        mape = np.mean(np.abs(errors / self.y_true)) * 100  # Assumes y_true != 0

        # Median absolute error (robust alternative)
        medae = np.median(abs_errors)

        # Mean absolute scaled error (MASE) - requires training data
        # Simplified: use naive forecast as benchmark
        naive_errors = np.abs(np.diff(self.y_true))
        mae_naive = np.mean(naive_errors)
        mase = mae / mae_naive if mae_naive > 0 else np.inf

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MedAE': medae,
            'MASE': mase
        }

    def probabilistic_metrics(self):
        """
        Compute probabilistic forecast metrics.
        """
        metrics = {}

        # Coverage for different confidence levels
        if self.intervals:
            for alpha, (lower, upper) in self.intervals.items():
                coverage = np.mean((self.y_true >= lower) & (self.y_true <= upper))
                width = np.mean(upper - lower)

                metrics[f'coverage_{int((1-alpha)*100)}%'] = coverage
                metrics[f'width_{int((1-alpha)*100)}%'] = width

        # CRPS (if predictive distribution available)
        if self.std is not None:
            crps_values = self._compute_crps_gaussian(
                self.y_true, self.mean, self.std
            )
            metrics['CRPS'] = np.mean(crps_values)

        # Log predictive density
        if self.std is not None:
            lpd_values = stats.norm.logpdf(self.y_true, self.mean, self.std)
            metrics['LPD'] = np.mean(lpd_values)

        # Prediction interval score (PIS)
        if 0.05 in self.intervals:  # 95% interval
            lower, upper = self.intervals[0.05]
            pis = self._compute_pis(self.y_true, lower, upper, alpha=0.05)
            metrics['PIS_95'] = np.mean(pis)

        return metrics

    def _compute_crps_gaussian(self, y, mu, sigma):
        """
        CRPS for Gaussian predictive distribution.
        """
        z = (y - mu) / sigma
        pdf_z = stats.norm.pdf(z)
        cdf_z = stats.norm.cdf(z)

        crps = sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1/np.sqrt(np.pi))
        return crps

    def _compute_pis(self, y, lower, upper, alpha=0.05):
        """
        Prediction Interval Score (Gneiting & Raftery 2007).

        PIS = width + (2/alpha) * penalties

        Penalizes: (1) wide intervals, (2) observations outside interval
        """
        width = upper - lower

        # Penalty for being below interval
        penalty_lower = np.where(y < lower, (2/alpha) * (lower - y), 0)

        # Penalty for being above interval
        penalty_upper = np.where(y > upper, (2/alpha) * (y - upper), 0)

        pis = width + penalty_lower + penalty_upper
        return pis

    def calibration_plot(self, n_bins=10):
        """
        Plot calibration curve (PIT histogram).

        Probability Integral Transform (PIT):
        If forecast well-calibrated, PIT ~ Uniform(0, 1)
        """
        if self.std is None:
            print("Cannot compute PIT without predictive distribution")
            return

        # Compute PIT values
        pit = stats.norm.cdf(self.y_true, self.mean, self.std)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: PIT histogram
        axes[0].hist(pit, bins=n_bins, density=True, alpha=0.7, edgecolor='black')
        axes[0].axhline(1.0, color='red', linestyle='--', linewidth=2,
                       label='Uniform (ideal)')
        axes[0].set_xlabel('PIT Value')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Probability Integral Transform (PIT) Histogram')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot 2: Coverage by confidence level
        alphas = np.linspace(0.01, 0.99, 50)
        coverages = []

        for alpha in alphas:
            z_score = stats.norm.ppf(alpha)
            lower = self.mean - z_score * self.std
            upper = self.mean + z_score * self.std
            coverage = np.mean((self.y_true >= lower) & (self.y_true <= upper))
            coverages.append(coverage)

        axes[1].plot(alphas, coverages, linewidth=2, label='Actual coverage')
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
        axes[1].set_xlabel('Nominal Coverage Level')
        axes[1].set_ylabel('Empirical Coverage')
        axes[1].set_title('Coverage Calibration Curve')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def residual_diagnostics(self):
        """
        Residual analysis for forecast errors.
        """
        residuals = self.y_true - self.mean
        standardized = residuals / self.std if self.std is not None else residuals

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Residuals over time
        axes[0, 0].plot(residuals, linewidth=1, alpha=0.7)
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].fill_between(range(self.T),
                                -2*self.std if self.std is not None else -2,
                                2*self.std if self.std is not None else 2,
                                alpha=0.2, color='red', label='±2σ')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: QQ plot
        stats.probplot(standardized if self.std is not None else residuals,
                       dist='norm', plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        axes[0, 1].grid(alpha=0.3)

        # Plot 3: Histogram
        axes[1, 0].hist(standardized if self.std is not None else residuals,
                       bins=30, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(-4, 4, 100)
        axes[1, 0].plot(x, stats.norm.pdf(x), 'r-', linewidth=2,
                       label='N(0,1)')
        axes[1, 0].set_xlabel('Standardized Residual')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Plot 4: ACF of residuals
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, lags=20, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title('Autocorrelation of Residuals')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def summary_report(self):
        """
        Comprehensive evaluation report.
        """
        point_metrics = self.point_forecast_metrics()
        prob_metrics = self.probabilistic_metrics()

        print("=" * 70)
        print("FORECAST EVALUATION REPORT")
        print("=" * 70)

        print("\nPOINT FORECAST ACCURACY:")
        print("-" * 70)
        for metric, value in point_metrics.items():
            print(f"  {metric:15s}: {value:10.4f}")

        print("\nPROBABILISTIC FORECAST QUALITY:")
        print("-" * 70)
        for metric, value in prob_metrics.items():
            if 'coverage' in metric:
                target = int(metric.split('_')[1].replace('%', '')) / 100
                status = "✓" if abs(value - target) < 0.05 else "✗"
                print(f"  {metric:15s}: {value:10.4f} (target: {target:.2f}) {status}")
            else:
                print(f"  {metric:15s}: {value:10.4f}")

        print("=" * 70)


# Example: Evaluate commodity price forecasts
np.random.seed(42)

# Simulate true prices
T = 100
y_true = 50 + np.cumsum(np.random.normal(0, 1, T))

# Simulate forecasts (with some error)
forecast_mean = y_true + np.random.normal(0, 0.5, T)
forecast_std = np.abs(np.random.normal(2, 0.5, T))

# Generate samples (for illustration)
forecast_samples = np.random.normal(
    forecast_mean[np.newaxis, :],
    forecast_std[np.newaxis, :],
    size=(1000, T)
)

# Compute prediction intervals
lower_95 = forecast_mean - 1.96 * forecast_std
upper_95 = forecast_mean + 1.96 * forecast_std

lower_50 = forecast_mean - 0.674 * forecast_std
upper_50 = forecast_mean + 0.674 * forecast_std

intervals = {
    0.05: (lower_95, upper_95),
    0.50: (lower_50, upper_50)
}

# Create evaluator
evaluator = ForecastEvaluator(
    y_true,
    predictive_samples=forecast_samples,
    intervals=intervals
)

# Generate report
evaluator.summary_report()

# Generate plots
fig1 = evaluator.calibration_plot()
plt.savefig('calibration_plot.png', dpi=150, bbox_inches='tight')

fig2 = evaluator.residual_diagnostics()
plt.savefig('residual_diagnostics.png', dpi=150, bbox_inches='tight')

plt.show()
```

</div>
</div>

### Comparing Multiple Forecasts

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
def compare_forecasts(y_true, forecasts_dict):
    """
    Compare multiple competing forecasts.

    Args:
        y_true: True values
        forecasts_dict: {name: (mean, std)} for each forecast

    Returns:
        comparison_df: Comparison table
    """
    results = []

    for name, (mean, std) in forecasts_dict.items():
        evaluator = ForecastEvaluator(y_true, predictive_mean=mean,
                                      predictive_std=std)

        point = evaluator.point_forecast_metrics()
        prob = evaluator.probabilistic_metrics()

        results.append({
            'Model': name,
            'MAE': point['MAE'],
            'RMSE': point['RMSE'],
            'CRPS': prob.get('CRPS', np.nan),
            'LPD': prob.get('LPD', np.nan)
        })

    df = pd.DataFrame(results)
    df = df.sort_values('CRPS')  # Rank by CRPS

    return df


# Example: Compare three models
forecasts = {
    'ARIMA': (forecast_mean, forecast_std),
    'Bayesian GP': (forecast_mean + np.random.normal(0, 0.3, T),
                    forecast_std * 1.1),
    'Fundamental': (forecast_mean + np.random.normal(0, 0.5, T),
                   forecast_std * 0.9)
}

comparison = compare_forecasts(y_true, forecasts)

print("\nModel Comparison:")
print(comparison.to_string(index=False))

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison))
width = 0.35

ax.bar(x - width/2, comparison['MAE'], width, label='MAE', alpha=0.7)
ax.bar(x + width/2, comparison['CRPS'], width, label='CRPS', alpha=0.7)

ax.set_ylabel('Error Metric')
ax.set_title('Forecast Comparison')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Model'])
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('forecast_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>
</div>

## Common Pitfalls

**1. Using Only Point Forecast Metrics**
- **Problem:** Evaluating Bayesian forecasts with MAE/RMSE alone
- **Symptom:** Missing poor calibration (overconfident predictions)
- **Solution:** Always evaluate probabilistic aspects (coverage, CRPS)

**2. In-Sample Evaluation**
- **Problem:** Testing forecasts on training data
- **Symptom:** Overly optimistic performance
- **Solution:** Use out-of-sample evaluation, rolling windows, or cross-validation

**3. Not Checking Calibration**
- **Problem:** Ignoring whether 95% intervals actually contain 95% of data
- **Symptom:** Miscalibrated uncertainty → poor decisions
- **Solution:** Plot calibration curves, check coverage at multiple levels

**4. Scale-Dependent Metrics**
- **Problem:** Comparing MAE across commodities with different price levels
- **Symptom:** Can't compare oil ($70) to gold ($1800) forecasts
- **Solution:** Use scale-free metrics (MAPE, MASE) or normalize

**5. Ignoring Temporal Dependence**
- **Problem:** Treating forecast errors as independent
- **Symptom:** Underestimated standard errors in tests
- **Solution:** Use Diebold-Mariano test with HAC standard errors

## Connections

**Builds on:**
- Module 8.3: BMA (comparing/combining model forecasts)
- Module 6: MCMC (generating predictive distributions)
- Time series forecasting (rolling windows, expanding windows)

**Leads to:**
- Trading strategies (forecast skill → position sizing)
- Risk management (calibrated intervals → VaR)
- Model improvement (diagnostic feedback loop)

**Related topics:**
- Online learning (sequential forecast evaluation)
- Forecast combination (optimal weights from evaluation)
- Decision theory (loss functions, utilities)

## Practice Problems

1. **Coverage Calculation**
   100 forecasts with 95% prediction intervals
   92 actuals fall within intervals

   Is this well-calibrated? What's the problem?

2. **CRPS Interpretation**
   Two forecasts for same data:
   - Model A: CRPS = 2.5
   - Model B: CRPS = 3.1

   Which model is better? By how much?

3. **Sharpness vs Calibration Trade-off**
   - Model 1: 95% coverage = 0.95, mean width = 5
   - Model 2: 95% coverage = 0.95, mean width = 10

   Which is better? Why?

4. **Diebold-Mariano Test**
   Forecast errors: e₁ = [2, -1, 3, -2, 1], e₂ = [3, -2, 4, -1, 2]
   Loss: Squared error

   Compute: d_t = e₁² - e₂²
   Test if mean(d) = 0

5. **Commodity Application**
   Evaluating oil price forecasts:
   - MAE = $2.50/barrel
   - 95% coverage = 0.89 (target: 0.95)

   What's the problem? How to fix the model?


---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving forecast evaluation for bayesian models, what would be your first three steps to apply the techniques from this guide?
</div>

## Further Reading

**Forecast Evaluation Theory:**
1. **Gneiting & Raftery (2007)** - "Strictly Proper Scoring Rules" - CRPS, log score
2. **Diebold & Mariano (1995)** - "Comparing Predictive Accuracy" - DM test
3. **Dawid (1984)** - "Statistical Theory: The Prequential Approach" - Calibration

**Proper Scoring Rules:**
4. **Gneiting & Katzfuss (2014)** - "Probabilistic Forecasting" - Comprehensive review
5. **Krueger et al. (2021)** - "Predictive Inference Based on Markov Chain Monte Carlo Output" - Bayesian forecasting

**Calibration:**
6. **Gneiting et al. (2007)** - "Probabilistic Forecasts, Calibration and Sharpness" - PIT, coverage
7. **Hamill (2001)** - "Interpretation of Rank Histograms for Verifying Ensemble Forecasts" - PIT for ensembles

**Economic/Commodity Applications:**
8. **Baumeister & Kilian (2014)** - "Real-Time Forecasts of the Real Price of Oil" - Oil forecast evaluation
9. **Clark & McCracken (2013)** - "Advances in Forecast Evaluation" - Econometric methods

**Software:**
10. **ArviZ Documentation** - Bayesian forecast evaluation tools
11. **scoringRules R package** - Proper scoring rules implementation


<div class="callout-key">
<strong>Key Concept Summary:</strong> Forecast evaluation assesses prediction quality using both point forecast accuracy (MAE, RMSE) and probabilistic calibration (coverage, sharpness, proper scoring rules).
</div>

---

*"Forecast evaluation should assess both accuracy and uncertainty. A perfect mean forecast with mis-calibrated intervals is still a bad forecast."*

---

## Cross-References

<a class="link-card" href="./04_forecast_evaluation_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_storage_theory.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
