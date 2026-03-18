# Reporting Causal Estimates: Effect Sizes, Intervals, and Sensitivity

## Learning Objectives

By the end of this guide, you will be able to:
1. Report treatment effect estimates with appropriate measures of uncertainty
2. Convert raw treatment effect estimates into interpretable effect sizes
3. Communicate Bayesian credible intervals and posterior probabilities
4. Structure a causal results section for a technical audience
5. Present robustness checks in a clear and concise format

---

## 1. What to Report Beyond the Point Estimate

A point estimate alone is incomplete. Every causal estimate should be reported with:

1. **Point estimate:** The treatment effect $\hat{\tau}$
2. **Uncertainty measure:** CI, HDI, SE, or posterior SD
3. **Effect size:** Standardised or interpretable metric
4. **Sample context:** N, bandwidth, cohort, calendar period
5. **Estimand:** ATT, ATE, LATE, or CATE

### The Complete Results Sentence

> "The [design] estimate suggests that [treatment] [increased/decreased] [outcome] by [magnitude] [units] ([95% CI: lower, upper], [N = n]), equivalent to [interpretable effect size]. This effect is local to [population/period/cutoff]."

Example:
> "The sharp RDD estimate suggests that receiving the merit scholarship increased first-year GPA by 0.25 grade points (95% CI: 0.11, 0.39, N = 312 within bandwidth). This is equivalent to moving a student from the 50th to the 61st percentile of the GPA distribution."

---

## 2. Bayesian vs Frequentist Reporting

### Frequentist Reporting

```python
import statsmodels.formula.api as smf
import numpy as np

model = smf.ols('outcome ~ treated + controls', data=df).fit(cov_type='HC1')
coef = model.params['treated']
se = model.bse['treated']
ci = model.conf_int().loc['treated'].values
p_val = model.pvalues['treated']

print(f"Treatment effect: {coef:.3f}")
print(f"Standard error:   {se:.3f}")
print(f"95% CI:           [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"p-value:          {p_val:.4f}")
```

The 95% CI means: "If we repeated this experiment many times, 95% of the CIs constructed this way would contain the true parameter." It does NOT mean "there is a 95% probability the true effect is in this range."

### Bayesian Reporting

```python
import causalpy as cp
import arviz as az
import numpy as np

result = cp.DifferenceInDifferences(...)
tau_samples = result.idata.posterior['post:treated'].values.flatten()

posterior_mean = tau_samples.mean()
posterior_sd = tau_samples.std()
hdi_94 = np.percentile(tau_samples, [3, 97])
p_positive = (tau_samples > 0).mean()

print(f"Posterior mean:   {posterior_mean:.3f}")
print(f"Posterior SD:     {posterior_sd:.3f}")
print(f"94% HDI:          [{hdi_94[0]:.3f}, {hdi_94[1]:.3f}]")
print(f"P(τ > 0):         {p_positive:.3f}")
```

The 94% HDI means: "There is 94% posterior probability the true effect is in this range." This is the direct interpretation practitioners usually want.

---

## 3. Effect Size Metrics

### Standardised Effect Size (Cohen's d)

Useful when the outcome is not on an interpretable scale:

```python
# Cohen's d: treatment effect in standard deviations of the outcome
outcome_sd = df['outcome'].std()
tau = 0.25  # treatment effect estimate

cohens_d = tau / outcome_sd
print(f"Effect size: d = {cohens_d:.3f}")
print(f"  d = 0.2: small")
print(f"  d = 0.5: medium")
print(f"  d = 0.8: large")
```

### Percentage Change

For outcomes on a ratio scale:

```python
baseline_mean = df[df['post']==0]['outcome'].mean()
tau = 0.25
pct_change = 100 * tau / baseline_mean
print(f"Effect as % of baseline: {pct_change:.1f}%")
```

### Number Needed to Treat (NNT)

For binary outcomes and policy evaluation:

```python
# If outcome is probability of employment (0-1)
tau_prob = 0.05  # 5 percentage point increase
nnt = 1 / tau_prob
print(f"NNT: {nnt:.0f} individuals treated for 1 additional employment outcome")
```

---

## 4. Reporting Robustness Checks

Robustness checks should be concise and structured. Present them in a table rather than prose.

### Template Robustness Table

```python
import pandas as pd

robustness_results = [
    # (Label, Estimate, CI_lo, CI_hi, N)
    ("Primary specification", 0.248, 0.112, 0.384, 1200),
    ("Bandwidth: half (h/2)", 0.231, 0.071, 0.391, 640),
    ("Bandwidth: double (2h)", 0.259, 0.143, 0.375, 2100),
    ("Polynomial order 2", 0.242, 0.098, 0.386, 1200),
    ("Without chain FE", 0.261, 0.122, 0.400, 1200),
    ("Donut (±0.05)", 0.238, 0.094, 0.382, 980),
]

rob_df = pd.DataFrame(robustness_results,
                       columns=['Specification', 'Estimate', 'CI_lo', 'CI_hi', 'N'])
rob_df['Significant'] = rob_df['CI_lo'] > 0  # above zero?

print("Table 2: Robustness Checks")
print("=" * 70)
header = f"{'Specification':<30} {'Estimate':>8} {'95% CI':>18} {'N':>6} {'Sig':>5}"
print(header)
print("-" * 70)
for _, row in rob_df.iterrows():
    ci_str = f"[{row['CI_lo']:>+.3f}, {row['CI_hi']:>+.3f}]"
    sig = "Yes" if row['Significant'] else "No"
    print(f"{row['Specification']:<30} {row['Estimate']:>+8.3f} {ci_str:>18} {int(row['N']):>6} {sig:>5}")
print("-" * 70)
print("All specifications show positive, significant effects.")
```

---

## 5. Visualising Results for Communication

### The "Main Results" Figure

For technical presentations, a forest plot showing the primary estimate with robustness checks is effective:

```python
import matplotlib.pyplot as plt
import numpy as np

def forest_plot(results_list, title='Treatment Effect Estimates', ax=None):
    """
    Create a forest plot from a list of (label, estimate, ci_lo, ci_hi) tuples.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, len(results_list) * 0.6 + 1.5))

    n = len(results_list)
    y_positions = list(range(n))

    for i, (label, est, lo, hi) in enumerate(results_list):
        is_primary = i == 0
        color = 'steelblue' if is_primary else 'gray'
        size = 100 if is_primary else 60
        lw = 2.5 if is_primary else 1.5

        ax.plot([lo, hi], [i, i], '-', color=color, linewidth=lw, alpha=0.8)
        ax.scatter(est, i, color=color, s=size, zorder=5)
        ax.text(hi + 0.01, i, f'{est:+.3f}', va='center', fontsize=9, color=color)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in results_list])
    ax.set_xlabel('Treatment Effect Estimate')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    return ax
```

### Bayesian Posterior Plot

For Bayesian results, show the full posterior distribution:

```python
def posterior_summary_plot(samples, true_value=None, title='Posterior Distribution'):
    fig, ax = plt.subplots(figsize=(9, 4))

    # Posterior histogram
    ax.hist(samples, bins=60, density=True, color='steelblue',
            alpha=0.7, edgecolor='white', label='Posterior')

    # HDI
    hdi_lo, hdi_hi = np.percentile(samples, [3, 97])
    ax.fill_betweenx([0, ax.get_ylim()[1]], hdi_lo, hdi_hi,
                      alpha=0.15, color='steelblue', label=f'94% HDI [{hdi_lo:.3f}, {hdi_hi:.3f}]')

    # Key markers
    ax.axvline(0, color='red', ls='--', lw=2, label='No effect')
    ax.axvline(np.mean(samples), color='black', ls='-', lw=2,
               label=f'Posterior mean = {np.mean(samples):.3f}')

    if true_value is not None:
        ax.axvline(true_value, color='green', ls=':', lw=2,
                   label=f'True value = {true_value:.3f}')

    # P(tau > 0) annotation
    p_pos = (samples > 0).mean()
    ax.text(0.02, 0.95, f'P(τ > 0) = {p_pos:.3f}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Treatment Effect (τ)')
    ax.set_ylabel('Posterior Density')
    ax.set_title(title)
    ax.legend(fontsize=9)
    return fig, ax
```

---

## 6. Sensitivity Analysis Reporting

Sensitivity analysis answers: "What would have to be true for my conclusion to be wrong?"

### For DiD: Roth Sensitivity

How large a pre-trend violation would overturn the result?

```python
# Conceptual: bound effect under pre-trend violations
def roth_sensitivity_table(estimate, se, pre_trend_range):
    """Show how the estimate shifts under different pre-trend violations."""
    print("Sensitivity to Pre-Trend Violations:")
    print(f"{'Violation':>12} {'Adjusted Estimate':>18} {'Significant?':>14}")
    print("-" * 50)
    for violation in pre_trend_range:
        adjusted = estimate - violation  # bias correction
        t_stat = adjusted / se
        sig = "Yes" if abs(t_stat) > 1.96 else "No"
        print(f"{violation:>12.3f} {adjusted:>18.3f} {sig:>14}")
```

### For RDD: Bandwidth Sensitivity Table

```python
def bandwidth_sensitivity_table(rdd_results_by_bandwidth):
    """
    rdd_results_by_bandwidth: dict of {bandwidth: (estimate, ci_lo, ci_hi)}
    """
    print("Bandwidth Sensitivity:")
    for h, (est, lo, hi) in sorted(rdd_results_by_bandwidth.items()):
        marker = " ← Primary" if h == primary_bandwidth else ""
        print(f"  h={h:.2f}: τ = {est:+.3f} [{lo:+.3f}, {hi:+.3f}]{marker}")
```

---

## 7. Summary: The Reporting Checklist

Before finalising a causal analysis report:

- [ ] Point estimate with SE or HDI
- [ ] Confidence or credible interval
- [ ] Effect size in interpretable units (and Cohen's d if needed)
- [ ] Sample size and relevant context (bandwidth, cohort, calendar period)
- [ ] Estimand (ATT, LATE, etc.) clearly stated
- [ ] Robustness table with ≥3 specifications
- [ ] Primary diagnostic evidence (pre-trend plot, McCrary test, etc.)
- [ ] Main limitation and direction of bias under violation
- [ ] Reproduced from raw data via documented code

---

**Previous:** [01 — Model Selection](01_model_selection_guide.md)
**Next:** [03 — Deployment Guide](03_deployment_guide.md)
