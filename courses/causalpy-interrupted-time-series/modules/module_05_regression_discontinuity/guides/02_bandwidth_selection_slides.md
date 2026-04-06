---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# RDD Bandwidth Selection

## The Most Consequential Tuning Choice

Module 05.2 | Causal Inference with CausalPy

<!-- Speaker notes: If there's one thing that separates a credible RDD from a questionable one, it's bandwidth selection and how it's reported. The choice of bandwidth determines your sample, your variance, your bias, and ultimately your conclusions. This module teaches you to make that choice principled and to prove to skeptics that your result isn't an artefact of a particular bandwidth. -->

---

## The Bandwidth Tradeoff

<div class="columns">

**Narrow bandwidth (small h):**
- Observations very close to cutoff
- Plausible continuity assumption
- Few observations → high variance
- Wide confidence intervals

**Wide bandwidth (large h):**
- Many observations → low variance
- Observations far from cutoff → potential bias
- Linear fit may miss nonlinearity

</div>

The **optimal bandwidth** minimises $\text{Bias}^2 + \text{Variance}$

<!-- Speaker notes: This is a classic statistical tradeoff, but in RDD it's particularly acute because the entire identification rests on observations near the cutoff. If you go too narrow, you don't have enough data to estimate the jump precisely. If you go too wide, your "local" regression isn't really local anymore — you're using observations that might have very different counterfactuals from those at the cutoff. -->

<div class="callout-info">
Info: Narrow bandwidth (small h):
</div>

---

## The MSE-Optimal Bandwidth

The Imbens-Kalyanaraman (IK) formula minimises asymptotic MSE:

$$h_{IK} = \left(\frac{\sigma^2}{n \cdot f(c) \cdot B^2}\right)^{1/5}$$

| Component | Meaning |
|-----------|---------|
| $\sigma^2$ | Outcome variance near cutoff |
| $f(c)$ | Running variable density at cutoff |
| $B$ | Curvature of regression function |
| $n$ | Sample size |

Bandwidth shrinks as $n \to \infty$ at rate $n^{-1/5}$

<!-- Speaker notes: The IK formula has nice intuition. Sigma-squared over n times f(c) is proportional to the variance — more data and denser running variable near the cutoff means you can afford a smaller bandwidth. B-squared is the square of the bias-driving curvature — more nonlinearity means you need a smaller bandwidth to keep bias low. The fifth root comes from the fact that bias is order h-squared and variance is order 1/(nh), and you're minimising their sum. -->

---

## Computing IK Bandwidth

```python
from rdrobust import rdbwselect, rdrobust
import numpy as np

# Select bandwidth
bw = rdbwselect(y=df['y'], x=df['x'], c=0, bwselect='mserd')
h_opt = bw.bws['h'][0]

print(f"IK optimal bandwidth: h = {h_opt:.3f}")
print(f"Effective sample size: n = {(np.abs(df['x']) <= h_opt).sum()}")

# Estimate with IK bandwidth
result = rdrobust(y=df['y'], x=df['x'], c=0, h=h_opt)
print(f"τ = {result.coef[0]:.3f}")
print(f"95% CI: [{result.ci[0]:.3f}, {result.ci[1]:.3f}]")
```

`bwselect='mserd'` → same bandwidth each side (default)
`bwselect='msetwo'` → different bandwidths each side

<!-- Speaker notes: In practice, just use rdbwselect from the rdrobust package and take the mserd bandwidth. That's the default MSE-optimal bandwidth using the same window on both sides of the cutoff. If the density of your running variable is very different on the two sides, msetwo allows asymmetric bandwidths. But for most applications, mserd is the right starting point, and you report it as your primary bandwidth. -->

---

## Bandwidth Sensitivity Plot

The critical robustness check: **do results hold across bandwidths?**

```python
bandwidths = np.linspace(0.05, 0.8, 30)
ests = []
cis = []

for h in bandwidths:
    r = rdrobust(y=df['y'], x=df['x'], c=0, h=h)
    ests.append(r.coef[0])
    cis.append((r.ci[0], r.ci[1]))

plt.plot(bandwidths, ests, 'o-', color='steelblue')
plt.fill_between(bandwidths, [c[0] for c in cis],
                              [c[1] for c in cis], alpha=0.2)
plt.axvline(h_opt, color='red', ls=':', label=f'IK h={h_opt:.2f}')
plt.xlabel('Bandwidth h')
plt.ylabel('Treatment Effect')
```

<!-- Speaker notes: The sensitivity plot is what reviewers really want to see. If the estimate is stable across a range of bandwidths — say, from 0.2 to 0.5 — that's strong evidence the result isn't an artefact of choosing a particular window. If the estimate jumps around wildly, you have a problem. Present this plot in your paper or presentation. The IK bandwidth should be in the middle of the stable region, not at an extreme. -->

<div class="callout-key">
Key Point: do results hold across bandwidths?
</div>

---

## Interpreting the Sensitivity Plot

```
Treatment Effect
  |
2 |    ●—●—●—●—●—●—●—●—●   ← stable (good)
  |                         ↑ IK bandwidth
1 |
0 |_____________________→ Bandwidth h

Treatment Effect
  |
3 |          ●●●●        ← only appears here
2 |     ●●                 (fragile = bad)
1 |  ●●
0 |●
  |_____________________→ Bandwidth h
```

**Stable:** Consistent estimate across a range → result is robust
**Fragile:** Significant only at one bandwidth → suspect artifact

<!-- Speaker notes: Reading the sensitivity plot is a skill. The top example shows what you want: a relatively flat line with overlapping confidence intervals across a range of bandwidths. The bottom example shows what should concern you: the effect only appears for a specific narrow bandwidth range and then disappears. That pattern suggests you're fitting noise, not detecting a real effect. Always present the sensitivity plot. Never cherry-pick a bandwidth that produces the result you want. -->

<div class="callout-insight">
Insight:  Consistent estimate across a range → result is robust

</div>

---

## Polynomial Order: Gelman & Imbens (2019)

**Do not use high-order global polynomials**

<div class="columns">

**Problems with order 4+:**
- Sensitive to far-off data points
- Runge's phenomenon near boundary
- Erratic confidence intervals
- Over-rejects the null

**Recommendation:**
- Local linear (p=1): primary estimate
- Local quadratic (p=2): robustness check
- Never p ≥ 3 without strong justification

</div>

<!-- Speaker notes: Gelman and Imbens wrote a paper specifically about this, and it changed practice. Before their paper, it was common to see RDD papers using global 4th or 5th degree polynomials. These look like they're fitting the data well globally, but near the boundary where the treatment effect is identified, they're driven by observations far away. The cure is to use local linear regression: fit a line in a bandwidth, not a polynomial across the full range. The result is more honest about what you're actually estimating. -->

<div class="callout-warning">
Warning: Do not use high-order global polynomials
</div>

---

## Comparing Polynomial Orders

```python
for p in [1, 2, 3, 4]:
    r = rdrobust(y=df['y'], x=df['x'],
                 c=0, h=h_opt, p=p)
    print(f"p={p}: τ={r.coef[0]:.3f}, "
          f"SE={r.se[0]:.3f}")
```

Expected output (well-behaved data):
```
p=1: τ=1.502, SE=0.087   ← primary
p=2: τ=1.487, SE=0.103   ← robustness (similar)
p=3: τ=1.531, SE=0.142   ← higher variance
p=4: τ=1.608, SE=0.198   ← unstable
```

If p=1 and p=2 agree → local linear is sufficient

<!-- Speaker notes: In practice, local linear and local quadratic usually give very similar estimates when the bandwidth is reasonable. If they diverge a lot, it means there's substantial curvature in the regression function near the cutoff, and you might want to widen the bandwidth slightly for the linear estimator or stick with quadratic. Report both. If even p=2 gives a very different answer from p=1, investigate whether there's non-linearity you need to address. -->

---

## The Donut RDD

Exclude the innermost observations to test for local manipulation:

```
Observations used:
  |  ●●●●    ●●●●  |
  |      [donut]   |
  |___←h_________h→|
         cutoff
```

```python
for donut_width in [0.01, 0.02, 0.05]:
    mask = (np.abs(df['x']) > donut_width) & (np.abs(df['x']) <= h_opt)
    donut_df = df[mask]
    r = rdrobust(y=donut_df['y'], x=donut_df['x'], c=0)
    print(f"Donut {donut_width}: τ={r.coef[0]:.3f}")
```

If estimates are stable → manipulation unlikely to drive results

<!-- Speaker notes: The donut RDD is a clever diagnostic. If there's local manipulation — students gaming their score to land just above the cutoff — those students are concentrated in the immediate neighbourhood of the cutoff. By excluding them (the "donut hole") and estimating the discontinuity using observations slightly further away, you test whether the result survives without those potentially manipulated observations. Stable estimates across donut widths is reassuring. -->

---

## Kernel Choice

| Kernel | Weights obs by distance? | Default? |
|--------|------------------------|---------|
| Triangular | Yes — linearly | Yes |
| Uniform | No | No |
| Epanechnikov | Yes — quadratically | No |

For boundary estimation, **triangular is optimal** — it weights nearby observations more and is the default in `rdrobust`.

In practice, kernel choice rarely changes conclusions.

<!-- Speaker notes: Kernel choice is the least consequential decision in RDD. The triangular kernel gives the most weight to observations near the cutoff and less to those farther away — this is optimal for boundary estimation theoretically. But in practice, the uniform kernel (equal weights within the window) usually gives similar results. If your conclusions change based on kernel choice, that's a sign your results are not robust, and the kernel choice is masking that. -->

---

## Standard Errors: Clustering

If observations cluster within groups, standard errors must account for that:

```python
# Cluster at school level (if students nest within schools)
result = rdrobust(y=df['y'], x=df['x'],
                  c=0, h=h_opt,
                  cluster=df['school_id'])
```

Ignoring clustering understates SEs → inflated t-statistics → false positives

**When to cluster:**
- Students within schools
- Workers within firms
- Counties within states

<!-- Speaker notes: Clustering is often overlooked in RDD papers. If your running variable is a school-level score and outcomes are measured at the student level, students in the same school are not independent observations. Clustering inflates your effective sample size if ignored, leading to confidence intervals that are too narrow. The rdrobust package handles clustering directly — just pass the cluster ID variable. This can meaningfully widen your confidence intervals, so don't be surprised if significance disappears when you add clustering. -->

---

## Summary: Bandwidth Reporting Checklist

Report in your paper or presentation:

- [ ] IK MSE-optimal bandwidth (primary estimate)
- [ ] Sensitivity plot across bandwidth range
- [ ] Estimates at 0.5× and 2× the IK bandwidth
- [ ] Local linear as primary, local quadratic as robustness
- [ ] Donut RDD results (at least one donut width)
- [ ] Cluster-robust SEs if data has natural clustering

<!-- Speaker notes: This is your reporting checklist. A credible RDD paper includes all of these. The primary estimate uses the IK bandwidth. The sensitivity plot shows it's not cherry-picked. The half and double bandwidth estimates are the standard robustness checks. The donut tests for local manipulation. And clustering, if applicable, ensures your standard errors are honest. Check off all of these before submitting. -->

---

<!-- _class: lead -->

## Next: CausalPy RDD in Practice

Three notebooks applying RDD to real-world estimation problems

→ [Notebooks: Sharp RDD, Sensitivity, CausalPy API](../notebooks/)

<!-- Speaker notes: We've covered the theory and the diagnostics. Now let's implement. The notebooks take you through a sharp RDD on an education policy dataset, a full sensitivity analysis workflow, and the CausalPy RegressionDiscontinuity class with Bayesian estimation. -->
