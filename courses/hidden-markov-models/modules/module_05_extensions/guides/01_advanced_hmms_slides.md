---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Advanced HMM Variants
## Beyond the Standard Model

### Module 05 — Extensions
### Hidden Markov Models Course

<!-- Speaker notes: This section covers extensions beyond the standard first-order HMM: higher-order models, input-output HMMs, hierarchical HMMs, and duration models. Each addresses a specific limitation of the basic HMM framework. -->
---

# HMM Variant Landscape

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    STD[\"Standard HMM\"] --> STICKY[\"Sticky HMM\"]
    STD --> MSAR[\"Markov-Switching AR\"]
    STD --> HHMM[\"Hierarchical HMM\"]
    STD --> IOHMM[\"Input-Output HMM\"]
    STD --> DHMM[\"Duration HMM\"]
    STICKY -->|\"Persistent regimes\"| FIN[\"Finance\"]
    MSAR -->|\"Autocorrelated data\"| FIN
    HHMM -->|\"Multi-scale structure\"| FIN
```

<div class="callout-key">

Key implementation detail -- study this pattern carefully.

</div>

<!-- Speaker notes: This overview diagram shows the five variants covered in this deck, each addressing a specific limitation of the standard HMM. The financial applications column shows which variants are most relevant. -->
---

<!-- _class: lead -->

# Sticky HMM

<!-- Speaker notes: The Sticky HMM addresses one of the most common practical complaints about standard HMMs: excessive state switching that produces noisy regime estimates. -->
---

# The Switching Problem

Standard HMMs may switch states **too frequently**. Sticky HMMs increase self-transition probability:

$$a_{ii}^{sticky} = \kappa + (1-\kappa) \cdot a_{ii}$$

where $\kappa \in [0, 1]$ is the stickiness parameter.

<div class="columns">
<div>

**Standard ($\kappa = 0$)**
- Original transitions
- May over-switch
- Noisy state sequences

</div>
<div>

**Sticky ($\kappa = 0.7$)**
- Boosted self-transitions
- Smoother regimes
- More persistent states

</div>
</div>

<!-- Speaker notes: Standard HMMs may switch states too frequently because the geometric duration distribution has its mode at 1. The stickiness parameter kappa boosts self-transition probabilities to create more persistent regimes. -->
---

# Sticky HMM Implementation

```python
class StickyGaussianHMM:
    def __init__(self, n_components: int, kappa: float = 0.5):
        self.n_components = n_components
        self.kappa = kappa

    def fit(self, X: np.ndarray):
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model = hmm.GaussianHMM(
            n_components=self.n_components, covariance_type="full",
            n_iter=100, random_state=42)
        self.model.fit(X)

        # Apply stickiness
        A = self.model.transmat_
        A_sticky = np.zeros_like(A)
        for i in range(self.n_components):
            for j in range(self.n_components):
                if i == j:
                    A_sticky[i, j] = self.kappa + (1 - self.kappa) * A[i, j]
                else:
                    A_sticky[i, j] = (1 - self.kappa) * A[i, j]
            A_sticky[i] /= A_sticky[i].sum()
        self.model.transmat_ = A_sticky
        return self
```

<div class="callout-insight">

This pattern recurs throughout the course. Understanding it deeply pays dividends later.

</div>

<!-- Speaker notes: The implementation is a simple post-processing step after standard HMM fitting. Self-transition probabilities are boosted by kappa, and off-diagonal probabilities are scaled down. Rows are renormalized to maintain row-stochasticity. -->
---

# Stickiness Effect

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart LR
    subgraph Standard[\"Standard HMM\"]
        S0A[\"Bull\"] -->|\"0.95\"| S0A
        S0A -->|\"0.05\"| S1A[\"Bear\"]
        S1A -->|\"0.10\"| S0A
        S1A -->|\"0.90\"| S1A
    end
    subgraph Sticky[\"Sticky HMM (kappa=0.5)\"]
        S0B[\"Bull\"] -->|\"0.975\"| S0B
        S0B -->|\"0.025\"| S1B[\"Bear\"]
        S1B -->|\"0.05\"| S0B
        S1B -->|\"0.95\"| S1B
    end
```

<div class="callout-warning">

Watch for edge cases with this implementation in production use.

</div>

> Stickiness parameter $\kappa$ controls the trade-off between responsiveness and stability.

<!-- Speaker notes: The side-by-side comparison shows how kappa equals 0.5 transforms the transition matrix. Self-transitions increase from 0.95 to 0.975, reducing the switching rate by half. This produces smoother, more persistent regime estimates. -->
---

<!-- _class: lead -->

# Markov-Switching Autoregressive Models

<!-- Speaker notes: MS-AR models are particularly important for financial time series because returns often exhibit regime-dependent autocorrelation that standard HMMs miss. -->
---

# MS-AR Model

Combine regime switching with autoregressive dynamics:

$$y_t = c_{s_t} + \phi_{s_t} y_{t-1} + \sigma_{s_t} \epsilon_t$$

Each regime $s_t$ has its own:
- **Intercept** $c_{s_t}$
- **AR coefficient** $\phi_{s_t}$
- **Volatility** $\sigma_{s_t}$

> Captures both regime changes **and** temporal autocorrelation within each regime.

<!-- Speaker notes: The Markov-Switching Autoregressive model adds temporal dynamics within each regime. Standard HMMs assume observations are conditionally independent given the state; MS-AR relaxes this by allowing autoregressive structure. -->
---

# MS-AR with statsmodels

```python
from statsmodels.tsa.regime_switching.markov_autoregression \
    import MarkovAutoregression

def fit_markov_switching_ar(y, k_regimes=2, order=1,
                             switching_variance=True):
    model = MarkovAutoregression(
        y, k_regimes=k_regimes, order=order,
        switching_ar=True,
        switching_variance=switching_variance)
    results = model.fit()
    return results

results = fit_markov_switching_ar(y, k_regimes=2, order=1)
smoothed_probs = results.smoothed_marginal_probabilities
```

<div class="callout-info">

This approach follows established best practices in the field.

</div>

<!-- Speaker notes: The statsmodels implementation provides a production-ready MS-AR model. The switching_ar and switching_variance parameters control which coefficients change between regimes. -->
---

# Regime-Dependent Forecasting

```python
def regime_forecast(results, y, horizon=5):
    current_probs = results.smoothed_marginal_probabilities[-1]
    P = results.regime_transition
    params = results.params
    forecasts = {'point': [], 'by_regime': {}}

    for regime in range(results.k_regimes):
        forecasts['by_regime'][regime] = []

    for h in range(1, horizon + 1):
        future_probs = np.dot(current_probs,
                              np.linalg.matrix_power(P, h))
        for regime in range(results.k_regimes):
            prev = y[-1] if h == 1 else \
                   forecasts['by_regime'][regime][-1]
            fc = params[f'const[{regime}]'] + \
                 params[f'ar.L1[{regime}]'] * prev
            forecasts['by_regime'][regime].append(fc)
        point_fc = sum(future_probs[r] * forecasts['by_regime'][r][-1]
                       for r in range(results.k_regimes))
        forecasts['point'].append(point_fc)
    return forecasts
```

<!-- Speaker notes: MS-AR enables regime-dependent point forecasts by weighting regime-specific forecasts by future regime probabilities. The transition matrix raised to the power h gives the h-step-ahead regime probabilities. -->
---

# MS-AR Forecasting Flow

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    DATA[\"Time Series\"] --> FIT[\"Fit MS-AR Model\"]
    FIT --> CP[\"Current Regime<br>Probabilities\"]
    FIT --> TP[\"Transition<br>Matrix P\"]
    CP --> FP[\"Future Probs<br>P^h\"]
    TP --> FP
    FIT --> RF[\"Regime-Specific<br>Forecasts\"]
    FP --> BLEND[\"Probability-Weighted<br>Point Forecast\"]
    RF --> BLEND
```

<!-- Speaker notes: The flow diagram shows how current regime probabilities propagate through the transition matrix to produce future regime probabilities, which weight the regime-specific forecasts. -->
---

<!-- _class: lead -->

# Hierarchical HMM

<!-- Speaker notes: Hierarchical HMMs capture multi-scale dynamics, such as slow macroeconomic cycles containing faster micro-regime fluctuations. -->
---

# Multi-Scale Structure

```
Super-states (macro regimes)   e.g., Expansion / Recession
      |
Sub-states (micro regimes)     e.g., Early/Late Expansion
      |
Observations                   e.g., Returns, Volatility
```

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    subgraph Super[\"Super-States\"]
        EXP[\"Expansion\"]
        REC[\"Recession\"]
    end
    subgraph Sub_Exp[\"Expansion Sub-States\"]
        EARLY[\"Early\"]
        LATE[\"Late\"]
    end
    subgraph Sub_Rec[\"Recession Sub-States\"]
        MILD[\"Mild\"]
        SEVERE[\"Severe\"]
    end
    EXP --> Sub_Exp
    REC --> Sub_Rec
    Sub_Exp --> OBS1[\"Observations\"]
    Sub_Rec --> OBS1
```

<!-- Speaker notes: Hierarchical HMMs model phenomena that operate at multiple time scales. For example, the economy cycles between expansion and recession at a slow time scale, while within each phase, sub-regimes capture faster dynamics. -->
---

# Hierarchical HMM Implementation

```python
class HierarchicalHMM:
    def __init__(self, n_super_states, n_sub_states_per_super):
        self.n_super = n_super_states
        self.n_sub = n_sub_states_per_super
        self.n_total = n_super_states * n_sub_states_per_super
        self.A_super = None
        self.A_sub = [None] * n_super_states

    def _super_to_sub_idx(self, super_state, sub_state):
        return super_state * self.n_sub + sub_state

    def _sub_to_super_idx(self, flat_idx):
        return flat_idx // self.n_sub, flat_idx % self.n_sub
```

<!-- Speaker notes: The implementation maps the two-level hierarchy to a flat state space. Each super-state has n_sub sub-states, giving n_super times n_sub total states. The mapping functions convert between flat and hierarchical indices. -->
---

# Full Transition Matrix Construction

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">build_full_transition_matrix.py</span>
</div>

```python
def build_full_transition_matrix(self):
    A_full = np.zeros((self.n_total, self.n_total))

    for super_i in range(self.n_super):
        for sub_i in range(self.n_sub):
            from_idx = self._super_to_sub_idx(super_i, sub_i)
            for super_j in range(self.n_super):
                for sub_j in range(self.n_sub):
                    to_idx = self._super_to_sub_idx(super_j, sub_j)
                    if super_i == super_j:
                        # Within same super-state: sub-level transitions
                        A_full[from_idx, to_idx] = (
                            (1 - self.A_super[super_i].sum() +
                             self.A_super[super_i, super_i]) *
                            self.A_sub[super_i][sub_i, sub_j])
                    else:
                        # Cross super-state: reset sub-state
                        A_full[from_idx, to_idx] = (
                            self.A_super[super_i, super_j] / self.n_sub)
    A_full = A_full / A_full.sum(axis=1, keepdims=True)
    return A_full
```

</div>

<!-- Speaker notes: The full transition matrix combines super-state transitions with sub-state transitions. Within a super-state, sub-state transitions are governed by the sub-state transition matrix. Cross-super-state transitions reset the sub-state uniformly. -->
---

<!-- _class: lead -->

# Input-Output HMM

<!-- Speaker notes: Input-Output HMMs incorporate exogenous variables like macro indicators or sentiment scores, making the emission model conditional on external information. -->
---

# Exogenous Variables in HMMs

Include external information that affects emissions:

$$P(o_t | s_t, x_t) = \mathcal{N}(\mu_{s_t} + W_{s_t} x_t, \sigma_{s_t}^2)$$

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">inputoutputhmm.py</span>
</div>

```python
class InputOutputHMM:
    def __init__(self, n_states, n_inputs):
        self.n_states = n_states
        self.n_inputs = n_inputs

    def emission_probability(self, observation, state, inputs):
        mean = self.means[state] + np.dot(self.input_weights[state],
                                           inputs)
        return stats.norm.pdf(observation, mean, self.stds[state])
```

</div>

> Use cases: macro indicators, sentiment scores, or sector signals as inputs.

<!-- Speaker notes: Input-Output HMMs allow external variables to influence the emission model. The mean of the emission distribution becomes a linear function of the inputs, enabling the model to incorporate macro indicators or sentiment scores. -->
---

<!-- _class: lead -->

# Duration-Dependent HMM

<!-- Speaker notes: Duration-dependent HMMs address the limitation that standard HMMs have geometric duration distributions, which may not match empirical regime durations. -->
---

# Beyond Geometric Durations

Standard HMMs have **geometric** state duration distributions. Explicit duration models allow **arbitrary** distributions:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">explicitdurationhmm.py</span>
</div>

```python
class ExplicitDurationHMM:
    def __init__(self, n_states, max_duration=50):
        self.n_states = n_states
        self.max_duration = max_duration
        self.duration_probs = None  # (n_states, max_duration)

    def set_poisson_durations(self, lambdas):
        self.duration_probs = np.zeros(
            (self.n_states, self.max_duration))
        for state, lam in enumerate(lambdas):
            for d in range(self.max_duration):
                self.duration_probs[state, d] = \
                    stats.poisson.pmf(d + 1, lam)
            self.duration_probs[state] /= \
                self.duration_probs[state].sum()
```

</div>

<!-- Speaker notes: Standard HMMs have geometric state duration distributions, which means the most likely duration is always 1. Duration-dependent HMMs allow arbitrary duration distributions like Poisson, which have a mode at lambda. -->
---

# Duration Distribution Comparison

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart LR
    subgraph Geometric[\"Standard HMM\"]
        G[\"P(d) = (1-p)^(d-1) * p\"]
        GP[\"Memoryless<br>Most probable d=1\"]
    end
    subgraph Poisson[\"Duration HMM (Poisson)\"]
        P[\"P(d) = Poisson(lambda)\"]
        PP[\"Mode at lambda<br>More realistic\"]
    end
    subgraph Custom[\"Duration HMM (Custom)\"]
        C[\"P(d) = any distribution\"]
        CP[\"Domain-specific<br>Maximum flexibility\"]
    end
```

> Financial regimes often have characteristic durations that geometric distributions cannot capture.

<!-- Speaker notes: The three-panel comparison shows how duration models differ. Geometric has mode at 1, Poisson has mode at lambda, and custom distributions can match domain-specific duration patterns. -->
---

# Variant Comparison

| Variant | Use Case | Complexity | Key Feature |
|----------|----------|----------|----------|
| **Standard HMM** | Basic regime detection | Low | Baseline |
| **Sticky HMM** | Persistent regimes | Low | Boosted self-transitions |
| **MS-AR** | Autocorrelated data | Medium | AR dynamics per regime |
| **Hierarchical HMM** | Multi-scale regimes | High | Nested state structure |
| **Duration HMM** | Non-geometric durations | High | Explicit duration model |
| **Input-Output HMM** | Exogenous variables | Medium | Conditional emissions |

<!-- Speaker notes: This table is the key reference for choosing a variant. Start with the standard HMM and add complexity only when the data demands it. Each additional feature increases both modeling power and estimation difficulty. -->

---

# Choosing the Right Variant

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    Q1{\"States switch<br>too often?\"} -->|Yes| STICKY[\"Sticky HMM\"]
    Q1 -->|No| Q2{\"Data is<br>autocorrelated?\"}
    Q2 -->|Yes| MSAR[\"MS-AR\"]
    Q2 -->|No| Q3{\"Multiple scales<br>of dynamics?\"}
    Q3 -->|Yes| HHMM[\"Hierarchical HMM\"]
    Q3 -->|No| Q4{\"External<br>variables?\"}
    Q4 -->|Yes| IOHMM[\"Input-Output HMM\"]
    Q4 -->|No| Q5{\"Duration<br>matters?\"}
    Q5 -->|Yes| DHMM[\"Duration HMM\"]
    Q5 -->|No| STD[\"Standard HMM\"]
```

<!-- Speaker notes: This decision tree guides variant selection based on observed data characteristics. Walk through each question: switching too often leads to Sticky, autocorrelation leads to MS-AR, and so on. -->
---

# Key Takeaways

| Takeaway | Detail |
|----------|----------|
| Sticky HMMs | Prevent excessive switching with $\kappa$ parameter |
| MS-AR models | Combine regime switching + autoregressive dynamics |
| Hierarchical HMMs | Multi-scale structure (macro + micro regimes) |
| Input-Output HMMs | Incorporate exogenous variables into emissions |
| Duration HMMs | Non-geometric state durations (Poisson, etc.) |
| Variant selection | Match model complexity to data characteristics |

<!-- Speaker notes: Advanced HMM variants address specific limitations of the standard model. Higher-order HMMs capture longer memory, input-output HMMs incorporate exogenous variables, hierarchical HMMs model multi-scale dynamics, and duration models handle non-geometric state durations. -->

---

# Connections

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart LR
    STD[\"Standard<br>HMM\"] --> EXT[\"Extensions\"]
    EXT --> STICKY[\"Sticky\"]
    EXT --> MSAR[\"MS-AR\"]
    EXT --> HHMM[\"Hierarchical\"]
    EXT --> IOHMM[\"Input-Output\"]
    EXT --> DHMM[\"Duration\"]
    STICKY --> APP[\"Financial<br>Applications\"]
    MSAR --> APP
    APP --> PROD[\"Production<br>Systems\"]
```

<!-- Speaker notes: This diagram places advanced HMMs in the broader landscape of probabilistic sequence models. Each variant extends the basic HMM in a different direction, and understanding when to use each variant is a key practical skill for applied work. -->
