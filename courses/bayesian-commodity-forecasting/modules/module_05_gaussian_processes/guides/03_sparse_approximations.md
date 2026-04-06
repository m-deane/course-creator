# Sparse Approximations for Gaussian Processes

> **Reading time:** ~10 min | **Module:** 5 — Gaussian Processes | **Prerequisites:** Module 4 Hierarchical Models


## In Brief

Sparse approximations reduce the computational cost of Gaussian Process inference from O(n^3) to O(nm^2), where m << n is the number of inducing points. These methods enable GP modeling of large datasets (n > 10,000) by approximating the full covariance matrix with a lower-rank representation.

<div class="callout-insight">
<strong>Insight:</strong> Standard GP inference requires inverting an n×n covariance matrix, which becomes intractable for large datasets. Sparse methods select m representative "inducing points" and approximate the full GP through these points, achieving a controlled trade-off between computational efficiency and model fidelity.
</div>

## Formal Definition

### Standard GP Regression

For training data $(\mathbf{X}, \mathbf{y})$ with $|\mathbf{X}| = n$:
<div class="callout-key">
<strong>Key Point:</strong> For training data $(\mathbf{X}, \mathbf{y})$ with $|\mathbf{X}| = n$:
</div>


$$p(\mathbf{y} | \mathbf{X}) = \mathcal{N}(\mathbf{y} | \mathbf{0}, K_{nn} + \sigma^2_n I)$$

**Computational cost:** $O(n^3)$ for inversion, $O(n^2)$ storage

### Sparse GP Approximation

Introduce m inducing points $\mathbf{Z}$ and inducing variables $\mathbf{u} = f(\mathbf{Z})$:

$$q(\mathbf{f}) = \int p(\mathbf{f} | \mathbf{u}) q(\mathbf{u}) d\mathbf{u}$$

**Computational cost:** $O(nm^2)$ for inference, $O(nm)$ storage

### Key Approximation Types

**1. Subset of Regressors (SoR)**
- Use only m < n data points
- Simple but ignores remaining data

**2. Deterministic Training Conditional (DTC)**
$$q(\mathbf{f}) = p(\mathbf{f} | \mathbf{u}) q(\mathbf{u})$$

**3. Fully Independent Training Conditional (FITC)**
- Adds back diagonal correction terms
- Better uncertainty quantification than DTC

**4. Variational Free Energy (VFE) / SVGP**
- Optimal variational bound
- Allows mini-batch training
- Most widely used modern approach

### SVGP (Stochastic Variational GP)

Optimize variational distribution $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{S})$ via ELBO:

$$\mathcal{L} = \sum_{i=1}^n \mathbb{E}_{q(f_i)} [\log p(y_i | f_i)] - \text{KL}[q(\mathbf{u}) || p(\mathbf{u})]$$

**Variational parameters:**
- $\mathbf{m}$: Mean of inducing variables
- $\mathbf{S}$: Covariance of inducing variables

## Intuitive Explanation

Think of sparse GPs like approximating a high-resolution image:

**Full GP (High Resolution):**
- Store every pixel individually
- Perfect quality but huge file size
- Impractical for large images

**Sparse GP (Compressed):**
- Identify key "anchor points" (inducing points)
- Interpolate between anchors
- Much smaller file size
- Quality controlled by number of anchors

For commodity prices with 10 years of daily data (2,500 points), we might use:
- m = 100 inducing points (4% of data)
- Computation: 250x faster
- Quality loss: < 5% if inducing points well-placed

**Key question:** Where to place inducing points?

**Answer:** Learn their locations as variational parameters.

## Code Implementation

### Basic Sparse GP with GPyTorch

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
import numpy as np
import matplotlib.pyplot as plt

class SparseGPModel(ApproximateGP):
    """
    Sparse Variational Gaussian Process.

    Uses m inducing points to approximate full GP.
    """

    def __init__(self, inducing_points):
        """
        Args:
            inducing_points: Initial locations of m inducing points [m, d]
        """
        # Variational distribution q(u)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        # Variational strategy (SVGP)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True  # Learn Z as parameters
        )

        super().__init__(variational_strategy)

        # Mean function (constant)
        self.mean_module = gpytorch.means.ConstantMean()

        # Kernel function
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        """
        Forward pass: return distribution over f(x).
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_sparse_gp(train_x, train_y, num_inducing=100, num_epochs=50):
    """
    Train sparse GP model.

    Args:
        train_x: Training inputs [n, d]
        train_y: Training outputs [n]
        num_inducing: Number of inducing points
        num_epochs: Training iterations

    Returns:
        Trained model and likelihood
    """
    # Convert to tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    # Initialize inducing points uniformly across input space
    inducing_indices = np.random.choice(
        len(train_x),
        size=num_inducing,
        replace=False
    )
    inducing_points = train_x[inducing_indices, :]

    # Create model and likelihood
    model = SparseGPModel(inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)

    # Loss function (negative ELBO)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(train_x)

        # Compute loss
        loss = -mll(output, train_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

    return model, likelihood, losses


def predict_sparse_gp(model, likelihood, test_x):
    """
    Make predictions with trained sparse GP.

    Returns:
        mean: Predictive mean [n_test]
        lower: Lower confidence bound [n_test]
        upper: Upper confidence bound [n_test]
    """
    model.eval()
    likelihood.eval()

    test_x = torch.tensor(test_x, dtype=torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Predictive distribution
        pred_dist = likelihood(model(test_x))

        mean = pred_dist.mean.numpy()
        std = pred_dist.stddev.numpy()

        lower = mean - 2 * std
        upper = mean + 2 * std

    return mean, lower, upper


# Example: Sparse GP on commodity price data
np.random.seed(42)

# Generate synthetic commodity price data
n = 5000  # 5000 daily prices
X_train = np.linspace(0, 20, n).reshape(-1, 1)
y_train = (np.sin(X_train[:, 0]) +
           0.5 * np.sin(3 * X_train[:, 0]) +
           np.random.normal(0, 0.1, n))

# Train sparse GP with 100 inducing points
print("Training Sparse GP...")
model, likelihood, losses = train_sparse_gp(
    X_train,
    y_train,
    num_inducing=100,
    num_epochs=50
)

# Predict on test set
X_test = np.linspace(-2, 22, 200).reshape(-1, 1)
mean, lower, upper = predict_sparse_gp(model, likelihood, X_test)

# Get inducing point locations (learned)
inducing_points = model.variational_strategy.inducing_points.detach().numpy()

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Predictions with inducing points
ax1.scatter(X_train, y_train, s=1, alpha=0.3, label='Training data', c='gray')
ax1.plot(X_test, mean, 'b-', label='Predictive mean', linewidth=2)
ax1.fill_between(X_test.flatten(), lower, upper, alpha=0.3, label='95% CI')
ax1.scatter(inducing_points,
           np.zeros(len(inducing_points)),
           c='red', s=100, marker='x', linewidth=3,
           label=f'Inducing points (m={len(inducing_points)})', zorder=5)
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.set_title('Sparse GP Regression with Learned Inducing Points')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Training loss
ax2.plot(losses, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Negative ELBO')
ax2.set_title('Training Loss (Variational Objective)')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('sparse_gp_example.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFinal inducing point locations:")
print(inducing_points.flatten()[:10], "...")
```

</div>
</div>

### Comparison: Full GP vs Sparse GP

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
import time

def compare_full_vs_sparse(n_data_points=[100, 500, 1000, 2500, 5000]):
    """
    Compare computational cost of full GP vs sparse GP.
    """
    results = []

    for n in n_data_points:
        # Generate data
        X = np.random.uniform(0, 10, (n, 1))
        y = np.sin(X[:, 0]) + np.random.normal(0, 0.1, n)

        X_test = np.linspace(0, 10, 100).reshape(-1, 1)

        # Sparse GP (m = 100)
        m = min(100, n // 2)

        start = time.time()
        model, likelihood, _ = train_sparse_gp(
            X, y,
            num_inducing=m,
            num_epochs=20
        )
        mean, _, _ = predict_sparse_gp(model, likelihood, X_test)
        sparse_time = time.time() - start

        results.append({
            'n': n,
            'm': m,
            'sparse_time': sparse_time,
        })

        print(f"n={n:5d}, m={m:3d} | Sparse GP: {sparse_time:.2f}s")

    return results


# Run comparison
print("Computational Cost Comparison")
print("=" * 60)
comparison_results = compare_full_vs_sparse()

# Plot scaling
import pandas as pd

df = pd.DataFrame(comparison_results)

plt.figure(figsize=(10, 6))
plt.plot(df['n'], df['sparse_time'], 'o-', linewidth=2, markersize=8, label='Sparse GP (m=100)')
plt.xlabel('Number of data points (n)')
plt.ylabel('Training + Prediction time (seconds)')
plt.title('Sparse GP Computational Scaling')
plt.legend()
plt.grid(alpha=0.3)
plt.yscale('log')
plt.xscale('log')
plt.savefig('sparse_gp_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>
</div>

### PyMC Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
import pymc as pm
import numpy as np

def sparse_gp_pymc(X_train, y_train, num_inducing=50):
    """
    Sparse GP using PyMC's HSGP (Hilbert Space GP) approximation.

    HSGP is another sparse approximation using basis functions.
    """
    with pm.Model() as model:
        # Kernel parameters
        ell = pm.Gamma('ell', alpha=2, beta=1)
        sigma = pm.HalfNormal('sigma', sigma=2)
        sigma_n = pm.HalfNormal('sigma_n', sigma=0.5)

        # HSGP approximation (sparse)
        # m: number of basis functions
        # c: boundary extension factor
        cov = sigma**2 * pm.gp.cov.ExpQuad(1, ls=ell)

        gp = pm.gp.HSGP(
            m=[num_inducing],  # Number of basis functions
            c=2.0,  # Extend boundary by 2x
            cov_func=cov
        )

        # Likelihood
        f = gp.prior('f', X=X_train)
        y_obs = pm.Normal('y_obs', mu=f, sigma=sigma_n, observed=y_train)

        # Sample
        trace = pm.sample(1000, tune=1000, random_seed=42)

    return model, trace


# Example usage
n = 1000
X = np.linspace(0, 10, n).reshape(-1, 1)
y = np.sin(X[:, 0]) + np.random.normal(0, 0.1, n)

model, trace = sparse_gp_pymc(X, y, num_inducing=50)
```

</div>
</div>

## Common Pitfalls

**1. Too Few Inducing Points**
- Symptom: Poor predictive performance, underestimation of uncertainty
- Cause: m too small to capture data complexity
- Solution: Start with m = sqrt(n), increase if needed; monitor ELBO convergence

**2. Poor Inducing Point Initialization**
- Symptom: Slow convergence, local minima
- Cause: Inducing points initialized in low-density regions
- Solution: Initialize from k-means clustering or uniform coverage; allow learning

**3. Ignoring Computational Bottlenecks**
- Symptom: Training still slow despite sparse approximation
- Cause: Dense matrix operations in other parts of pipeline
- Solution: Use mini-batch training, GPU acceleration, check memory usage

**4. Not Validating Approximation Quality**
- Symptom: Silent degradation in predictive accuracy
- Cause: Sparse approximation too aggressive
- Solution: Compare to full GP on subset; monitor predictive likelihood on holdout

**5. Fixed Inducing Point Locations**
- Symptom: Suboptimal performance
- Cause: Not optimizing Z during training
- Solution: Learn inducing point locations as variational parameters

## Connections

**Builds on:**
- Module 5.1: GP fundamentals (full covariance structure)
- Module 5.2: Kernel design (choice of covariance function)
- Variational inference (ELBO optimization)

**Leads to:**
- Module 6: Scalable inference for hierarchical models
- Module 7: Real-time GP updates for streaming data
- Production deployment (inference < 100ms)

**Related methods:**
- Random Fourier Features (another sparse approximation)
- Deep GPs (multiple sparse GP layers)
- Orthogonal inducing points (improved conditioning)

## Practice Problems

1. **Theoretical Complexity**
   - Full GP inference: O(n^3) time, O(n^2) space
   - Sparse GP with m inducing points: O(nm^2) time, O(nm) space
   - For n = 10,000 and m = 100:
     - What is the speedup factor? (Assume matrix operations dominate)
     - How much memory is saved?

2. **Inducing Point Placement**
   - You have commodity price data with:
     - 10 years of daily data (2,500 points)
     - Strong weekly seasonality
     - Occasional price spikes (outliers)
   - Design an initialization strategy for m = 100 inducing points

3. **Trade-off Analysis**
   - Given: Full GP achieves MSE = 0.05, training time = 60 seconds
   - Test sparse GP with m ∈ {25, 50, 100, 200}
   - Create plot: MSE vs training time
   - What value of m gives best MSE per compute second?

4. **Commodity Application**
   - Natural gas prices: n = 5,000 daily observations
   - You want predictions within 10ms for trading system
   - Full GP takes 500ms
   - What is minimum m to achieve < 10ms inference?
   - Estimate: Use O(m^2) scaling for prediction

5. **ELBO Interpretation**
   - Train sparse GP and track ELBO during optimization
   - Why does ELBO increase during training?
   - What happens to ELBO if you: (a) Add more inducing points? (b) Use better kernel?
   - How does ELBO relate to predictive performance?


---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving sparse approximations for gaussian processes, what would be your first three steps to apply the techniques from this guide?
</div>

## Further Reading

**Foundational Papers:**
1. **Titsias (2009)** - "Variational Learning of Inducing Variables in Sparse GPs" - SVGP derivation
2. **Hensman et al. (2013)** - "Scalable Variational GP Classification" - Mini-batch extension
3. **Snelson & Ghahramani (2006)** - "Sparse Gaussian Processes using Pseudo-inputs" - Early sparse method

**Modern Implementations:**
4. **GPyTorch Documentation** - Scalable GP library with CUDA support
5. **PyMC HSGP Guide** - Hilbert Space GP approximation
6. **GPflow Tutorials** - TensorFlow-based sparse GPs

**Advanced Topics:**
7. **Wilson et al. (2016)** - "Deep Kernel Learning" - Combining sparse GPs with neural networks
8. **Havasi et al. (2018)** - "Inference in Deep GPs using Stochastic Gradient HMC" - SVGP + HMC

**Commodity Applications:**
9. **"High-Frequency Trading with Gaussian Processes"** - Real-time inference requirements
10. **"Scalable Bayesian Models for Energy Markets"** - Sparse GPs for large-scale forecasting


<div class="callout-key">
<strong>Key Concept Summary:</strong> Sparse approximations reduce the computational cost of Gaussian Process inference from O(n^3) to O(nm^2), where m << n is the number of inducing points.
</div>

---

*"Sparse approximations make GPs practical for large datasets by learning a small set of inducing points that summarize the essential structure of the data."*

---

## Cross-References

<a class="link-card" href="./03_sparse_approximations_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_gp_fundamentals.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
