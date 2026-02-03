# Variational Inference for Bayesian Models

## In Brief

Variational Inference (VI) approximates intractable posterior distributions by transforming Bayesian inference into an optimization problem. Instead of sampling (MCMC), VI finds the best approximation from a simpler family of distributions by maximizing the Evidence Lower Bound (ELBO).

## Key Insight

MCMC explores the posterior through random sampling—accurate but slow. VI turns inference into optimization: find the distribution in a tractable family (e.g., Gaussians) that is closest to the true posterior. Trade exactness for speed—VI can be 100x faster than MCMC, enabling models with millions of parameters.

## Formal Definition

### The Inference Problem

Given data $\mathbf{y}$ and model parameters $\boldsymbol{\theta}$:

$$p(\boldsymbol{\theta} | \mathbf{y}) = \frac{p(\mathbf{y} | \boldsymbol{\theta}) p(\boldsymbol{\theta})}{p(\mathbf{y})}$$

**Challenge:** $p(\mathbf{y}) = \int p(\mathbf{y} | \boldsymbol{\theta}) p(\boldsymbol{\theta}) d\boldsymbol{\theta}$ is intractable.

### Variational Approximation

Instead of computing $p(\boldsymbol{\theta} | \mathbf{y})$ exactly, approximate it:

$$p(\boldsymbol{\theta} | \mathbf{y}) \approx q(\boldsymbol{\theta}; \boldsymbol{\phi})$$

where $q$ is a tractable family (e.g., Gaussian) parameterized by $\boldsymbol{\phi}$.

### Evidence Lower Bound (ELBO)

Cannot directly minimize $\text{KL}[q || p]$ (requires knowing $p$). Instead, maximize:

$$\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{q(\boldsymbol{\theta})}[\log p(\mathbf{y}, \boldsymbol{\theta})] - \mathbb{E}_{q(\boldsymbol{\theta})}[\log q(\boldsymbol{\theta})]$$

**Equivalently:**

$$\mathcal{L}(\boldsymbol{\phi}) = \log p(\mathbf{y}) - \text{KL}[q(\boldsymbol{\theta}) || p(\boldsymbol{\theta} | \mathbf{y})]$$

Since $\log p(\mathbf{y})$ is constant w.r.t. $\boldsymbol{\phi}$:

$$\max_{\boldsymbol{\phi}} \mathcal{L}(\boldsymbol{\phi}) \iff \min_{\boldsymbol{\phi}} \text{KL}[q || p]$$

### Mean-Field Variational Family

**Fully factorized approximation:**

$$q(\boldsymbol{\theta}) = \prod_{i=1}^d q_i(\theta_i)$$

Each parameter has independent approximate posterior.

**Common choice:** Gaussian mean-field:

$$q(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$$

Variational parameters: $\boldsymbol{\phi} = \{\boldsymbol{\mu}, \boldsymbol{\sigma}\}$

## Intuitive Explanation

Think of finding a lost item in your house:

**MCMC (Sampling):**
- Randomly search rooms
- Eventually cover entire house
- Guaranteed to find item (given enough time)
- Slow but thorough

**VI (Optimization):**
- Start with guess: "probably in kitchen"
- Refine guess: "upper left cabinet"
- Stop when guess seems good enough
- Fast but approximate

For commodity forecasting with 10,000 parameters:
- MCMC: 2 hours to converge
- VI: 5 minutes to approximate

VI trades accuracy for speed. How much accuracy? Depends on:
1. How well the variational family matches the true posterior
2. How complex the posterior is
3. How well optimization converges

## Code Implementation

### Basic Variational Inference in PyMC

```python
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Generate synthetic data
np.random.seed(42)
n = 200
X = np.random.normal(0, 1, n)
true_alpha = 2.5
true_beta = 1.3
true_sigma = 0.5
y = true_alpha + true_beta * X + np.random.normal(0, true_sigma, n)

# Build Bayesian linear regression model
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=2)

    # Linear model
    mu = alpha + beta * X

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Variational Inference with ADVI
    approx = pm.fit(
        n=20000,  # Number of optimization iterations
        method='advi',  # Automatic Differentiation VI
        callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)]
    )

    # Sample from variational approximation
    trace_vi = approx.sample(2000)

    # For comparison: MCMC sampling
    trace_mcmc = pm.sample(2000, tune=1000, random_seed=42)

# Compare VI vs MCMC
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

params = ['alpha', 'beta', 'sigma']
true_values = [true_alpha, true_beta, true_sigma]

for i, (param, true_val) in enumerate(zip(params, true_values)):
    # Trace plot
    axes[0, i].plot(trace_vi.posterior[param].values.flatten(),
                    alpha=0.5, label='VI', linewidth=1)
    axes[0, i].plot(trace_mcmc.posterior[param].values.flatten(),
                    alpha=0.5, label='MCMC', linewidth=1)
    axes[0, i].axhline(true_val, color='red', linestyle='--',
                       label='True', linewidth=2)
    axes[0, i].set_ylabel(param)
    axes[0, i].legend()
    axes[0, i].set_title(f'{param} - Trace')

    # Posterior distribution
    axes[1, i].hist(trace_vi.posterior[param].values.flatten(),
                    bins=50, alpha=0.5, density=True, label='VI')
    axes[1, i].hist(trace_mcmc.posterior[param].values.flatten(),
                    bins=50, alpha=0.5, density=True, label='MCMC')
    axes[1, i].axvline(true_val, color='red', linestyle='--',
                       label='True', linewidth=2)
    axes[1, i].set_xlabel(param)
    axes[1, i].legend()
    axes[1, i].set_title(f'{param} - Posterior')

plt.tight_layout()
plt.savefig('vi_vs_mcmc_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Variational Inference Results:")
print(az.summary(trace_vi))
print("\nMCMC Results:")
print(az.summary(trace_mcmc))
```

### Custom Variational Family (Full-Rank Gaussian)

```python
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

def custom_vi_linear_regression(X, y, n_iter=5000):
    """
    Custom VI implementation with full-rank Gaussian approximation.

    Approximating posterior: q(theta) = N(mu, Sigma)
    where theta = [alpha, beta, log_sigma]
    """
    n = len(y)
    d = 3  # Number of parameters

    # Initialize variational parameters
    mu = np.random.randn(d)
    L = np.eye(d)  # Cholesky factor of covariance

    def elbo(phi, X, y):
        """
        Compute ELBO for linear regression.

        Args:
            phi: Variational parameters [mu (d), L_vec (d*(d+1)/2)]
        """
        mu = phi[:d]

        # Reconstruct Cholesky factor
        L = np.zeros((d, d))
        L[np.tril_indices(d)] = phi[d:]

        # Sample from variational distribution
        n_samples = 100
        epsilon = np.random.randn(n_samples, d)
        theta_samples = mu + epsilon @ L.T

        # Compute log joint for each sample
        log_joints = []
        for theta in theta_samples:
            alpha, beta, log_sigma = theta
            sigma = np.exp(log_sigma)

            # Log prior
            log_prior = (
                -0.5 * (alpha**2 / 100 + beta**2 / 100) +  # N(0, 10) priors
                -log_sigma - 0.5 * (sigma / 2)**2  # HalfNormal(2) prior (approx)
            )

            # Log likelihood
            y_pred = alpha + beta * X
            log_lik = -n * log_sigma - 0.5 * np.sum((y - y_pred)**2) / sigma**2

            log_joints.append(log_prior + log_lik)

        # Expected log joint
        E_log_joint = np.mean(log_joints)

        # Entropy of q (Gaussian entropy)
        entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + np.sum(np.log(np.diag(L)))

        # ELBO = E[log p(y, theta)] + Entropy[q(theta)]
        return -(E_log_joint + entropy)  # Negative for minimization

    # Pack initial parameters
    L_vec = L[np.tril_indices(d)]
    phi_init = np.concatenate([mu, L_vec])

    # Optimize ELBO
    print("Optimizing ELBO...")
    result = minimize(
        elbo,
        phi_init,
        args=(X, y),
        method='L-BFGS-B',
        options={'maxiter': n_iter, 'disp': True}
    )

    # Extract optimized parameters
    phi_opt = result.x
    mu_opt = phi_opt[:d]
    L_opt = np.zeros((d, d))
    L_opt[np.tril_indices(d)] = phi_opt[d:]
    Sigma_opt = L_opt @ L_opt.T

    return mu_opt, Sigma_opt

# Example usage
mu_opt, Sigma_opt = custom_vi_linear_regression(X, y)

print("\nOptimized Variational Parameters:")
print(f"Mean: {mu_opt}")
print(f"Covariance:\n{Sigma_opt}")
print(f"\nTrue values: alpha={true_alpha}, beta={true_beta}, log_sigma={np.log(true_sigma)}")
```

### Stochastic Variational Inference (Mini-Batch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class BayesianLinearRegression(nn.Module):
    """
    Bayesian linear regression with stochastic VI.

    Uses mini-batch gradient ascent on ELBO.
    """

    def __init__(self, input_dim):
        super().__init__()

        # Variational parameters (mean-field Gaussian)
        self.alpha_mu = nn.Parameter(torch.randn(1))
        self.alpha_logstd = nn.Parameter(torch.randn(1))

        self.beta_mu = nn.Parameter(torch.randn(input_dim))
        self.beta_logstd = nn.Parameter(torch.randn(input_dim))

        self.sigma_mu = nn.Parameter(torch.randn(1))
        self.sigma_logstd = nn.Parameter(torch.randn(1))

    def sample_parameters(self, n_samples=1):
        """
        Sample parameters from variational distribution.
        """
        # Reparameterization trick
        alpha = Normal(self.alpha_mu, self.alpha_logstd.exp()).rsample((n_samples,))
        beta = Normal(self.beta_mu, self.beta_logstd.exp()).rsample((n_samples,))
        log_sigma = Normal(self.sigma_mu, self.sigma_logstd.exp()).rsample((n_samples,))
        sigma = log_sigma.exp()

        return alpha, beta, sigma

    def elbo_loss(self, X_batch, y_batch, n_total):
        """
        Compute negative ELBO for mini-batch.

        Args:
            X_batch: Mini-batch of inputs [batch_size]
            y_batch: Mini-batch of outputs [batch_size]
            n_total: Total dataset size (for scaling)
        """
        batch_size = len(X_batch)

        # Sample parameters
        alpha, beta, sigma = self.sample_parameters(n_samples=5)

        # Compute log likelihood for samples (Monte Carlo estimate)
        log_likelihoods = []
        for i in range(5):
            y_pred = alpha[i] + beta[i] * X_batch
            log_lik = Normal(y_pred, sigma[i]).log_prob(y_batch).sum()
            # Scale to full dataset
            log_lik_scaled = log_lik * (n_total / batch_size)
            log_likelihoods.append(log_lik_scaled)

        E_log_likelihood = torch.stack(log_likelihoods).mean()

        # KL divergence: q(theta) || p(theta)
        # Prior: alpha ~ N(0, 10), beta ~ N(0, 10), log_sigma ~ N(0, 1)
        kl_alpha = Normal(self.alpha_mu, self.alpha_logstd.exp()).log_prob(alpha).mean() - \
                   Normal(0, 10).log_prob(alpha).mean()
        kl_beta = Normal(self.beta_mu, self.beta_logstd.exp()).log_prob(beta).mean() - \
                  Normal(0, 10).log_prob(beta).mean()
        kl_sigma = Normal(self.sigma_mu, self.sigma_logstd.exp()).log_prob(sigma.log()).mean() - \
                   Normal(0, 1).log_prob(sigma.log()).mean()

        kl_divergence = kl_alpha + kl_beta.sum() + kl_sigma

        # ELBO = E[log p(y|theta)] - KL[q||p]
        elbo = E_log_likelihood - kl_divergence

        return -elbo  # Negative for minimization

def train_svi(X, y, batch_size=32, n_epochs=100):
    """
    Train using stochastic variational inference.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    n_total = len(X)

    model = BayesianLinearRegression(input_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(n_total)
        X_shuffled = X_tensor[perm]
        y_shuffled = y_tensor[perm]

        epoch_loss = 0
        n_batches = 0

        # Mini-batch loop
        for i in range(0, n_total, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            loss = model.elbo_loss(X_batch.squeeze(), y_batch, n_total)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Avg Loss: {avg_loss:.4f}")

    return model, losses

# Train model
model, losses = train_svi(X, y, batch_size=32, n_epochs=100)

# Extract posterior approximation
with torch.no_grad():
    alpha_samples = Normal(model.alpha_mu, model.alpha_logstd.exp()).sample((1000,))
    beta_samples = Normal(model.beta_mu, model.beta_logstd.exp()).sample((1000,))

    print(f"\nPosterior Mean:")
    print(f"  alpha: {alpha_samples.mean():.3f} (true: {true_alpha})")
    print(f"  beta: {beta_samples.mean():.3f} (true: {true_beta})")
```

## Common Pitfalls

**1. Mean-Field Assumption Too Restrictive**
- **Problem**: Variational distribution cannot capture posterior correlations
- **Symptom**: Underestimated uncertainty, poor posterior approximation
- **Solution**: Use full-rank Gaussian or normalizing flows

**2. Local Optima in ELBO**
- **Problem**: Optimization gets stuck in poor solutions
- **Symptom**: ELBO plateaus early, posterior mean far from MCMC
- **Solution**: Multiple random initializations, better optimizers (Adam), learning rate schedules

**3. Ignoring ELBO Convergence**
- **Problem**: Stopping optimization too early
- **Symptom**: Unstable posterior estimates
- **Solution**: Monitor ELBO convergence, use callbacks to check plateau

**4. High-Variance Gradients**
- **Problem**: Monte Carlo gradient estimates too noisy
- **Symptom**: Slow convergence, oscillating ELBO
- **Solution**: Increase MC samples for gradient, use control variates

**5. Overstating Certainty**
- **Problem**: VI typically underestimates posterior variance
- **Symptom**: Overconfident predictions
- **Solution**: Check against MCMC on subset, use ensemble of VI runs

## Connections

**Builds on:**
- Module 1: Bayesian inference fundamentals (posterior, likelihood, prior)
- Module 6.1: MCMC foundations (comparison baseline)
- Optimization theory (gradient ascent, Adam)

**Leads to:**
- Module 5.3: Sparse GP inference via SVGP (VI application)
- Production systems (fast inference for real-time predictions)
- Deep learning (variational autoencoders use VI)

**Related techniques:**
- Expectation Maximization (special case of VI)
- Laplace approximation (Gaussian approximation at MAP)
- Normalizing flows (flexible variational families)

## Practice Problems

1. **ELBO Decomposition**
   Given ELBO: $\mathcal{L} = \mathbb{E}_q[\log p(\mathbf{y}, \boldsymbol{\theta})] - \mathbb{E}_q[\log q(\boldsymbol{\theta})]$

   Show that: $\mathcal{L} = \log p(\mathbf{y}) - \text{KL}[q || p]$

   (Hint: Start with $\log p(\mathbf{y}) = \log \int p(\mathbf{y}, \boldsymbol{\theta}) d\boldsymbol{\theta}$)

2. **Mean-Field Limitations**
   True posterior: $p(\alpha, \beta | \mathbf{y})$ with strong correlation
   Mean-field: $q(\alpha, \beta) = q(\alpha) q(\beta)$

   - What information does mean-field lose?
   - How does this affect prediction intervals?
   - When is mean-field adequate?

3. **Computational Advantage**
   Dataset: n = 100,000 observations
   Parameters: d = 50

   - MCMC: 10,000 iterations, each requires full dataset pass
   - VI: 1,000 iterations, mini-batch size = 1,000

   Calculate speedup factor (ignore per-iteration cost differences)

4. **Convergence Diagnosis**
   You run VI for 10,000 iterations. ELBO values:
   - Iterations 1-1000: ELBO increases rapidly
   - Iterations 1000-5000: ELBO increases slowly
   - Iterations 5000-10000: ELBO oscillates around constant

   - Has VI converged?
   - What should you do?

5. **Commodity Forecasting Application**
   Model: Gaussian process with 1,000 inducing points
   Hyperparameters: {length_scale, signal_variance, noise_variance}

   - Design VI approximation: what variational family?
   - Estimate computational cost vs MCMC (order of magnitude)
   - What convergence diagnostics would you use?

## Further Reading

**Foundational Papers:**
1. **Jordan et al. (1999)** - "An Introduction to Variational Methods for Graphical Models" - Classic overview
2. **Blei et al. (2017)** - "Variational Inference: A Review for Statisticians" - Modern comprehensive review
3. **Kingma & Welling (2014)** - "Auto-Encoding Variational Bayes" - Reparameterization trick

**Automatic Differentiation VI:**
4. **Kucukelbir et al. (2017)** - "Automatic Differentiation Variational Inference" - ADVI algorithm
5. **PyMC Documentation** - Practical ADVI implementation

**Advanced Variational Families:**
6. **Rezende & Mohamed (2015)** - "Variational Inference with Normalizing Flows" - Flexible posteriors
7. **Ranganath et al. (2016)** - "Hierarchical Variational Models" - Structured variational families

**Stochastic VI:**
8. **Hoffman et al. (2013)** - "Stochastic Variational Inference" - Mini-batch VI
9. **Hensman et al. (2013)** - "Scalable Variational GP Classification" - SVGP derivation

**Diagnostics:**
10. **Yao et al. (2018)** - "Yes, but Did It Work?: Evaluating Variational Inference" - Quality checks

---

*"Variational inference trades sampling for optimization: instead of exploring the posterior, find the best simple approximation. Faster but approximate."*
