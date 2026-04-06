# Machine Learning Connections to Factor Models

> **Reading time:** ~16 min | **Module:** Module 8: Advanced Topics | **Prerequisites:** Modules 0-7

<div class="callout-key">

**Key Concept Summary:** Linear autoencoders with tied weights are mathematically equivalent to PCA—both seek low-dimensional representations minimizing reconstruction error. Deep neural networks generalize this to nonlinear factor structures, while variational autoencoders add probabilistic interpretations. Understandin...

</div>

## In Brief

Linear autoencoders with tied weights are mathematically equivalent to PCA—both seek low-dimensional representations minimizing reconstruction error. Deep neural networks generalize this to nonlinear factor structures, while variational autoencoders add probabilistic interpretations. Understanding these connections reveals when to use traditional econometric methods versus modern machine learning, and how to combine both.

<div class="callout-insight">

**Insight:** Factor models and neural networks solve the same fundamental problem: find low-dimensional structure in high-dimensional data. Traditional factor models assume linearity and provide statistical theory (standard errors, hypothesis tests). Deep learning relaxes linearity but sacrifices interpretability and theory. The frontier lies in hybrid approaches that preserve economic interpretability while leveraging neural network flexibility.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. Linear Autoencoders and PCA

### Formal Definition

**Linear Autoencoder:**
$$\text{Encoder: } F_t = W_1^T X_t$$
$$\text{Decoder: } \hat{X}_t = W_2 F_t$$

Minimize reconstruction error:
$$\min_{W_1, W_2} \sum_{t=1}^T \|X_t - W_2 W_1^T X_t\|^2$$

**Principal Component Analysis (PCA):**
$$\min_{\Lambda, F} \sum_{t=1}^T \|X_t - \Lambda F_t\|^2$$

subject to $\Lambda^T \Lambda = I$ and $\frac{1}{T} F^T F = \text{diag}(\sigma_1^2, \ldots, \sigma_r^2)$.

### Mathematical Equivalence Theorem

**Theorem:** A linear autoencoder with tied weights ($W_2 = W_1$) and orthogonality constraint on $W_1$ is equivalent to PCA.

**Proof Sketch:**

1. Objective function with tied weights:
$$\mathcal{L} = \sum_t \|X_t - W_1 W_1^T X_t\|^2$$

2. Take derivative with respect to $W_1$:
$$\frac{\partial \mathcal{L}}{\partial W_1} = -2 \sum_t (X_t - W_1 W_1^T X_t) X_t^T$$

3. Setting to zero:
$$\sum_t X_t X_t^T W_1 = \sum_t X_t X_t^T W_1 W_1^T W_1$$

4. With $W_1^T W_1 = I$:
$$\Sigma_X W_1 = \Sigma_X W_1$$

This is the eigenvalue problem! Columns of $W_1$ are eigenvectors of $\Sigma_X$.

**Key Differences Without Constraints:**

| Feature | PCA | Linear Autoencoder (unconstrained) |
|---------|-----|-----------------------------------|
| Weight constraint | $W_1^T W_1 = I$ | None |
| Encoder/decoder | Tied ($W_2 = W_1$) | Separate |
| Solution | Unique (up to rotation) | Non-unique |
| Interpretation | Variance maximization | Reconstruction minimization |

### Code Implementation: Verifying Equivalence


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">linearautoencoder.py</span>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim

class LinearAutoencoder(nn.Module):
    """
    Linear autoencoder with optional weight tying.

    Architecture:
        Encoder: X (N dim) -> F (r dim)
        Decoder: F (r dim) -> X_hat (N dim)
    """

    def __init__(self, n_input, n_latent, tied_weights=True):
        """
        Parameters
        ----------
        n_input : int
            Input dimension (N variables)
        n_latent : int
            Latent dimension (r factors)
        tied_weights : bool
            If True, decoder weights = encoder weights^T
        """
        super(LinearAutoencoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.tied_weights = tied_weights

        # Encoder: X -> F
        self.encoder = nn.Linear(n_input, n_latent, bias=False)

        # Decoder: F -> X_hat
        if not tied_weights:
            self.decoder = nn.Linear(n_latent, n_input, bias=False)

    def forward(self, x):
        """
        Forward pass: encode then decode.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, n_input)

        Returns
        -------
        x_reconstructed : torch.Tensor, shape (batch_size, n_input)
        latent : torch.Tensor, shape (batch_size, n_latent)
        """
        latent = self.encoder(x)

        if self.tied_weights:
            # Tied weights: decoder = encoder^T
            x_reconstructed = torch.matmul(latent, self.encoder.weight)
        else:
            x_reconstructed = self.decoder(latent)

        return x_reconstructed, latent

    def get_loadings(self):
        """
        Extract encoder weights (factor loadings).

        Returns
        -------
        loadings : ndarray, shape (n_input, n_latent)
        """
        return self.encoder.weight.data.cpu().numpy().T


def train_autoencoder(model, X, n_epochs=1000, lr=0.01, verbose=False):
    """
    Train linear autoencoder.

    Parameters
    ----------
    model : LinearAutoencoder
        Model to train
    X : ndarray, shape (T, N)
        Training data
    n_epochs : int
        Number of training epochs
    lr : float
        Learning rate
    verbose : bool
        Print progress

    Returns
    -------
    losses : list
        Training loss history
    """
    X_tensor = torch.FloatTensor(X)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(n_epochs):
        # Forward pass
        X_reconstructed, _ = model(X_tensor)
        loss = criterion(X_reconstructed, X_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    return losses


def compare_pca_autoencoder(X, n_components=2):
    """
    Compare PCA with linear autoencoder (tied weights).

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Data matrix
    n_components : int
        Number of components/factors

    Returns
    -------
    results : dict
        Comparison results
    """
    T, N = X.shape

    # 1. PCA
    print("Fitting PCA...")
    pca = PCA(n_components=n_components)
    F_pca = pca.fit_transform(X)
    Lambda_pca = pca.components_.T  # (N, r)
    X_reconstructed_pca = F_pca @ Lambda_pca.T
    mse_pca = np.mean((X - X_reconstructed_pca) ** 2)

    # 2. Linear Autoencoder (tied weights)
    print("Training Linear Autoencoder (tied weights)...")
    ae_tied = LinearAutoencoder(N, n_components, tied_weights=True)
    losses_tied = train_autoencoder(ae_tied, X, n_epochs=1000, lr=0.01, verbose=False)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        X_reconstructed_ae, F_ae = ae_tied(X_tensor)
        X_reconstructed_ae = X_reconstructed_ae.numpy()
        F_ae = F_ae.numpy()

    Lambda_ae = ae_tied.get_loadings()
    mse_ae = np.mean((X - X_reconstructed_ae) ** 2)

    # 3. Linear Autoencoder (separate weights)
    print("Training Linear Autoencoder (separate weights)...")
    ae_separate = LinearAutoencoder(N, n_components, tied_weights=False)
    losses_separate = train_autoencoder(ae_separate, X, n_epochs=1000, lr=0.01, verbose=False)

    with torch.no_grad():
        X_reconstructed_ae_sep, _ = ae_separate(X_tensor)
        X_reconstructed_ae_sep = X_reconstructed_ae_sep.numpy()

    mse_ae_separate = np.mean((X - X_reconstructed_ae_sep) ** 2)

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"PCA MSE: {mse_pca:.6f}")
    print(f"Autoencoder (tied) MSE: {mse_ae:.6f}")
    print(f"Autoencoder (separate) MSE: {mse_ae_separate:.6f}")

    # Compare loadings (up to sign and rotation)
    print(f"\nLoading correlation (PCA vs AE tied):")
    for i in range(n_components):
        corr = np.corrcoef(Lambda_pca[:, i], Lambda_ae[:, i])[0, 1]
        print(f"  Factor {i}: {np.abs(corr):.4f}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Training loss
    axes[0].plot(losses_tied, label='Tied weights')
    axes[0].plot(losses_separate, label='Separate weights')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Loading comparison
    axes[1].scatter(Lambda_pca[:, 0], Lambda_ae[:, 0], alpha=0.6)
    axes[1].plot([-2, 2], [-2, 2], 'r--', alpha=0.5)
    axes[1].set_xlabel('PCA Loading (Factor 1)')
    axes[1].set_ylabel('Autoencoder Loading (Factor 1)')
    axes[1].set_title('Loading Comparison')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Reconstruction comparison
    axes[2].scatter(X_reconstructed_pca.ravel(), X_reconstructed_ae.ravel(), alpha=0.1)
    axes[2].plot([X.min(), X.max()], [X.min(), X.max()], 'r--', alpha=0.5)
    axes[2].set_xlabel('PCA Reconstruction')
    axes[2].set_ylabel('Autoencoder Reconstruction')
    axes[2].set_title('Reconstruction Comparison')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'pca': {'loadings': Lambda_pca, 'factors': F_pca, 'mse': mse_pca},
        'ae_tied': {'loadings': Lambda_ae, 'factors': F_ae, 'mse': mse_ae},
        'ae_separate': {'mse': mse_ae_separate}
    }


# Example: Verify equivalence
np.random.seed(42)
torch.manual_seed(42)

T, N, r_true = 200, 20, 3

# Generate factor model data
F_true = np.random.randn(T, r_true)
Lambda_true = np.random.randn(N, r_true)
X_data = F_true @ Lambda_true.T + np.random.randn(T, N) * 0.5

# Standardize
X_data = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)

# Compare methods
results = compare_pca_autoencoder(X_data, n_components=r_true)
```

</div>
</div>

---

## 2. Deep Autoencoders for Nonlinear Factors

### Motivation for Nonlinearity

**Linear factor model limitation:**
$$X_t = \Lambda F_t + e_t$$

Assumes variables are **linear combinations** of factors.

**Nonlinear reality:**
- Volatility (quadratic in returns)
- Regime-dependent relationships
- Interaction effects
- Threshold effects

**Example:** Stock return $R_t$ and volatility $\text{Vol}_t$ both driven by latent "risk sentiment" factor $F_t$:
$$R_t = \lambda_R F_t + e_t$$
$$\text{Vol}_t = \lambda_V F_t^2 + \varepsilon_t$$

Linear factor model cannot capture $\text{Vol}_t$ dependence!

### Deep Autoencoder Architecture

**Nonlinear Encoder:**
$$h_1 = \sigma(W_1 X + b_1)$$
$$h_2 = \sigma(W_2 h_1 + b_2)$$
$$F = W_3 h_2 + b_3 \quad \text{(bottleneck layer)}$$

**Nonlinear Decoder:**
$$h_3 = \sigma(W_4 F + b_4)$$
$$h_4 = \sigma(W_5 h_3 + b_5)$$
$$\hat{X} = W_6 h_4 + b_6$$

where $\sigma(\cdot)$ is activation function (ReLU, tanh, sigmoid).

**Objective:**
$$\min_{\{W_i, b_i\}} \sum_t \|X_t - \hat{X}_t\|^2 + \lambda R(W)$$

where $R(W)$ is regularization (L2, dropout).

**Universal Approximation:** Deep networks can approximate any continuous function arbitrarily well.

### Code Implementation: Deep Autoencoder


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">deepautoencoder.py</span>

```python
class DeepAutoencoder(nn.Module):
    """
    Deep autoencoder for nonlinear factor extraction.

    Architecture:
        Encoder: X -> h1 -> h2 -> F (bottleneck)
        Decoder: F -> h3 -> h4 -> X_hat
    """

    def __init__(self, n_input, n_latent, hidden_dims=[64, 32]):
        """
        Parameters
        ----------
        n_input : int
            Input dimension
        n_latent : int
            Latent factor dimension
        hidden_dims : list
            Hidden layer dimensions
        """
        super(DeepAutoencoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent

        # Encoder
        encoder_layers = []
        prev_dim = n_input
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim

        encoder_layers.append(nn.Linear(prev_dim, n_latent))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = n_latent
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, n_input))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Forward pass.

        Returns
        -------
        x_reconstructed, latent_factors
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def extract_factors(self, X):
        """
        Extract latent factors from data.

        Parameters
        ----------
        X : ndarray, shape (T, N)

        Returns
        -------
        F : ndarray, shape (T, r)
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            _, F = self.forward(X_tensor)
            return F.numpy()


def train_deep_autoencoder(model, X, n_epochs=200, batch_size=32, lr=0.001, verbose=True):
    """
    Train deep autoencoder with mini-batch SGD.

    Parameters
    ----------
    model : DeepAutoencoder
    X : ndarray, shape (T, N)
    n_epochs : int
    batch_size : int
    lr : float
    verbose : bool

    Returns
    -------
    losses : list
    """
    X_tensor = torch.FloatTensor(X)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for batch in dataloader:
            X_batch = batch[0]

            # Forward pass
            X_reconstructed, _ = model(X_batch)
            loss = criterion(X_reconstructed, X_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)

        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")

    return losses


def compare_linear_deep_autoencoder(X, n_factors=3):
    """
    Compare linear PCA vs deep nonlinear autoencoder.

    Parameters
    ----------
    X : ndarray, shape (T, N)
    n_factors : int

    Returns
    -------
    results : dict
    """
    T, N = X.shape

    # Linear: PCA
    print("Fitting PCA (linear)...")
    pca = PCA(n_components=n_factors)
    F_linear = pca.fit_transform(X)
    X_reconstructed_linear = pca.inverse_transform(F_linear)
    mse_linear = np.mean((X - X_reconstructed_linear) ** 2)

    # Nonlinear: Deep Autoencoder
    print("Training Deep Autoencoder (nonlinear)...")
    deep_ae = DeepAutoencoder(N, n_factors, hidden_dims=[32, 16])
    losses = train_deep_autoencoder(deep_ae, X, n_epochs=100, batch_size=32, lr=0.001, verbose=True)

    F_deep = deep_ae.extract_factors(X)
    deep_ae.eval()
    with torch.no_grad():
        X_reconstructed_deep = deep_ae.decoder(torch.FloatTensor(F_deep)).numpy()

    mse_deep = np.mean((X - X_reconstructed_deep) ** 2)

    print("\n" + "="*60)
    print("LINEAR vs NONLINEAR COMPARISON")
    print("="*60)
    print(f"PCA (linear) MSE: {mse_linear:.6f}")
    print(f"Deep AE (nonlinear) MSE: {mse_deep:.6f}")
    print(f"Improvement: {(1 - mse_deep/mse_linear)*100:.2f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Deep Autoencoder Training')
    axes[0].grid(True, alpha=0.3)

    # Factor comparison (first factor)
    axes[1].scatter(F_linear[:, 0], F_deep[:, 0], alpha=0.5)
    axes[1].set_xlabel('Linear Factor 1 (PCA)')
    axes[1].set_ylabel('Nonlinear Factor 1 (Deep AE)')
    axes[1].set_title('Factor Comparison')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'linear': {'factors': F_linear, 'mse': mse_linear},
        'deep': {'factors': F_deep, 'mse': mse_deep}
    }


# Example with nonlinear data
np.random.seed(42)
torch.manual_seed(42)

T, N, r = 500, 30, 3

# Generate nonlinear factor structure
F_true = np.random.randn(T, r)

# Nonlinear loadings: some variables quadratic in factors
Lambda_linear = np.random.randn(N // 2, r) * 0.5
X_linear = F_true @ Lambda_linear.T

# Nonlinear relationships
X_nonlinear = np.zeros((T, N // 2))
for i in range(N // 2):
    # Mix of linear and quadratic
    X_nonlinear[:, i] = (
        0.5 * F_true[:, i % r] +
        0.3 * F_true[:, (i+1) % r] ** 2 +
        np.random.randn(T) * 0.2
    )

X_data = np.hstack([X_linear, X_nonlinear])
X_data = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)

print("Data contains both linear and nonlinear factor relationships.")
results = compare_linear_deep_autoencoder(X_data, n_factors=r)
```

</div>
</div>

---

## 3. Variational Autoencoders (VAE)

### Probabilistic Factor Model Interpretation

**Traditional Factor Model (Probabilistic):**
$$p(X | F, \Lambda, \Sigma) = N(X; \Lambda F, \Sigma)$$
$$p(F) = N(F; 0, I)$$

**Variational Autoencoder:**
$$p(X | F; \theta) = N(X; \mu_\theta(F), \Sigma_\theta(F))$$
$$p(F) = N(F; 0, I)$$
$$q(F | X; \phi) = N(F; \mu_\phi(X), \Sigma_\phi(X))$$

where $\mu_\theta(\cdot)$, $\Sigma_\theta(\cdot)$ are decoder networks, $\mu_\phi(\cdot)$, $\Sigma_\phi(\cdot)$ are encoder networks.

### Evidence Lower Bound (ELBO)

**Objective:** Maximize marginal likelihood $\log p(X)$

**ELBO:**
$$\log p(X) \geq \mathbb{E}_{q(F|X)}[\log p(X|F)] - D_{KL}(q(F|X) \| p(F))$$

**Two terms:**
1. **Reconstruction:** $\mathbb{E}[\log p(X|F)]$ (like standard autoencoder)
2. **Regularization:** $D_{KL}$ penalizes divergence from prior

**Benefits:**
- Uncertainty quantification (variance of $q(F|X)$)
- Regularization prevents overfitting
- Generative model (can sample new data)

### Code Implementation: VAE for Factors


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">variationalautoencoder.py</span>

```python
class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for probabilistic factor extraction.

    Learns distribution q(F|X) over latent factors.
    """

    def __init__(self, n_input, n_latent, hidden_dim=64):
        super(VariationalAutoencoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent

        # Encoder: X -> (mu, log_var)
        self.encoder_hidden = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, n_latent)
        self.fc_logvar = nn.Linear(hidden_dim, n_latent)

        # Decoder: F -> X
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, n_input)
        )

    def encode(self, x):
        """
        Encode X to latent distribution parameters.

        Returns
        -------
        mu, log_var : torch.Tensor
        """
        h = self.encoder_hidden(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: F = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent factors to reconstructed X."""
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass.

        Returns
        -------
        x_reconstructed, mu, log_var
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

    def loss_function(self, x_reconstructed, x, mu, log_var):
        """
        VAE loss = Reconstruction loss + KL divergence.

        Returns
        -------
        loss, recon_loss, kl_loss
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_reconstructed, x, reduction='sum')

        # KL divergence: D_KL(q(F|X) || N(0, I))
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + kl_loss, recon_loss, kl_loss


def train_vae(model, X, n_epochs=200, batch_size=32, lr=0.001, verbose=True):
    """Train Variational Autoencoder."""
    X_tensor = torch.FloatTensor(X)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = {'total': [], 'recon': [], 'kl': []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0

        for batch in dataloader:
            X_batch = batch[0]

            # Forward
            X_recon, mu, log_var = model(X_batch)
            loss, recon, kl = model.loss_function(X_recon, X_batch, mu, log_var)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        n_batches = len(dataloader)
        losses['total'].append(epoch_loss / n_batches)
        losses['recon'].append(epoch_recon / n_batches)
        losses['kl'].append(epoch_kl / n_batches)

        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={losses['total'][-1]:.2f}, "
                  f"Recon={losses['recon'][-1]:.2f}, KL={losses['kl'][-1]:.2f}")

    return losses


# Example: VAE for factor extraction
print("Training Variational Autoencoder...")
vae = VariationalAutoencoder(N, r, hidden_dim=32)
vae_losses = train_vae(vae, X_data, n_epochs=100, batch_size=32, lr=0.001, verbose=True)

# Extract factors with uncertainty
vae.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_data)
    mu, log_var = vae.encode(X_tensor)
    F_vae = mu.numpy()
    F_std = torch.exp(0.5 * log_var).numpy()

print(f"\nExtracted factors with uncertainty:")
print(f"Factor 1 - Mean: {F_vae[0, 0]:.3f}, Std: {F_std[0, 0]:.3f}")

# Plot losses
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(vae_losses['total'], label='Total Loss')
ax.plot(vae_losses['recon'], label='Reconstruction')
ax.plot(vae_losses['kl'], label='KL Divergence')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('VAE Training')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

</div>
</div>

---

## 4. When to Use ML vs Traditional Factor Models

### Comparison Table

| Criterion | Traditional Factor Model | Neural Network Autoencoder |
|-----------|-------------------------|---------------------------|
| **Linearity** | Assumes linear factors | Can capture nonlinearity |
| **Interpretability** | High (economic meaning) | Low (black box) |
| **Statistical Theory** | Complete (SEs, tests) | Limited/none |
| **Sample Size Needs** | Moderate (T > N) | Large (T >> N) |
| **Computational Cost** | Low (closed form) | High (iterative optimization) |
| **Overfitting Risk** | Low | High (needs regularization) |
| **Out-of-Sample** | Generally good | Can be poor without care |
| **Use Case** | Inference, forecasting | Prediction with big data |

### Decision Framework

**Use Traditional Factor Models When:**
<div class="flow">
<div class="flow-step mint">1. Interpretation matte...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Statistical inferenc...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Small/moderate sampl...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Linear relationships...</div>

</div>


1. **Interpretation matters** (policy analysis, structural models)
2. **Statistical inference needed** (hypothesis tests, confidence intervals)
3. **Small/moderate sample** (T < 1000)
4. **Linear relationships suffice** (macro aggregates often linear)
5. **Transparency required** (regulatory, academic)

**Use Neural Network Autoencoders When:**
1. **Large datasets** (T > 10,000)
2. **Nonlinear structure evident** (interactions, thresholds)
3. **Prediction primary goal** (less concern for why)
4. **Rich features** (text, images embedded in factors)
5. **Computational resources available**

**Hybrid Approaches:**
1. **PCA + Nonlinear Prediction:** Extract linear factors, use neural network for forecasting
2. **Interpretable Constraints:** Add economic constraints to neural networks (monotonicity, sparsity)
3. **Ensemble Methods:** Average forecasts from both approaches
4. **Two-Step:** Screen variables with neural network, estimate interpretable model on subset

---

## Common Pitfalls

### 1. Overfitting Neural Networks on Time Series
- **Mistake**: Training deep autoencoder on small T with many parameters
- **Result**: Perfect in-sample fit, poor out-of-sample
- **Fix**: Strong regularization (dropout, early stopping), cross-validation, simpler architecture

### 2. Ignoring Temporal Dependence
- **Mistake**: Treating observations as i.i.d. when training neural networks
- **Result**: Overstated prediction accuracy, data snooping
- **Fix**: Use time-series cross-validation, respect temporal ordering

### 3. Confusing Reconstruction Error with Predictive Accuracy
- **Mistake**: Choosing model based on lowest autoencoder reconstruction loss
- **Result**: Overfitting to in-sample data
- **Fix**: Evaluate on out-of-sample forecasting task

### 4. Sacrificing Interpretability Unnecessarily
- **Mistake**: Using complex neural network when linear model performs similarly
- **Result**: Lost economic insights without performance gain
- **Fix**: Always compare to linear baseline; justify added complexity

---

## Connections

- **Builds on:** PCA, factor models, statistical learning theory
- **Leads to:** Deep learning for time series, hybrid econometric-ML models
- **Related to:** Dimensionality reduction, manifold learning, representation learning

---

## Practice Problems

### Conceptual

1. Prove that a linear autoencoder with tied weights and orthonormal encoder weights reduces to PCA.

2. Explain why VAE's KL divergence term acts as regularization. What happens if you remove it?

3. When would nonlinear factors be essential for macroeconomic forecasting? Give specific examples.

### Implementation

4. Implement a "denoising autoencoder" that adds noise to inputs during training. Compare to standard autoencoder.

5. Create a hybrid model: extract PCA factors, use them as inputs to a neural network forecasting model.

6. Extend the VAE to predict one-step-ahead $X_{t+1}$ given latent factors $F_t$.

### Extension

7. Research "disentangled representations" in VAE literature. How could this improve factor interpretability?

8. Implement attention mechanism in autoencoder to identify which variables contribute to each factor.

9. Compare computational cost: PCA, linear autoencoder (SGD), deep autoencoder. Plot time vs (T, N).

---

## Further Reading

- **Goodfellow, I., Bengio, Y. & Courville, A.** (2016). *Deep Learning*. MIT Press. Chapter 14 (Autoencoders).
  - Comprehensive treatment of autoencoder architectures

- **Kingma, D.P. & Welling, M.** (2014). "Auto-encoding variational Bayes." *ICLR*.
  - Original VAE paper with clear derivations

- **Gu, S., Kelly, B. & Xiu, D.** (2020). "Empirical asset pricing via machine learning." *Review of Financial Studies*, 33(5), 2223-2273.
  - ML methods for factor models in finance

- **Kelly, B., Pruitt, S. & Su, Y.** (2019). "Characteristics are covariances: A unified model of risk and return." *Journal of Financial Economics*, 134(3), 501-524.
  - Instrumented PCA connecting characteristics and factors

- **Chen, L., Pelger, M. & Zhu, J.** (2023). "Deep learning in asset pricing." *Management Science*, forthcoming.
  - Neural networks for factor extraction and pricing

- **Lettau, M. & Pelger, M.** (2020). "Factors that fit the time series and cross-section of stock returns." *Review of Financial Studies*, 33(5), 2274-2325.
  - Advances in statistical factor models

- **Han, Y., He, A. & Rapach, D.** (2021). "Deep learning in asset pricing." *Working Paper*.
  - Comprehensive survey of neural network methods in finance

---

<div class="callout-insight">

**Insight:** Understanding machine learning connections to factor models is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Summary

**Key Takeaways:**

1. **Linear autoencoders with tied weights are mathematically equivalent to PCA**—both minimize reconstruction error
2. **Deep autoencoders** generalize to nonlinear factor structures but sacrifice interpretability and statistical theory
3. **Variational autoencoders** add probabilistic interpretation and uncertainty quantification via ELBO objective
4. **Trade-offs are fundamental**: interpretability/theory vs flexibility/prediction
5. **Hybrid approaches** combining traditional econometrics with machine learning often perform best in practice

**Course Conclusion:**

You've now completed a comprehensive journey through factor models:
- **Static factors** (PCA, maximum likelihood)
- **Dynamic factors** (state-space, Kalman filtering)
- **Sparse methods** (variable selection, LASSO)
- **Mixed-frequency models** (nowcasting applications)
- **Advanced topics** (time-varying parameters, non-Gaussian errors, ML connections)

**Key Skills Acquired:**
- Extract interpretable common factors from high-dimensional data
- Implement Kalman filter and EM algorithm for dynamic models
- Build forecasting systems combining factors with other predictors
- Select relevant variables in high dimensions
- Apply modern robust and machine learning methods
- Critically evaluate trade-offs between traditional and modern approaches

**Next Steps:**
- Apply these methods to your domain (macro, finance, marketing, etc.)
- Read cutting-edge research papers in top journals
- Contribute to open-source factor modeling libraries
- Develop novel extensions addressing unsolved problems

**The field continues to evolve rapidly. Stay curious, rigorous, and always validate with out-of-sample data!**

---

## Conceptual Practice Questions

1. In your own words, explain the difference between common factors and idiosyncratic components.

2. Why do factor models require identification restrictions? Give a concrete example.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./03_ml_connections_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_time_varying_factors.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_time_varying_parameters.md">
  <div class="link-card-title">01 Time Varying Parameters</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_non_gaussian_factors.md">
  <div class="link-card-title">02 Non Gaussian Factors</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

