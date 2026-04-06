# Transition Matrices and State Dynamics

> **Reading time:** ~8 min | **Module:** Module 0: Foundations | **Prerequisites:** Basic linear algebra, Python

<div class="callout-key">

**Key Concept Summary:** The transition matrix is the heart of Markov processes. It encodes the probabilistic rules governing state evolution.

</div>

## Introduction

The transition matrix is the heart of Markov processes. It encodes the probabilistic rules governing state evolution.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Definition and Properties

### The Transition Matrix

For a Markov chain with $K$ states, the transition matrix $A$ is $K \times K$:

$$A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1K} \\
a_{21} & a_{22} & \cdots & a_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
a_{K1} & a_{K2} & \cdots & a_{KK}
\end{bmatrix}$$

where $a_{ij} = P(S_{t+1} = j | S_t = i)$

### Required Properties

1. **Non-negativity**: $a_{ij} \geq 0$ for all $i, j$
2. **Row-stochastic**: $\sum_j a_{ij} = 1$ for all $i$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def create_transition_matrix(n_states, style='random'):
    """
    Create different types of transition matrices.
    """
    if style == 'random':
        # Random transition matrix
        A = np.random.rand(n_states, n_states)
        A = A / A.sum(axis=1, keepdims=True)

    elif style == 'persistent':
        # High self-transition probabilities
        A = np.eye(n_states) * 0.8
        off_diag = (1 - 0.8) / (n_states - 1)
        A[A == 0] = off_diag

    elif style == 'cyclic':
        # Cycle through states
        A = np.zeros((n_states, n_states))
        for i in range(n_states):
            A[i, (i + 1) % n_states] = 0.9
            A[i, i] = 0.1

    elif style == 'absorbing':
        # One absorbing state
        A = np.random.rand(n_states, n_states)
        A = A / A.sum(axis=1, keepdims=True)
        A[-1, :] = 0
        A[-1, -1] = 1

    return A

# Examples
np.random.seed(42)

print("Different Transition Matrix Styles:")
print("=" * 60)

for style in ['random', 'persistent', 'cyclic', 'absorbing']:
    A = create_transition_matrix(3, style)
    print(f"\n{style.upper()}:")
    print(A.round(3))
    print(f"Row sums: {A.sum(axis=1).round(3)}")  # Should all be 1
```

## Multi-Step Transitions

### Chapman-Kolmogorov Equation

The probability of transitioning from state $i$ to state $j$ in $n$ steps:

$$P(S_{t+n} = j | S_t = i) = [A^n]_{ij}$$

```python
def analyze_multistep_transitions(A, max_steps=10):
    """
    Analyze transition probabilities over multiple steps.
    """
    n_states = A.shape[0]

    # Compute powers of A
    powers = [np.eye(n_states)]  # A^0 = I
    for step in range(1, max_steps + 1):
        powers.append(A @ powers[-1])

    # Plot transition from state 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Transition probabilities from state 0 over time
    ax1 = axes[0]
    for j in range(n_states):
        probs = [powers[n][0, j] for n in range(max_steps + 1)]
        ax1.plot(range(max_steps + 1), probs, 'o-', label=f'P(S_t={j}|S_0=0)')

    ax1.set_xlabel('Number of Steps')
    ax1.set_ylabel('Probability')
    ax1.set_title('Transition Probabilities from State 0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Matrix power convergence
    ax2 = axes[1]
    # Check if stationary distribution exists
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1))
    stationary = np.real(eigenvectors[:, stationary_idx])
    stationary = stationary / stationary.sum()

    for i in range(n_states):
        convergence = [np.abs(powers[n][0, i] - stationary[i]) for n in range(max_steps + 1)]
        ax2.plot(range(max_steps + 1), convergence, 'o-', label=f'State {i}')

    ax2.set_xlabel('Number of Steps')
    ax2.set_ylabel('|P^n - π|')
    ax2.set_title('Convergence to Stationary Distribution')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return powers, stationary

# Analyze a persistent transition matrix
A = create_transition_matrix(3, 'persistent')
print("Analyzing persistent transition matrix:")
print(A.round(3))

powers, stationary = analyze_multistep_transitions(A)
print(f"\nStationary distribution: {stationary.round(4)}")
```

## Stationary Distribution

### Definition

A stationary distribution $\pi$ satisfies:
$$\pi = \pi A$$

Or equivalently, $\pi$ is a left eigenvector of $A$ with eigenvalue 1.

### Computing the Stationary Distribution


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">compute_stationary_distribution.py</span>

```python
def compute_stationary_distribution(A, method='eigenvalue'):
    """
    Compute stationary distribution using different methods.
    """
    n_states = A.shape[0]

    if method == 'eigenvalue':
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        # Find eigenvector for eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()

    elif method == 'power':
        # Power iteration
        pi = np.ones(n_states) / n_states
        for _ in range(1000):
            pi_new = pi @ A
            if np.allclose(pi, pi_new):
                break
            pi = pi_new

    elif method == 'linear_solve':
        # Solve (A^T - I)π = 0 with constraint sum(π) = 1
        # Augment with normalization constraint
        B = np.vstack([A.T - np.eye(n_states), np.ones(n_states)])
        b = np.zeros(n_states + 1)
        b[-1] = 1

        pi, _, _, _ = np.linalg.lstsq(B, b, rcond=None)

    return pi

# Compare methods
A = create_transition_matrix(4, 'random')

print("Computing Stationary Distribution:")
print("=" * 60)

for method in ['eigenvalue', 'power', 'linear_solve']:
    pi = compute_stationary_distribution(A, method)
    # Verify: π = πA
    residual = np.max(np.abs(pi - pi @ A))
    print(f"\n{method.upper()} method:")
    print(f"  π = {pi.round(4)}")
    print(f"  ||π - πA|| = {residual:.2e}")
```

</div>
</div>

## Expected Hitting Times

### First Passage Time

Expected steps to reach state $j$ starting from state $i$:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">compute_hitting_times.py</span>

```python
def compute_hitting_times(A, target_state):
    """
    Compute expected hitting time to target state from all other states.
    """
    n_states = A.shape[0]

    # For hitting times, we solve: h_i = 1 + Σ_j≠target A[i,j] * h_j
    # This is a system of linear equations

    # Create reduced system (exclude target state)
    states = [i for i in range(n_states) if i != target_state]

    # A_reduced[i,j] = P(go from states[i] to states[j])
    A_reduced = A[np.ix_(states, states)]

    # Solve (I - A_reduced) h = 1
    I = np.eye(len(states))
    try:
        h_reduced = np.linalg.solve(I - A_reduced, np.ones(len(states)))

        # Full hitting time vector
        h = np.zeros(n_states)
        for idx, state in enumerate(states):
            h[state] = h_reduced[idx]

        return h
    except:
        return None

# Example
A = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.6, 0.3],
    [0.2, 0.3, 0.5]
])

print("Expected Hitting Times:")
print("=" * 60)
print(f"Transition matrix:\n{A}")

for target in range(3):
    h = compute_hitting_times(A, target)
    if h is not None:
        print(f"\nExpected steps to reach state {target}:")
        for i in range(3):
            if i != target:
                print(f"  From state {i}: {h[i]:.2f} steps")
```

</div>
</div>

## Classification of States

### Communicating Classes

States $i$ and $j$ communicate if you can get from $i$ to $j$ and from $j$ to $i$.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">if.py</span>

```python
def find_communicating_classes(A, max_power=100):
    """
    Find communicating classes in a transition matrix.
    """
    n_states = A.shape[0]

    # Reachability matrix: can reach j from i in some number of steps?
    reach = np.zeros((n_states, n_states), dtype=bool)

    A_power = np.eye(n_states)
    for _ in range(max_power):
        A_power = A_power @ A
        reach = reach | (A_power > 1e-10)

    # Communication: i <-> j if i -> j AND j -> i
    communicate = reach & reach.T

    # Find classes using union-find
    classes = list(range(n_states))

    for i in range(n_states):
        for j in range(i + 1, n_states):
            if communicate[i, j]:
                # Merge classes
                old_class = classes[j]
                new_class = classes[i]
                classes = [new_class if c == old_class else c for c in classes]

    # Group states by class
    class_dict = {}
    for state, cls in enumerate(classes):
        if cls not in class_dict:
            class_dict[cls] = []
        class_dict[cls].append(state)

    return list(class_dict.values()), reach

# Example with reducible chain
A_reducible = np.array([
    [0.5, 0.5, 0.0, 0.0],
    [0.3, 0.7, 0.0, 0.0],
    [0.0, 0.0, 0.6, 0.4],
    [0.0, 0.0, 0.2, 0.8]
])

classes, reach = find_communicating_classes(A_reducible)
print("Communicating Classes:")
print("=" * 60)
print(f"Transition matrix:\n{A_reducible}")
print(f"\nCommunicating classes: {classes}")
print(f"(This chain has {len(classes)} communicating classes - it's reducible)")
```

</div>
</div>

### Periodicity


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">compute_period.py</span>

```python
def compute_period(A, state=0, max_steps=100):
    """
    Compute the period of a state.
    """
    return_times = []

    A_power = A.copy()
    for step in range(1, max_steps + 1):
        if A_power[state, state] > 1e-10:
            return_times.append(step)
        A_power = A_power @ A

    if len(return_times) < 2:
        return 1  # Assume aperiodic if only one return

    # Period is GCD of return times
    from math import gcd
    from functools import reduce

    period = reduce(gcd, return_times)
    return period

# Aperiodic example
A_aperiodic = np.array([
    [0.5, 0.5],
    [0.3, 0.7]
])

# Periodic example (period 2)
A_periodic = np.array([
    [0.0, 1.0],
    [1.0, 0.0]
])

print("Periodicity Analysis:")
print("=" * 60)

print("\nAperiodic chain:")
print(A_aperiodic)
print(f"Period: {compute_period(A_aperiodic)}")

print("\nPeriodic chain:")
print(A_periodic)
print(f"Period: {compute_period(A_periodic)}")
```

</div>
</div>

## Visualization


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">visualize_transition_matrix.py</span>

```python
def visualize_transition_matrix(A, state_names=None):
    """
    Visualize transition matrix as heatmap and directed graph.
    """
    n_states = A.shape[0]
    if state_names is None:
        state_names = [f'S{i}' for i in range(n_states)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Heatmap
    ax1 = axes[0]
    im = ax1.imshow(A, cmap='Blues', vmin=0, vmax=1)

    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            color = 'white' if A[i, j] > 0.5 else 'black'
            ax1.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center', color=color)

    ax1.set_xticks(range(n_states))
    ax1.set_yticks(range(n_states))
    ax1.set_xticklabels(state_names)
    ax1.set_yticklabels(state_names)
    ax1.set_xlabel('To State')
    ax1.set_ylabel('From State')
    ax1.set_title('Transition Matrix')
    plt.colorbar(im, ax=ax1)

    # Right: Network representation
    ax2 = axes[1]

    # Position states in a circle
    angles = np.linspace(0, 2 * np.pi, n_states, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])

    # Draw edges (transitions)
    for i in range(n_states):
        for j in range(n_states):
            if A[i, j] > 0.01:  # Only draw significant transitions
                # Draw arrow
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]

                if i == j:  # Self-loop
                    circle = plt.Circle((positions[i, 0] + 0.2, positions[i, 1] + 0.2),
                                        0.15, fill=False, alpha=A[i, j])
                    ax2.add_patch(circle)
                else:
                    ax2.annotate('', xy=(positions[j, 0], positions[j, 1]),
                                 xytext=(positions[i, 0], positions[i, 1]),
                                 arrowprops=dict(arrowstyle='->', lw=A[i, j] * 3,
                                                 alpha=min(1, A[i, j] + 0.2),
                                                 connectionstyle='arc3,rad=0.1'))

    # Draw nodes
    for i in range(n_states):
        circle = plt.Circle(positions[i], 0.15, color='lightblue', ec='black', zorder=10)
        ax2.add_patch(circle)
        ax2.text(positions[i, 0], positions[i, 1], state_names[i],
                 ha='center', va='center', fontsize=12, zorder=11)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('State Transition Diagram')

    plt.tight_layout()
    plt.show()

# Example
A = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.6, 0.3],
    [0.3, 0.2, 0.5]
])

visualize_transition_matrix(A, ['Bull', 'Neutral', 'Bear'])
```

</div>
</div>

<div class="callout-insight">

**Insight:** Understanding transition matrices and state dynamics is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Transition matrices encode state dynamics** - row $i$ gives probabilities of next state given current state $i$

2. **Multi-step transitions** are computed via matrix powers: $P^n$

3. **Stationary distributions** represent long-run behavior

4. **Expected hitting times** measure reachability between states

5. **Communication classes and periodicity** characterize chain structure

---

## Conceptual Practice Questions

1. How do you interpret the entries of a transition matrix in a financial context?

2. What does it mean for a Markov chain to be ergodic, and why does this matter?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./02_transition_matrices_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_markov_chain_basics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_markov_chains.md">
  <div class="link-card-title">01 Markov Chains</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_probability_review.md">
  <div class="link-card-title">02 Probability Review</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

