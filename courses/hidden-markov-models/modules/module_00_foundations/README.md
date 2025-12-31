# Module 0: Foundations

## Overview

Review the probabilistic and mathematical foundations for Hidden Markov Models. Understand Markov chains and the concepts underlying hidden state models.

**Time Estimate:** 4-6 hours

## Learning Objectives

By completing this module, you will:
1. Understand discrete-time Markov chains
2. Review key probability concepts
3. Motivate the hidden state framework
4. Set up your computational environment

## Contents

### Guides
- `01_markov_chains.md` - Markov property and chains
- `02_probability_review.md` - Conditional probability, Bayes
- `03_hidden_states.md` - Why "hidden"?

### Notebooks
- `01_markov_chain_basics.ipynb` - Markov chain simulations
- `02_environment_setup.ipynb` - hmmlearn, pomegranate

## Key Concepts

### Markov Property

$$P(X_{t+1} | X_t, X_{t-1}, ..., X_1) = P(X_{t+1} | X_t)$$

"The future depends only on the present, not the past."

### Transition Matrix

For states $S = \{s_1, s_2, ..., s_K\}$:

$$A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1K} \\
a_{21} & a_{22} & \cdots & a_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
a_{K1} & a_{K2} & \cdots & a_{KK}
\end{pmatrix}$$

Where $a_{ij} = P(X_{t+1} = s_j | X_t = s_i)$

### Observable vs Hidden

| Observable Markov | Hidden Markov |
|-------------------|---------------|
| States are known | States are hidden |
| Observe states directly | Observe emissions |
| Deterministic observation | Probabilistic observation |

### Motivation

Financial regimes (bull/bear) are hidden—we only observe prices. HMMs model this:

```
Hidden States:  [Bull] → [Bear] → [Bull] → ...
                   ↓        ↓        ↓
Observations:   [+2%]    [-3%]    [+1%]   → (Returns)
```

## Prerequisites

- Probability basics
- Linear algebra
- Python proficiency
