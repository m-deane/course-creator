# Meta-Learning for Feature Selection

## In Brief

Meta-learning applies machine learning to the problem of selecting machine learning algorithms. For feature selection, a meta-learner observes properties of a dataset (meta-features) and predicts which feature selection method will perform best, replacing expensive trial-and-error exploration with a principled recommendation system.

## Key Insight

Every dataset has a measurable character: its dimensionality, class balance, feature correlation structure, linearity of target relationship, and noise level. These meta-features predict which selector works best. By training a meta-model on many (dataset, best-selector) pairs, we automate the algorithm selection problem — the same problem solved manually by expert practitioners.

---

## Feature Selection as a Meta-Learning Problem

### The Algorithm Selection Problem

Given a dataset $\mathcal{D} = (X, y)$, we want to identify the feature selector $A^*$ from a portfolio $\mathcal{A} = \{A_1, \ldots, A_k\}$ that maximises downstream model performance:

$$A^*(\mathcal{D}) = \arg\max_{A \in \mathcal{A}} \text{Performance}(\text{Model}(A(\mathcal{D})))$$

Solving this directly requires running all $|\mathcal{A}|$ selectors on $\mathcal{D}$ — expensive. The meta-learning shortcut: learn a function $f: \mathcal{M}(\mathcal{D}) \to \mathcal{A}$ where $\mathcal{M}(\mathcal{D})$ is a vector of cheap-to-compute meta-features of $\mathcal{D}$.

Once $f$ is trained on a diverse collection of datasets, applying it to a new dataset costs only the time to compute meta-features (seconds) rather than running all selectors (hours).

### The CASH Problem

The **Combined Algorithm Selection and Hyperparameter** (CASH) problem extends algorithm selection to also optimise hyperparameters:

$$A^*, \lambda^* = \arg\max_{A \in \mathcal{A}, \lambda \in \Lambda_A} \text{CV}(A(\mathcal{D}; \lambda))$$

AutoML systems (Auto-sklearn, FLAML, H2O AutoML) solve the full CASH problem using Bayesian optimisation over the joint (algorithm, hyperparameter) space. For feature selection, the CASH formulation includes:
- Which selector(s) to use
- Hyperparameters of each selector (regularisation strength, number of features, population size)
- Whether to use a cascade and at which sizes

---

## Dataset Meta-Features

Meta-features are dataset properties that are:
1. **Cheap to compute:** at most $O(np^2)$
2. **Predictive of selector performance:** empirically validated
3. **Scale-invariant:** comparable across datasets of different sizes

### Statistical Meta-Features

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

def compute_statistical_meta_features(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Statistical meta-features describing the dataset's distributional properties.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)

    Returns
    -------
    dict
        Meta-feature name → scalar value.
    """
    n, p = X.shape

    mf = {}

    # --- Basic dimensionality ---
    mf['n_samples'] = n
    mf['n_features'] = p
    mf['log_n'] = np.log1p(n)
    mf['log_p'] = np.log1p(p)
    mf['log_ratio_n_p'] = np.log1p(n / p)   # n/p ratio (curse of dimensionality proxy)

    # --- Target distribution ---
    if len(np.unique(y)) <= 20:  # classification
        class_counts = np.bincount(y.astype(int))
        mf['n_classes'] = len(class_counts)
        mf['class_balance'] = class_counts.min() / class_counts.max()   # 1.0 = balanced
        mf['class_entropy'] = stats.entropy(class_counts / n)
    else:  # regression
        mf['n_classes'] = 1
        mf['class_balance'] = 1.0
        mf['class_entropy'] = 0.0
        mf['target_skew'] = float(stats.skew(y))
        mf['target_kurtosis'] = float(stats.kurtosis(y))

    # --- Feature statistics ---
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    feature_skews = stats.skew(X, axis=0)
    feature_kurtoses = stats.kurtosis(X, axis=0)

    mf['mean_feature_skew'] = float(np.mean(np.abs(feature_skews)))
    mf['mean_feature_kurtosis'] = float(np.mean(feature_kurtoses))
    mf['frac_near_constant'] = float(np.mean(feature_stds < 0.01))

    return mf
```

### Correlation Structure Meta-Features

The correlation structure of the feature matrix is one of the strongest predictors of which selector works best.

```python
def compute_correlation_meta_features(X: np.ndarray) -> dict:
    """
    Correlation-based meta-features: captures inter-feature redundancy.
    Uses a sample of features for large p to control compute cost.
    """
    n, p = X.shape

    # Subsample features if p is large (avoid O(p^2) bottleneck)
    sample_p = min(p, 200)
    if p > 200:
        rng = np.random.default_rng(42)
        feature_idx = rng.choice(p, size=sample_p, replace=False)
        X_sample = X[:, feature_idx]
    else:
        X_sample = X

    corr_matrix = np.corrcoef(X_sample.T)
    np.fill_diagonal(corr_matrix, 0)  # exclude self-correlation

    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    abs_upper = np.abs(upper_tri)

    mf = {}
    mf['mean_abs_correlation'] = float(np.mean(abs_upper))
    mf['max_abs_correlation'] = float(np.max(abs_upper))
    mf['frac_highly_correlated'] = float(np.mean(abs_upper > 0.7))
    mf['std_abs_correlation'] = float(np.std(abs_upper))

    # Effective rank: number of principal components explaining 95% variance
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, sample_p, n - 1))
    pca.fit(X_sample)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    mf['effective_rank_ratio'] = n_components_95 / sample_p   # 1.0 = no redundancy

    return mf
```

### Information-Theoretic Meta-Features

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def compute_information_meta_features(X: np.ndarray, y: np.ndarray,
                                       sample_features: int = 50) -> dict:
    """
    MI-based meta-features describing feature-target relevance structure.

    Uses a random sample of features for efficiency on high-dimensional data.
    """
    n, p = X.shape
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(p, size=min(sample_features, p), replace=False)
    X_sample = X[:, sample_idx]

    is_classification = len(np.unique(y)) <= 20
    if is_classification:
        mi_scores = mutual_info_classif(X_sample, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X_sample, y, random_state=42)

    mf = {}
    mf['mean_mi'] = float(np.mean(mi_scores))
    mf['max_mi'] = float(np.max(mi_scores))
    mf['min_mi'] = float(np.min(mi_scores))
    mf['mi_skew'] = float(stats.skew(mi_scores))
    # Fraction of features with near-zero MI (likely noise)
    mf['frac_zero_mi'] = float(np.mean(mi_scores < 0.01))
    # "Spikiness": how concentrated the MI is in a few features
    mf['mi_gini'] = float(gini_coefficient(mi_scores))

    return mf


def gini_coefficient(arr: np.ndarray) -> float:
    """Gini coefficient of an array (concentration measure, range [0, 1])."""
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    cumsum = np.cumsum(arr)
    return float((2 * np.sum(cumsum) / (n * cumsum[-1])) - (n + 1) / n)
```

### Landmarking Meta-Features

Landmarking uses performance of simple algorithms as fast proxies for dataset difficulty:

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

def compute_landmarking_meta_features(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Landmarking: performance of fast simple algorithms on the raw dataset.

    These serve as dataset-difficulty proxies: a dataset where a depth-1 tree
    achieves 90% accuracy has a very different character from one where it
    achieves 51%.
    """
    n, p = X.shape
    X_scaled = StandardScaler().fit_transform(X)

    # Subsample for speed
    max_n = min(n, 500)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=max_n, replace=False)
    X_sub, y_sub = X_scaled[idx], y[idx]

    mf = {}
    scoring = 'balanced_accuracy' if len(np.unique(y)) > 1 else 'r2'

    # Depth-1 decision tree (best single-feature split)
    dt1 = DecisionTreeClassifier(max_depth=1, random_state=42)
    mf['dt1_acc'] = float(cross_val_score(dt1, X_sub, y_sub,
                                           cv=3, scoring=scoring).mean())

    # 1-NN (local structure proxy)
    knn1 = KNeighborsClassifier(n_neighbors=1)
    mf['knn1_acc'] = float(cross_val_score(knn1, X_sub, y_sub,
                                            cv=3, scoring=scoring).mean())

    # Majority class baseline
    dummy = DummyClassifier(strategy='most_frequent')
    mf['baseline_acc'] = float(cross_val_score(dummy, X_sub, y_sub,
                                                cv=3, scoring=scoring).mean())

    # Relative performance (how much better than baseline)
    mf['dt1_lift'] = mf['dt1_acc'] - mf['baseline_acc']
    mf['knn1_lift'] = mf['knn1_acc'] - mf['baseline_acc']

    return mf


def compute_all_meta_features(X: np.ndarray, y: np.ndarray) -> dict:
    """Compute all meta-features for a dataset."""
    mf = {}
    mf.update(compute_statistical_meta_features(X, y))
    mf.update(compute_correlation_meta_features(X))
    mf.update(compute_information_meta_features(X, y))
    mf.update(compute_landmarking_meta_features(X, y))
    return mf
```

---

## Learning Which Selector Works Best

### Constructing the Meta-Dataset

To train a meta-learner, collect:
1. **Meta-features** $m_i$ for each dataset $\mathcal{D}_i$ (using the functions above)
2. **Target label** $t_i$ = which selector produced the best downstream performance on $\mathcal{D}_i$

```python
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
import openml

SELECTORS = ['mi_filter', 'lasso', 'boruta', 'shap', 'ga_wrapper']

def build_meta_dataset(datasets: list[tuple],
                       selector_portfolio: list[str],
                       eval_model,
                       n_features_final: int = 10) -> pd.DataFrame:
    """
    Build meta-dataset: (meta-features, best-selector) pairs.

    Parameters
    ----------
    datasets : list of (X, y, name) tuples
    selector_portfolio : list of selector names to evaluate
    eval_model : sklearn estimator for downstream performance evaluation
    n_features_final : target number of features to select

    Returns
    -------
    DataFrame with meta-features as columns and 'best_selector' as target.
    """
    from sklearn.model_selection import cross_val_score

    rows = []
    for X, y, name in datasets:
        # Compute meta-features
        meta = compute_all_meta_features(X, y)
        meta['dataset_name'] = name

        # Evaluate each selector
        selector_scores = {}
        for sel_name in selector_portfolio:
            selected_features = run_selector(sel_name, X, y, n_features_final)
            X_selected = X[:, selected_features]
            score = cross_val_score(eval_model, X_selected, y,
                                    cv=5, scoring='balanced_accuracy').mean()
            selector_scores[sel_name] = score

        # Record best selector
        meta['best_selector'] = max(selector_scores, key=selector_scores.get)
        meta['best_score'] = max(selector_scores.values())
        # Record relative performance of each selector
        for sel_name, score in selector_scores.items():
            meta[f'score_{sel_name}'] = score

        rows.append(meta)

    return pd.DataFrame(rows)
```

### Training the Meta-Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder

def train_meta_model(meta_df: pd.DataFrame,
                     meta_feature_cols: list[str]) -> tuple:
    """
    Train a classifier to predict the best feature selector from meta-features.

    Uses Leave-One-Dataset-Out cross-validation for unbiased evaluation.

    Parameters
    ----------
    meta_df : DataFrame with meta-features and 'best_selector' target.
    meta_feature_cols : columns to use as input features.

    Returns
    -------
    (fitted_model, label_encoder, cv_scores)
    """
    X_meta = meta_df[meta_feature_cols].values
    y_meta = meta_df['best_selector'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_meta)

    # Leave-One-Dataset-Out CV
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    loo_scores = cross_val_score(meta_model, X_meta, y_encoded,
                                 cv=LeaveOneOut(), scoring='accuracy')

    # Fit on all data for production use
    meta_model.fit(X_meta, y_encoded)

    print(f"Leave-One-Out accuracy: {loo_scores.mean():.3f} ± {loo_scores.std():.3f}")
    print(f"Classes: {le.classes_}")

    return meta_model, le, loo_scores
```

---

## AutoML Feature Selection Components

### FLAML Integration

FLAML (Fast and Lightweight AutoML) includes a feature selection component that wraps learner-specific selection:

```python
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

def flaml_automl_with_selection(X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  time_budget: int = 60) -> dict:
    """
    Run FLAML AutoML with automatic feature selection.

    FLAML internally searches over learner + hyperparameter combinations.
    The selected model implicitly defines which features are most important
    (via SHAP values or feature importances of the best model).

    Parameters
    ----------
    time_budget : seconds for AutoML search

    Returns
    -------
    dict with best_model, feature_importances, test_score
    """
    if not FLAML_AVAILABLE:
        return {'error': 'flaml not installed. pip install flaml'}

    automl = AutoML()
    automl.fit(X_train, y_train, task='classification',
               time_budget=time_budget, verbose=0)

    # Extract feature importances from best model
    best_model = automl.model.estimator
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.ones(X_train.shape[1]) / X_train.shape[1]

    test_score = (best_model.predict(X_test) == y_test).mean()

    return {
        'best_estimator': automl.best_estimator,
        'feature_importances': importances,
        'test_score': test_score,
        'best_config': automl.best_config
    }
```

### Bayesian Model Averaging over Feature Subsets

Rather than selecting a single feature subset, Bayesian Model Averaging (BMA) averages predictions over multiple subsets weighted by their posterior probability:

$$P(y^* | x^*, \mathcal{D}) = \sum_{S \subseteq \mathcal{F}} P(y^* | x^*_S, \mathcal{D}) \cdot P(S | \mathcal{D})$$

The posterior weight $P(S | \mathcal{D})$ is proportional to the marginal likelihood of the model using feature subset $S$:

$$P(S | \mathcal{D}) \propto P(\mathcal{D} | S) \cdot P(S)$$

In practice, we approximate BMA using a finite ensemble of high-performing subsets from the evolutionary search:

```python
def bayesian_model_averaging_selector(X: np.ndarray, y: np.ndarray,
                                        eval_model,
                                        n_subsets: int = 20,
                                        top_k: int = 15) -> np.ndarray:
    """
    Approximate BMA feature selection: generate diverse subsets, weight by
    cross-validated log-likelihood, average predictions.

    Parameters
    ----------
    n_subsets : number of diverse subsets to generate and weight
    top_k : features per subset

    Returns
    -------
    Feature importance array (higher = more frequently in high-weight subsets).
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    n, p = X.shape
    rng = np.random.default_rng(42)

    subset_scores = []
    subset_masks = []

    for _ in range(n_subsets):
        # Random subset of size top_k
        subset = rng.choice(p, size=top_k, replace=False)
        mask = np.zeros(p, dtype=bool)
        mask[subset] = True

        X_sub = X[:, mask]
        score = cross_val_score(eval_model, X_sub, y,
                                cv=3, scoring='balanced_accuracy').mean()
        subset_scores.append(score)
        subset_masks.append(mask)

    # Convert scores to BMA weights (softmax over scores)
    scores_arr = np.array(subset_scores)
    weights = np.exp(scores_arr - scores_arr.max())
    weights /= weights.sum()

    # Aggregate: feature importance = weighted inclusion probability
    importance = np.zeros(p)
    for mask, w in zip(subset_masks, weights):
        importance[mask] += w

    return importance
```

---

## Building a Meta-Learning Recommender

### The Full Pipeline

```python
class FeatureSelectionRecommender:
    """
    Meta-learning recommender: given a new dataset, predict the best
    feature selection method.

    Training: (meta-features of dataset) → (best selector for that dataset)
    Inference: compute_meta_features(new_dataset) → recommend selector
    """

    # Core meta-feature columns (subset that are most predictive)
    META_FEATURE_COLS = [
        'log_n', 'log_p', 'log_ratio_n_p',
        'n_classes', 'class_balance', 'class_entropy',
        'mean_abs_correlation', 'max_abs_correlation', 'frac_highly_correlated',
        'effective_rank_ratio',
        'mean_mi', 'max_mi', 'frac_zero_mi', 'mi_gini',
        'dt1_lift', 'knn1_lift',
    ]

    def __init__(self):
        self.meta_model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, meta_df: pd.DataFrame) -> 'FeatureSelectionRecommender':
        """
        Train the meta-model on a dataset of (meta-features, best-selector) pairs.

        Parameters
        ----------
        meta_df : DataFrame with META_FEATURE_COLS columns and 'best_selector' column.
        """
        available_cols = [c for c in self.META_FEATURE_COLS if c in meta_df.columns]
        X_meta = meta_df[available_cols].fillna(0).values
        y_meta = self.label_encoder.fit_transform(meta_df['best_selector'].values)

        self.meta_model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.meta_model.fit(X_meta, y_meta)
        self.fitted_cols = available_cols
        self.is_fitted = True
        return self

    def recommend(self, X: np.ndarray, y: np.ndarray,
                   top_n: int = 3) -> list[tuple[str, float]]:
        """
        Recommend top-n feature selectors for a new dataset.

        Parameters
        ----------
        X, y : New dataset.
        top_n : Number of recommendations to return.

        Returns
        -------
        List of (selector_name, probability) tuples, sorted by probability.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        meta_features = compute_all_meta_features(X, y)
        meta_vector = np.array([meta_features.get(col, 0.0)
                                 for col in self.fitted_cols]).reshape(1, -1)

        proba = self.meta_model.predict_proba(meta_vector)[0]
        selector_names = self.label_encoder.classes_

        recommendations = sorted(
            zip(selector_names, proba),
            key=lambda x: x[1],
            reverse=True
        )
        return recommendations[:top_n]

    def explain(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Explain the recommendation by showing which meta-features drove it.

        Returns a DataFrame of meta-features with their values for context.
        """
        meta_features = compute_all_meta_features(X, y)
        recommendations = self.recommend(X, y, top_n=len(self.label_encoder.classes_))

        explanation = pd.DataFrame({
            'selector': [r[0] for r in recommendations],
            'confidence': [r[1] for r in recommendations]
        })

        # Append key meta-features for interpretation
        key_features = ['log_ratio_n_p', 'mean_abs_correlation',
                         'frac_highly_correlated', 'frac_zero_mi', 'mi_gini']
        for feat in key_features:
            if feat in meta_features:
                explanation[feat] = meta_features[feat]

        return explanation
```

---

## Evaluation of the Meta-Recommender

### Leave-One-Dataset-Out Validation

The gold standard for meta-learning evaluation is Leave-One-Dataset-Out (LODO) cross-validation:
- Train the meta-model on all but one dataset
- Test prediction on the held-out dataset
- Repeat for each dataset

LODO measures whether the meta-model generalises to truly unseen datasets, not just unseen observations from the same distribution.

```python
def evaluate_recommender_lodo(meta_df: pd.DataFrame,
                               dataset_col: str = 'dataset_name') -> dict:
    """
    Leave-One-Dataset-Out evaluation of the meta-recommender.

    Returns
    -------
    dict with mean_accuracy, top3_accuracy, avg_rank_of_best.
    """
    datasets = meta_df[dataset_col].unique()
    correct_top1 = []
    correct_top3 = []
    ranks_of_best = []

    for held_out in datasets:
        train_df = meta_df[meta_df[dataset_col] != held_out]
        test_df  = meta_df[meta_df[dataset_col] == held_out]

        if len(train_df) < 5:  # need enough training examples
            continue

        recommender = FeatureSelectionRecommender()
        recommender.fit(train_df)

        for _, row in test_df.iterrows():
            true_best = row['best_selector']
            X_test = np.array([row.get(col, 0.0)
                                for col in recommender.fitted_cols]).reshape(1, -1)
            proba = recommender.meta_model.predict_proba(X_test)[0]
            ranked = recommender.label_encoder.classes_[np.argsort(-proba)]

            correct_top1.append(ranked[0] == true_best)
            correct_top3.append(true_best in ranked[:3])
            rank = np.where(ranked == true_best)[0]
            ranks_of_best.append(int(rank[0]) + 1 if len(rank) > 0 else len(ranked))

    return {
        'top1_accuracy': np.mean(correct_top1),
        'top3_accuracy': np.mean(correct_top3),
        'avg_rank_of_best': np.mean(ranks_of_best),
        'n_evaluations': len(correct_top1)
    }
```

### What Good Performance Looks Like

For a 5-class prediction problem (5 selectors), random baseline accuracy is 20%. A well-trained meta-recommender on 30+ diverse datasets achieves:
- Top-1 accuracy: 40–65%
- Top-3 accuracy: 75–90%
- Average rank of best selector: 1.8–2.5

The top-3 accuracy is the practical metric: if your meta-recommender recommends 3 selectors and the best one is in that list 80% of the time, you've reduced the exploration space by 40%.

---

## Common Pitfalls

- **Too few training datasets:** Meta-learning requires at least 20–30 diverse datasets to generalise. With fewer, the meta-model overfits to the specific datasets in the training set.
- **Homogeneous meta-training data:** If all training datasets are from the same domain (e.g., all are genomics datasets), the meta-recommender will be biased toward selectors that work well in that domain and fail on tabular or time-series data.
- **Expensive landmarking:** Landmarking meta-features require model fitting. On large datasets, use subsampling (max 500 samples) to keep meta-feature computation under 10 seconds.
- **Dataset shift in meta-features:** If your production datasets have very different characteristics from your meta-training datasets (different n, p, domain), the recommender's accuracy degrades. Maintain a growing library of meta-training datasets.
- **Treating the recommendation as final:** The meta-recommender suggests a starting point. Always validate the recommendation with a quick cross-validation on the actual task.

---

## Connections

- **Builds on:** Guide 01 (ensemble selection), Guide 02 (hybrid methods), all prior modules
- **Leads to:** Module 11 (production feature selection, model registries)
- **Related to:** AutoML (Auto-sklearn, FLAML); Transfer Learning; Hyperparameter Optimisation (Optuna, Hyperopt); No-Free-Lunch theorem

---

## Further Reading

- Vanschoren, J. (2019). **Meta-learning: A survey.** *arXiv:1810.03548*. Comprehensive overview.
- Feurer, M. et al. (2015). **Efficient and robust automated machine learning.** *NeurIPS 2015*. Auto-sklearn with meta-learning warm-start.
- Bilalli, B., Abelló, A., & Aluja-Banet, T. (2017). **On the predictive power of meta-features in OpenML.** *Computer Science and Information Systems*, 14(3). Empirical study of meta-feature predictiveness.
- Thornton, C. et al. (2013). **Auto-WEKA: Combined selection and hyperparameter optimisation of classification algorithms.** *KDD 2013*. The original CASH formulation.
