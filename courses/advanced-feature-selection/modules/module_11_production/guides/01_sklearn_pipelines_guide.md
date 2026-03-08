# Feature Selection in sklearn Pipelines

## In Brief

sklearn `Pipeline` objects chain preprocessing, feature selection, and modelling into a single estimator. Wrapping any selection method as a custom transformer gives it the `fit`/`transform` interface that `Pipeline` requires, enabling correct cross-validation, serialisation, and deployment with zero data leakage.

## Key Insight

The cardinal rule: **feature selection must be inside the pipeline, not before it**. Selection fitted on the full dataset and then cross-validated leaks information from validation folds into training folds, producing optimistically biased scores. A `Pipeline` enforces this boundary automatically.

---

## The fit/transform Interface

Every sklearn-compatible transformer must implement three methods:

```python
class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Learn from training data; store state in self.*
        return self           # always return self

    def transform(self, X):
        # Apply learned state; must not use y
        return X_transformed  # same n_samples, potentially fewer columns

    def get_feature_names_out(self, input_features=None):
        # Required for Pipeline.set_output(transform='pandas')
        return selected_feature_names
```

`TransformerMixin` provides `fit_transform` for free. `BaseEstimator` provides `get_params`/`set_params` for free, which enables `GridSearchCV` to tune transformer hyperparameters.

---

## Building a Reusable Selector Transformer

The pattern below wraps **any** scikit-learn selector (or custom selection function) into a reusable transformer.

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Wrap any sklearn selector into a Pipeline-compatible transformer.

    Parameters
    ----------
    selector : sklearn selector instance
        Must expose fit(X, y) and either support_ or get_support().
    """

    def __init__(self, selector):
        self.selector = selector

    def fit(self, X, y):
        """Learn which features to keep."""
        # Store input feature names when X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array(
                [f'x{i}' for i in range(X.shape[1])]
            )
        self.selector.fit(X, y)
        # Store the boolean mask for fast transform
        self.support_ = self.selector.get_support()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Apply the fitted selection mask."""
        check_is_fitted(self, 'support_')
        if hasattr(X, 'iloc'):
            return X.iloc[:, self.support_]
        return X[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        """Return names of selected features."""
        check_is_fitted(self, 'support_')
        return self.feature_names_in_[self.support_]
```

### Why This Matters

- `check_is_fitted` raises `NotFittedError` with a helpful message if `transform` is called before `fit` — essential for debugging in production.
- Storing `feature_names_in_` mirrors the sklearn 1.0+ convention so downstream steps can propagate feature names.
- Accepting both `DataFrame` and `ndarray` makes the transformer robust regardless of how the pipeline receives its data.

---

## Pipeline Composition: Preprocessing → Selection → Model

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Build the selector: select features whose RF importance >= median
rf_selector = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median',
)

pipeline = Pipeline([
    ('scaler',    StandardScaler()),
    ('selector',  SelectorTransformer(rf_selector)),
    ('model',     GradientBoostingClassifier(n_estimators=200, random_state=42)),
])
```

The pipeline is itself an estimator: `pipeline.fit(X_train, y_train)` calls `fit_transform` on each step in sequence, forwarding the output of step $n$ as the input to step $n+1$. `pipeline.predict(X_test)` calls `transform` on all steps except the last, then calls `predict` on the final estimator.

---

## ColumnTransformer Integration

Real datasets mix numeric and categorical features. `ColumnTransformer` applies different preprocessing to each column type before a unified selection step.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif

numeric_cols = ['price', 'volume', 'volatility', 'rsi', 'macd']
categorical_cols = ['sector', 'exchange', 'market_cap_bucket']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(),        numeric_cols),
    ('cat', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False),        categorical_cols),
])

full_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('selector',   SelectKBest(score_func=f_classif, k=10)),
    ('model',      GradientBoostingClassifier(random_state=42)),
])
```

After `preprocessor.fit_transform`, all columns are numeric. `SelectKBest` then operates on the flattened numeric matrix.

**Gotcha**: `get_feature_names_out()` on `ColumnTransformer` returns names like `num__price` and `cat__sector_Technology`. These names propagate through if you set `pipeline.set_output(transform='pandas')`.

---

## Cross-Validation with Selection Inside the Pipeline

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(
    full_pipeline,
    X, y,
    cv=cv,
    scoring=['roc_auc', 'f1'],
    return_train_score=True,
    n_jobs=-1,
)

print(f"Val AUC:  {results['test_roc_auc'].mean():.4f} "
      f"± {results['test_roc_auc'].std():.4f}")
print(f"Train AUC: {results['train_roc_auc'].mean():.4f} "
      f"± {results['train_roc_auc'].std():.4f}")
```

Each fold:
1. Calls `pipeline.fit(X_train_fold, y_train_fold)` — scaler, selector, and model all fitted on training data only.
2. Calls `pipeline.predict/score(X_val_fold)` — all transforms applied with training-fold parameters.

This is correct. The validation fold never influences any fitted parameter.

### Common Leakage Anti-Pattern

```python
# WRONG — selector sees all of X before CV split
selector.fit(X, y)
X_selected = selector.transform(X)
cv_score = cross_val_score(model, X_selected, y, cv=5)
```

The selector's knowledge of the validation fold is embedded in `support_`. Any validation metric computed this way is optimistically biased by the selection step.

---

## Hyperparameter Tuning Across the Entire Pipeline

`GridSearchCV` uses `__` to address nested parameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'selector__selector__threshold':  ['mean', 'median', '0.5*mean'],
    'selector__selector__estimator__n_estimators': [50, 100],
    'model__n_estimators': [100, 200],
    'model__max_depth':    [3, 5],
}

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    refit=True,
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")
```

Every candidate in the grid is independently cross-validated with selection re-run from scratch on each training fold. No leakage; no shortcuts.

---

## Serialisation: Pickling Pipelines with Selection State

```python
import joblib
import pathlib

# Serialise the entire fitted pipeline
model_path = pathlib.Path('artefacts/feature_pipeline_v1.pkl')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(grid_search.best_estimator_, model_path)

# Reload and verify selection state is intact
loaded_pipeline = joblib.load(model_path)
selector_step = loaded_pipeline.named_steps['selector']

print("Selected features after reload:")
for name in selector_step.get_feature_names_out():
    print(f"  {name}")

# Confirm predictions are identical
y_pred_original = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
y_pred_loaded   = loaded_pipeline.predict_proba(X_test)[:, 1]
assert np.allclose(y_pred_original, y_pred_loaded), "Mismatch after reload"
print("Serialisation verified — predictions identical.")
```

`joblib` is preferred over `pickle` for large numpy arrays because it uses memory-mapped files, making load/save significantly faster.

---

## Stability Selection as a Custom Transformer

Stability selection runs the base selector on many bootstrap samples and retains only features that are consistently selected. Wrapping it as a transformer makes it a first-class pipeline citizen.

```python
from sklearn.utils import resample


class StabilitySelector(BaseEstimator, TransformerMixin):
    """
    Stability selection: bootstrap the base selector and keep
    features selected in at least `threshold` fraction of runs.

    Parameters
    ----------
    base_selector : fitted-selector-like
        Exposes fit(X, y) and get_support().
    n_bootstrap : int
        Number of bootstrap iterations.
    threshold : float
        Selection frequency threshold in (0.5, 1.0].
    subsample : float
        Fraction of training data per bootstrap.
    random_state : int | None
    """

    def __init__(
        self,
        base_selector,
        n_bootstrap: int = 100,
        threshold: float = 0.8,
        subsample: float = 0.5,
        random_state: int | None = 42,
    ):
        self.base_selector = base_selector
        self.n_bootstrap    = n_bootstrap
        self.threshold      = threshold
        self.subsample      = subsample
        self.random_state   = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        selection_counts = np.zeros(n_features, dtype=int)

        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = np.asarray(y)

        for _ in range(self.n_bootstrap):
            seed = int(rng.integers(0, 2**31))
            X_boot, y_boot = resample(
                X_arr, y_arr,
                n_samples=int(len(y_arr) * self.subsample),
                random_state=seed,
            )
            import sklearn.base as _base
            sel = _base.clone(self.base_selector)
            sel.fit(X_boot, y_boot)
            selection_counts += sel.get_support().astype(int)

        self.selection_scores_ = selection_counts / self.n_bootstrap
        self.support_ = self.selection_scores_ >= self.threshold

        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array(
                [f'x{i}' for i in range(n_features)]
            )
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        check_is_fitted(self, 'support_')
        if hasattr(X, 'iloc'):
            return X.iloc[:, self.support_]
        return X[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'support_')
        return self.feature_names_in_[self.support_]

    def get_selection_scores(self) -> pd.Series:
        """Return per-feature selection frequency as a Series."""
        check_is_fitted(self, 'selection_scores_')
        return pd.Series(
            self.selection_scores_,
            index=self.feature_names_in_,
        ).sort_values(ascending=False)
```

---

## Pipeline Gotchas

### Feature Name Propagation

After sklearn 1.0, pipelines propagate feature names only if every step implements `get_feature_names_out`. Enable DataFrame output globally:

```python
from sklearn import set_config
set_config(transform_output='pandas')
```

Now `pipeline.transform(X)` returns a DataFrame with correct column names at every step. This makes debugging far easier.

### Inverse Transform

`Pipeline.inverse_transform` only works if every step implements `inverse_transform`. `StandardScaler` does; most selectors do not. Build a manual reverse lookup when needed:

```python
def decode_predictions(pipeline, X_raw, y_pred):
    """Map predictions back to original feature names."""
    selector = pipeline.named_steps['selector']
    selected = selector.get_feature_names_out()
    return pd.DataFrame({'feature': selected, 'prediction': y_pred[:len(selected)]})
```

### set_output with ColumnTransformer

`ColumnTransformer` with `set_output(transform='pandas')` requires all sub-transformers to implement `get_feature_names_out`. `OneHotEncoder` does; `StandardScaler` does since sklearn 1.0. Custom transformers must implement it explicitly.

### Memory Step Caching

For expensive selectors (e.g., stability selection), cache fitted transformers:

```python
from sklearn.pipeline import Pipeline
from joblib import Memory

memory = Memory(location='cache/', verbose=0)

cached_pipeline = Pipeline(
    [('scaler', StandardScaler()), ('selector', StabilitySelector(...)), ('model', ...)],
    memory=memory,
)
```

With caching, re-fitting the pipeline with the same data and parameters skips re-computation of cached steps.

---

## Common Pitfalls

- **Fitting the pipeline before train/test split**: Always split first, then call `pipeline.fit(X_train, y_train)`.
- **Passing `X_test` to `pipeline.fit`**: Any data passed to `fit` may influence the selector. Only training data belongs there.
- **Ignoring `get_feature_names_out`**: Custom transformers that omit this method break `set_output`, inspection, and some AutoML frameworks.
- **Pickling lambda functions**: Pipeline steps that use lambdas (e.g., as part of `FunctionTransformer`) cannot be pickled. Use named functions instead.

---

## Connections

- **Builds on:** Module 10 ensemble hybrid selection, Module 03 wrapper methods
- **Leads to:** Module 11 drift monitoring (pipelines wrap the entire re-selection cycle)
- **Related to:** Module 09 causal selection (causal selector drops in as a custom transformer)

---

## Further Reading

- scikit-learn Pipeline documentation: https://scikit-learn.org/stable/modules/compose.html
- Meinshausen & Bühlmann (2010). Stability Selection. *JRSS-B*, 72(4), 417–473.
- Buitinck et al. (2013). API Design for Machine Learning Software. *ECML PKDD Workshop*.
