# MLOps Integration for Feature Selection

## In Brief

Production feature selection is not a one-time computation — it is an ongoing process that must be tracked, versioned, reproduced, and audited. This guide covers the MLOps infrastructure that supports production feature selection: MLflow experiment tracking, feature store integration with Feast, selection versioning, and three real-world case studies that show how budget, regulatory, and latency constraints shape selection decisions.

## Key Insight

Every feature selection run is an experiment: a combination of method, hyperparameters, dataset version, and random seed that produces a specific feature subset. Treating it as an informal script run means you cannot reproduce last quarter's results, cannot explain to a regulator which features were selected and why, and cannot roll back when a new selection degrades model performance. MLflow and structured versioning eliminate all of these problems.

---

## MLflow Integration for Feature Selection

### Experiment Setup

```python
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
import json


EXPERIMENT_NAME = "feature-selection-production"

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # local; use server URI in production
mlflow.set_experiment(EXPERIMENT_NAME)
```

### Logging a Selection Run

```python
def run_selection_with_mlflow(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    selector,
    selector_name: str,
    selector_params: dict,
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    dataset_version: str = "v1",
    tags: dict | None = None,
) -> mlflow.ActiveRun:
    """
    Run feature selection and log everything to MLflow.

    Parameters
    ----------
    selector : fitted-selector-like
        Any object with fit(X, y) and get_support() / get_feature_names_out().
    selector_params : dict
        Hyperparameters of the selector (for logging).
    dataset_version : str
        Identifier for the training dataset (e.g. git hash, date range).
    tags : dict | None
        Additional MLflow tags (e.g. {'environment': 'staging'}).
    """
    with mlflow.start_run(tags=tags or {}):
        # --- Log parameters ---
        mlflow.log_param("selector_name",    selector_name)
        mlflow.log_param("dataset_version",  dataset_version)
        mlflow.log_param("n_features_total", X_train.shape[1])
        for k, v in selector_params.items():
            mlflow.log_param(f"selector.{k}", v)

        # --- Fit selector ---
        selector.fit(X_train, y_train)
        selected_mask  = selector.get_support()
        selected_names = X_train.columns[selected_mask].tolist()
        n_selected     = len(selected_names)

        # --- Log selection metrics ---
        mlflow.log_metric("n_features_selected", n_selected)
        mlflow.log_metric("selection_ratio",
                          n_selected / X_train.shape[1])

        # --- Train model on selected features ---
        X_tr_sel = X_train.iloc[:, selected_mask]
        X_va_sel = X_val.iloc[:,   selected_mask]
        model.fit(X_tr_sel, y_train)

        from sklearn.metrics import roc_auc_score, average_precision_score
        y_prob = model.predict_proba(X_va_sel)[:, 1]
        auc    = roc_auc_score(y_val, y_prob)
        ap     = average_precision_score(y_val, y_prob)

        mlflow.log_metric("val_roc_auc",             auc)
        mlflow.log_metric("val_average_precision",   ap)

        # --- Log artefacts ---
        # Feature list as JSON
        artefact_path = Path("mlflow_tmp")
        artefact_path.mkdir(exist_ok=True)

        features_file = artefact_path / "selected_features.json"
        features_file.write_text(json.dumps({
            "selected_features": selected_names,
            "n_selected":        n_selected,
            "selector":          selector_name,
            "dataset_version":   dataset_version,
        }, indent=2))
        mlflow.log_artifact(str(features_file))

        # Feature importance scores (if available)
        if hasattr(selector, 'scores_'):
            scores_df = pd.DataFrame({
                'feature': X_train.columns,
                'score':   selector.scores_,
                'selected': selected_mask,
            }).sort_values('score', ascending=False)
            scores_file = artefact_path / "feature_scores.csv"
            scores_df.to_csv(scores_file, index=False)
            mlflow.log_artifact(str(scores_file))

        # Log the full sklearn pipeline as a model
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  model),
        ])
        pipeline.fit(X_tr_sel, y_train)
        mlflow.sklearn.log_model(pipeline, "pipeline")

        print(f"Run complete: {n_selected} features selected, "
              f"Val AUC={auc:.4f}")
        return mlflow.active_run()
```

### Comparing Selection Runs

```python
def compare_selection_runs(experiment_name: str) -> pd.DataFrame:
    """
    Fetch all runs from an experiment and compare selection outcomes.

    Returns a DataFrame sorted by val_roc_auc descending.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_roc_auc DESC"],
    )

    records = []
    for run in runs:
        records.append({
            'run_id':           run.info.run_id[:8],
            'selector':         run.data.params.get('selector_name', ''),
            'dataset_version':  run.data.params.get('dataset_version', ''),
            'n_selected':       int(run.data.metrics.get('n_features_selected', 0)),
            'val_roc_auc':      run.data.metrics.get('val_roc_auc', np.nan),
            'val_ap':           run.data.metrics.get('val_average_precision', np.nan),
            'start_time':       pd.Timestamp(run.info.start_time, unit='ms'),
        })

    return pd.DataFrame(records)
```

---

## Feature Stores: Feast Integration

Feature stores centralise feature computation and serving, ensuring that training and serving use identical feature transformations. After selection, you register only the chosen features in the feature store.

### Why Feature Stores Matter for Selection

Without a feature store, the features selected during training may be computed differently during serving (different preprocessing order, different scaler state, different aggregation windows). This "training-serving skew" is one of the most common causes of degraded production performance.

```python
# feast_feature_store_config.py
"""
Example: Register selected commodity features in Feast.
"""
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Data source
commodity_source = FileSource(
    path="data/commodity_features.parquet",
    event_timestamp_column="event_timestamp",
)

# Entity (the thing we're predicting about)
instrument = Entity(
    name="instrument_id",
    value_type=ValueType.STRING,
    description="Commodity instrument identifier",
)


def register_selected_features(
    selected_feature_names: list[str],
    feature_store,
    view_name: str = "selected_commodity_features",
    ttl_days: int = 30,
) -> None:
    """
    Register selected features as a Feast FeatureView.

    Parameters
    ----------
    selected_feature_names : list[str]
        Names of features chosen by the selection algorithm.
    feature_store : feast.FeatureStore
        Initialised Feast store.
    view_name : str
        Name for the new feature view.
    ttl_days : int
        Time-to-live for feature values in the online store.
    """
    features = [
        Feature(name=name, dtype=ValueType.DOUBLE)
        for name in selected_feature_names
    ]

    selected_view = FeatureView(
        name=view_name,
        entities=["instrument_id"],
        ttl=timedelta(days=ttl_days),
        features=features,
        source=commodity_source,
        tags={
            "selection_method": "stability_selection",
            "n_features": str(len(features)),
        },
    )

    feature_store.apply([instrument, selected_view])
    print(f"Registered {len(features)} features as FeatureView '{view_name}'")


def retrieve_selected_features_for_training(
    feature_store,
    entity_df: pd.DataFrame,
    view_name: str,
    selected_feature_names: list[str],
) -> pd.DataFrame:
    """
    Retrieve historical feature values for training using point-in-time joins.

    This guarantees no future leakage — Feast joins features to entities
    using the event_timestamp in entity_df.
    """
    feature_refs = [f"{view_name}:{name}" for name in selected_feature_names]
    training_df = feature_store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()
    return training_df
```

### Training-Serving Consistency

The critical pattern: use the **same** feature view for both training and online serving.

```python
# Training time
training_data = retrieve_selected_features_for_training(
    feature_store, entity_df_train, "selected_commodity_features", selected_features
)

# Serving time (real-time, single entity)
online_features = feature_store.get_online_features(
    features=[f"selected_commodity_features:{f}" for f in selected_features],
    entity_rows=[{"instrument_id": "CRUDE_WTI"}],
).to_dict()
```

Both paths use the same Feast transformations, eliminating training-serving skew.

---

## Selection Versioning

Every production feature selection must be versioned so you can reproduce it exactly, compare across time, and roll back if needed.

```python
import hashlib
import datetime
import json
from dataclasses import dataclass, asdict


@dataclass
class FeatureSelectionVersion:
    """
    Immutable record of a feature selection event.

    Fields
    ------
    version_id : str
        SHA-256 hash of (selected_features + dataset_hash + method + params).
    selected_features : list[str]
        Ordered list of selected feature names.
    n_features_total : int
        Total features considered.
    method : str
        Selection method (e.g. 'stability_selection').
    params : dict
        Method hyperparameters.
    dataset_hash : str
        SHA-256 hash of training data (ensures reproducibility).
    random_seed : int
        Random seed used.
    selection_date : str
        ISO-8601 date of selection.
    val_roc_auc : float
        Validation AUC at selection time.
    mlflow_run_id : str | None
        MLflow run ID for full audit trail.
    """
    version_id:        str
    selected_features: list[str]
    n_features_total:  int
    method:            str
    params:            dict
    dataset_hash:      str
    random_seed:       int
    selection_date:    str
    val_roc_auc:       float
    mlflow_run_id:     str | None = None

    @classmethod
    def create(
        cls,
        selected_features: list[str],
        n_features_total: int,
        method: str,
        params: dict,
        dataset_hash: str,
        random_seed: int,
        val_roc_auc: float,
        mlflow_run_id: str | None = None,
    ) -> "FeatureSelectionVersion":
        fingerprint = json.dumps({
            "features": sorted(selected_features),
            "dataset_hash": dataset_hash,
            "method": method,
            "params": params,
            "seed": random_seed,
        }, sort_keys=True)
        version_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]

        return cls(
            version_id=version_id,
            selected_features=sorted(selected_features),
            n_features_total=n_features_total,
            method=method,
            params=params,
            dataset_hash=dataset_hash,
            random_seed=random_seed,
            selection_date=datetime.date.today().isoformat(),
            val_roc_auc=val_roc_auc,
            mlflow_run_id=mlflow_run_id,
        )

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str) -> "FeatureSelectionVersion":
        return cls(**json.loads(Path(path).read_text()))


def hash_dataframe(df: pd.DataFrame) -> str:
    """Deterministic hash of a DataFrame for versioning."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]
```

### Version Registry

```python
class SelectionVersionRegistry:
    """
    File-based registry of all feature selection versions.
    In production, back this with a database or blob store.
    """

    def __init__(self, registry_dir: str = "selection_versions/"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def save(self, version: FeatureSelectionVersion) -> None:
        path = self.registry_dir / f"{version.selection_date}_{version.version_id}.json"
        version.to_json(str(path))
        print(f"Saved version {version.version_id} to {path}")

    def load(self, version_id: str) -> FeatureSelectionVersion:
        matches = list(self.registry_dir.glob(f"*{version_id}*.json"))
        if not matches:
            raise KeyError(f"Version {version_id} not found")
        return FeatureSelectionVersion.from_json(str(matches[0]))

    def list_versions(self) -> pd.DataFrame:
        versions = []
        for f in sorted(self.registry_dir.glob("*.json")):
            v = FeatureSelectionVersion.from_json(str(f))
            versions.append({
                'version_id': v.version_id,
                'date': v.selection_date,
                'method': v.method,
                'n_selected': len(v.selected_features),
                'val_roc_auc': v.val_roc_auc,
            })
        return pd.DataFrame(versions).sort_values('date', ascending=False)

    def get_current(self) -> FeatureSelectionVersion:
        """Return the most recently saved version."""
        df = self.list_versions()
        if df.empty:
            raise RuntimeError("No versions in registry")
        return self.load(df.iloc[0]['version_id'])
```

---

## Reproducibility: Seeding, Logging, and Audit Trails

### The Reproducibility Contract

A feature selection run is reproducible if, given the same inputs (data, method, hyperparameters, seed), it produces exactly the same output (selected features). This contract must be enforced by design, not by hope.

```python
import os
import random


def set_all_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for reproducible feature selection.

    Covers: Python random, NumPy, and sklearn (via np.random).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def assert_reproducible(selector_class, selector_params, X, y, seed=42):
    """
    Verify that two runs with the same seed produce identical selections.
    Raises AssertionError if they differ.
    """
    set_all_seeds(seed)
    sel1 = selector_class(**{**selector_params, 'random_state': seed})
    sel1.fit(X, y)
    features1 = sorted(X.columns[sel1.get_support()].tolist())

    set_all_seeds(seed)
    sel2 = selector_class(**{**selector_params, 'random_state': seed})
    sel2.fit(X, y)
    features2 = sorted(X.columns[sel2.get_support()].tolist())

    assert features1 == features2, (
        f"Reproducibility failure: different features selected with same seed.\n"
        f"Run 1: {features1}\nRun 2: {features2}"
    )
    print("Reproducibility verified: identical feature sets across runs.")
```

### Audit Trail Structure

Every production selection run should emit a structured audit record:

```python
def create_audit_record(
    version: FeatureSelectionVersion,
    triggered_by: str,
    trigger_reason: str,
    approved_by: str | None = None,
) -> dict:
    """
    Create a structured audit record for regulatory or operational review.

    Parameters
    ----------
    triggered_by : str
        'scheduled', 'psi_trigger', 'ks_trigger', 'manual'.
    trigger_reason : str
        Human-readable description of why re-selection was triggered.
    approved_by : str | None
        User ID of approver (required for regulated industries).
    """
    return {
        "audit_timestamp":   datetime.datetime.utcnow().isoformat(),
        "version_id":        version.version_id,
        "selection_date":    version.selection_date,
        "method":            version.method,
        "params":            version.params,
        "n_features_total":  version.n_features_total,
        "n_features_selected": len(version.selected_features),
        "selected_features": version.selected_features,
        "val_roc_auc":       version.val_roc_auc,
        "dataset_hash":      version.dataset_hash,
        "random_seed":       version.random_seed,
        "mlflow_run_id":     version.mlflow_run_id,
        "triggered_by":      triggered_by,
        "trigger_reason":    trigger_reason,
        "approved_by":       approved_by,
    }
```

---

## Computational Budget Allocation

Feature selection is not free. Runtime grows with dataset size and method complexity. The Pareto front of (accuracy, feature count, compute cost) guides budget allocation.

```python
import time


def pareto_selection_budget(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    candidate_selectors: list[dict],
    time_budget_seconds: float = 300.0,
) -> pd.DataFrame:
    """
    Evaluate candidate selectors within a time budget.
    Returns Pareto-optimal candidates by (val_auc, n_features, runtime).

    Parameters
    ----------
    candidate_selectors : list of dicts
        Each dict: {'name': str, 'selector': selector_instance,
                   'model': model_instance}.
    time_budget_seconds : float
        Total wall-clock budget for all evaluations.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    results = []
    elapsed = 0.0

    for config in candidate_selectors:
        if elapsed >= time_budget_seconds:
            print(f"Time budget exhausted. Evaluated {len(results)} methods.")
            break

        t0 = time.perf_counter()
        try:
            pipeline = Pipeline([
                ('scaler',   StandardScaler()),
                ('selector', config['selector']),
                ('model',    config['model']),
            ])
            scores = cross_val_score(pipeline, X_train, y_train,
                                     cv=3, scoring='roc_auc')
            pipeline.fit(X_train, y_train)
            n_sel = pipeline.named_steps['selector'].support_.sum()
            runtime = time.perf_counter() - t0
            elapsed += runtime

            results.append({
                'name':      config['name'],
                'val_auc':   scores.mean(),
                'n_features': int(n_sel),
                'runtime_s': round(runtime, 2),
            })
        except Exception as exc:
            print(f"  {config['name']} failed: {exc}")

    df = pd.DataFrame(results)
    df['is_pareto'] = _compute_pareto_front(df)
    return df.sort_values('val_auc', ascending=False)


def _compute_pareto_front(df: pd.DataFrame) -> list[bool]:
    """
    Identify Pareto-optimal rows: maximise val_auc, minimise n_features and runtime.
    """
    is_pareto = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            # other dominates row if it's at least as good on all objectives
            # and strictly better on at least one
            if (other['val_auc']   >= row['val_auc'] and
                other['n_features'] <= row['n_features'] and
                other['runtime_s']  <= row['runtime_s'] and
                (other['val_auc']   > row['val_auc'] or
                 other['n_features'] < row['n_features'] or
                 other['runtime_s']  < row['runtime_s'])):
                dominated = True
                break
        is_pareto.append(not dominated)
    return is_pareto
```

---

## Case Studies

### Case Study 1: Commodity Price Forecasting — 500+ Features → 15 Selected

**Context:** A quantitative trading desk builds a next-day crude oil price direction model. The feature library has 512 engineered features: technical indicators across multiple timeframes, macro variables, cross-commodity spreads, and sentiment scores.

**Challenge:** With 252 trading days per year and 512 features, the feature-to-sample ratio is 2:1 — severe overfitting territory without selection.

**Approach:**
1. Stability selection (n_bootstrap=500, threshold=0.8) on 3 years of daily data
2. Regime-aware selection: separate selection for bull/bear regimes (defined by 200-day MA)
3. Feature set capped at 15 for interpretability and transaction-cost estimation

**Results:**
- 15 features selected (97% reduction from 512)
- Out-of-sample Sharpe ratio: 0.89 vs. 0.31 with all features
- Model interpretable to portfolio managers and risk committee
- Monthly re-selection via PSI trigger (PSI>0.2 on >15% of features)

**Key lesson:** Regime-aware selection produced different feature subsets for bull and bear regimes. The momentum features that work in trending markets were actively harmful in mean-reverting regimes. A single feature set for all regimes performed worse than both conditional models.

---

### Case Study 2: Credit Scoring — Regulatory Feature Interpretability

**Context:** A consumer lending firm builds a default probability model. Basel III and ECOA regulations require that any feature used in a credit decision must be (a) individually explainable to the applicant, (b) causally related to creditworthiness, and (c) not a proxy for protected characteristics.

**Challenge:** Many high-predictive-power features (e.g., purchase category clusters from transaction data) are difficult to explain and may proxy for ethnicity or zip code.

**Approach:**
1. Expert pre-screening: legal and compliance team whitelist 80 permissible features from a candidate pool of 340
2. Stability selection on whitelisted features (n_bootstrap=200, threshold=0.75)
3. Disparity impact testing: check that selected features do not produce adverse impact (>20% differential in approval rates) across protected groups
4. Final review: model validation team approves final feature set

**Results:**
- 23 features selected from 80 whitelisted (71% reduction)
- AUC: 0.79 (vs. 0.83 with unrestricted features — small cost of interpretability)
- All features explainable in plain English to applicants
- Adverse impact testing passed; model approved by compliance
- Selection versioned with approver signature in audit trail

**Key lesson:** Accuracy is not the only objective in regulated industries. The MLops infrastructure — versioning, audit trails, approver signatures — is not optional; it is legally required.

---

### Case Study 3: Real-Time Fraud Detection — Latency Constraints on Feature Computation

**Context:** A payments processor builds a transaction fraud classifier that must return a score within 50ms of a transaction being submitted. Feature computation time is part of the latency budget.

**Challenge:** The best-performing features include complex rolling window aggregates (e.g., "number of transactions > $500 in the past 2 hours for this card"). These take 20-40ms to compute from raw events. With 50ms total latency, only 2-3 such features can be included.

**Approach:**
1. Feature profiling: measure computation latency for each candidate feature in isolation (P99 latency)
2. Latency-aware selection: add a latency penalty to the fitness function

    $$\text{fitness}(S) = \text{AUC}(S) - \lambda \cdot \sum_{i \in S} \text{latency}_i$$

3. Constraint enforcement: hard constraint $\sum_{i \in S} \text{latency}_i \leq 35\text{ms}$ (leaving 15ms headroom)
4. Pre-computation: the 3 high-latency features that pass selection are pre-computed and cached in Redis with 2-minute TTL

**Results:**
- P99 scoring latency: 31ms (within 50ms SLA)
- 12 features selected: 9 fast features (<1ms each) + 3 pre-computed high-value features
- AUC: 0.87 vs. 0.91 with unconstrained selection (cost of latency constraint: 4% AUC)
- Fraud prevention rate improved 22% over legacy rule-based system

**Key lesson:** Feature selection in production is always multi-objective. Accuracy, feature count, compute cost, latency, interpretability, and regulatory compliance all compete. The Pareto framework from Guide 02 applies directly: accept the accuracy cost of the latency constraint because the latency constraint is non-negotiable.

---

## Common Pitfalls

- **Not versioning random seeds**: Two runs with the same data and method but different seeds may select different features. Always log `random_state` in every experiment.
- **MLflow parameter count limits**: MLflow has a 500-parameter limit per run. Log feature lists as JSON artefacts, not as individual parameters.
- **Feast materialization gaps**: If the feature store has not materialised recent data, online serving returns stale values. Monitor materialisation job health alongside model health.
- **Audit records without timestamps**: An audit trail without UTC timestamps cannot be used in regulatory proceedings. Use `datetime.datetime.utcnow().isoformat()` everywhere.
- **Pareto budget ignoring memory**: Time budget alone is insufficient — some selectors (e.g., stability selection with 500 bootstraps on 1M rows) exceed memory limits. Add a memory budget check using `psutil`.

---

## Connections

- **Builds on:** Guide 01 (pipeline serialisation as MLflow artefact), Guide 02 (drift triggers that initiate re-selection runs)
- **Leads to:** Notebook 03 (hands-on MLflow tracking), production deployment patterns
- **Related to:** Module 10 ensemble selection (the selection method being tracked), Module 07 time series (rolling selection logged across windows)

---

## Further Reading

- Zaharia, M., et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow. *IEEE Data Eng. Bull.*
- Feast documentation: https://docs.feast.dev/
- Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS*.
- European Banking Authority (2022). *Guidelines on the Use of Machine Learning for IRB Models*. (Credit scoring regulatory requirements)
