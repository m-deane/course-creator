# Production LLM Monitoring and Drift Detection

> **Reading time:** ~12 min | **Module:** Module 6: Production | **Prerequisites:** Modules 0-5

<div class="callout-key">

**Key Concept Summary:** Production monitoring tracks LLM performance in real-time to detect when model behavior degrades, prompts become stale, or market conditions invalidate historical calibrations. Drift detection identifies when the statistical properties of inputs or outputs change, triggering model retraining, pro...

</div>

## In Brief

Production monitoring tracks LLM performance in real-time to detect when model behavior degrades, prompts become stale, or market conditions invalidate historical calibrations. Drift detection identifies when the statistical properties of inputs or outputs change, triggering model retraining, prompt updates, or manual intervention before financial losses accumulate.

<div class="callout-insight">

**Insight:** LLMs fail silently—they always return plausible-sounding text even when wrong. Unlike traditional ML where accuracy metrics drop visibly, LLM degradation manifests as subtle shifts: slightly lower conviction, different reasoning patterns, or increased uncertainty. Production monitoring must measure semantic drift (meaning changes), performance drift (accuracy decline), and input drift (market regime changes) across multiple timescales to catch failures early.

</div>
## Intuitive Explanation

## Formal Definition

### Types of Drift

**1. Input Drift (Covariate Shift)**

Distribution of inputs changes:
$$P_{\text{prod}}(X) \neq P_{\text{train}}(X)$$

Example: Backtest trained on normal volatility, production sees VIX > 50

**Detection: KL Divergence**
$$D_{\text{KL}}(P_{\text{train}} || P_{\text{prod}}) = \sum_x P_{\text{train}}(x) \log \frac{P_{\text{train}}(x)}{P_{\text{prod}}(x)}$$

**2. Prediction Drift (Concept Drift)**

Relationship between inputs and outputs changes:
$$P_{\text{prod}}(Y|X) \neq P_{\text{train}}(Y|X)$$

Example: "OPEC cut" historically bullish, now bearish due to recession

**Detection: Population Stability Index (PSI)**
$$\text{PSI} = \sum_{i=1}^{k} (\text{Expected}_i - \text{Actual}_i) \times \ln\left(\frac{\text{Expected}_i}{\text{Actual}_i}\right)$$

Thresholds:
- PSI < 0.1: No significant drift
- 0.1 < PSI < 0.2: Moderate drift, investigate
- PSI > 0.2: Significant drift, retrain/recalibrate

**3. Performance Drift**

Accuracy degrades over time:
$$\text{Accuracy}_{\text{prod}}(t) < \text{Accuracy}_{\text{train}} - \epsilon$$

**Detection: CUSUM (Cumulative Sum)**
$$S_t = \max(0, S_{t-1} + (x_t - \mu - k))$$

Where $k$ is allowance (half the shift size to detect)

**4. Semantic Drift**

LLM outputs change meaning while maintaining syntax:

Example:
- Month 1: "Bullish" → Long position
- Month 6: "Bullish" → Same text but now hedge language appears

**Detection: Embedding Distance**
$$\text{drift} = ||\text{avg}(\text{embed}_{\text{recent}}) - \text{avg}(\text{embed}_{\text{baseline}})||_2$$

### Monitoring Metrics

**Real-Time Metrics:**
1. Response latency (p50, p95, p99)
2. Error rate (API failures, parsing errors)
3. Confidence distribution
4. Signal distribution (% bullish/bearish/neutral)

**Daily Metrics:**
1. Win rate (if outcomes available)
2. Calibration error
3. Signal diversity (entropy)
4. Prompt success rate

**Weekly Metrics:**
1. Sharpe ratio (rolling)
2. Drawdown
3. Correlation with benchmark
4. Feature importance shifts

### Why LLMs Drift

**Scenario 1: Market Regime Change**
- Training: Bull market 2019-2021
- Production: Bear market 2022
- Issue: Bullish signals over-generated because prompt optimized for bull regime
- Detection: Signal distribution shifts from 60% bullish to 45% bullish

**Scenario 2: Data Source Quality**
- Training: High-quality curated news
- Production: Noisy social media feeds
- Issue: Sentiment polarity weakens
- Detection: Confidence scores decline, variance increases

**Scenario 3: Prompt Staleness**
- Training: "OPEC cuts are bullish"
- Production: Market now expects cuts (priced in)
- Issue: Historical relationship breaks
- Detection: Win rate drops from 65% to 50%

### Monitoring Dashboard View

```
┌─────────────────────────────────────────────────────────────┐
│ LLM Trading Signal Monitor                  Status: ⚠ WARN │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ PERFORMANCE METRICS (Last 30 Days)                          │
│   Win Rate:      52.3%  (baseline: 62.1%) ↓ -15.8%         │
│   Sharpe Ratio:  1.2    (baseline: 1.8)   ↓ -33.3%         │
│   Calibration:   ECE = 0.14 (was 0.08)    ⚠ DRIFT          │
│                                                              │
│ DRIFT INDICATORS                                             │
│   Signal PSI:    0.18   (threshold: 0.20) ⚠ WARNING        │
│   Input PSI:     0.09   (threshold: 0.20) ✓ OK             │
│   Semantic Drift: 0.23  (threshold: 0.25) ⚠ WARNING        │
│                                                              │
│ RECENT ALERTS                                                │
│   [2025-01-15] Confidence distribution shifted              │
│   [2025-01-12] API latency spike (p99 > 5s)                │
│   [2025-01-10] Win rate below threshold                     │
└─────────────────────────────────────────────────────────────┘
```

## Code Implementation

### Drift Detection System


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">class.py</span>
</div>

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import warnings

@dataclass
class DriftMetrics:
    """Drift detection metrics."""
    psi: float
    kl_divergence: float
    wasserstein_distance: float
    alert_triggered: bool
    timestamp: datetime = field(default_factory=datetime.now)


class PopulationStabilityIndex:
    """
    Calculate Population Stability Index for drift detection.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.baseline_dist = None
        self.bin_edges = None

    def fit(self, baseline_data: np.ndarray):
        """
        Fit baseline distribution.

        Args:
            baseline_data: Training/baseline data
        """
        # Create bins
        self.bin_edges = np.percentile(
            baseline_data,
            np.linspace(0, 100, self.n_bins + 1)
        )
        self.bin_edges[-1] += 1e-10  # Include right edge

        # Baseline distribution
        baseline_counts, _ = np.histogram(baseline_data, bins=self.bin_edges)
        self.baseline_dist = baseline_counts / len(baseline_data)

        # Avoid zero probabilities
        self.baseline_dist = np.maximum(self.baseline_dist, 1e-10)

    def calculate(self, current_data: np.ndarray) -> float:
        """
        Calculate PSI for current data vs baseline.

        Returns:
            PSI value (>0.2 indicates significant drift)
        """
        if self.baseline_dist is None:
            raise ValueError("Must call fit() first")

        # Current distribution
        current_counts, _ = np.histogram(current_data, bins=self.bin_edges)
        current_dist = current_counts / len(current_data)
        current_dist = np.maximum(current_dist, 1e-10)

        # PSI calculation
        psi = np.sum(
            (current_dist - self.baseline_dist) *
            np.log(current_dist / self.baseline_dist)
        )

        return psi


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) for detecting performance degradation.
    """

    def __init__(
        self,
        baseline_mean: float,
        threshold: float,
        allowance: float = None
    ):
        """
        Initialize CUSUM detector.

        Args:
            baseline_mean: Expected mean performance
            threshold: Alert threshold
            allowance: Half the shift to detect (default: threshold/2)
        """
        self.baseline_mean = baseline_mean
        self.threshold = threshold
        self.allowance = allowance or threshold / 2
        self.cusum_pos = 0
        self.cusum_neg = 0
        self.alerts = []

    def update(self, value: float) -> bool:
        """
        Update CUSUM and check for alerts.

        Args:
            value: New observation (e.g., win rate, accuracy)

        Returns:
            True if alert triggered
        """
        # Positive CUSUM (detecting increase)
        self.cusum_pos = max(
            0,
            self.cusum_pos + (value - self.baseline_mean - self.allowance)
        )

        # Negative CUSUM (detecting decrease)
        self.cusum_neg = max(
            0,
            self.cusum_neg - (value - self.baseline_mean + self.allowance)
        )

        # Check thresholds
        alert = (self.cusum_pos > self.threshold or
                self.cusum_neg > self.threshold)

        if alert:
            self.alerts.append({
                'timestamp': datetime.now(),
                'value': value,
                'cusum_pos': self.cusum_pos,
                'cusum_neg': self.cusum_neg
            })

        return alert


class SemanticDriftDetector:
    """
    Detect semantic drift in LLM outputs using embeddings.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.baseline_embeddings = None
        self.baseline_mean = None

    def fit(self, embeddings: np.ndarray):
        """
        Fit baseline from embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        """
        self.baseline_embeddings = embeddings
        self.baseline_mean = embeddings.mean(axis=0)

    def calculate(self, current_embeddings: np.ndarray) -> float:
        """
        Calculate semantic drift as embedding distance.

        Args:
            current_embeddings: Recent embeddings

        Returns:
            Normalized distance (0 = no drift, 1+ = significant drift)
        """
        if self.baseline_mean is None:
            raise ValueError("Must call fit() first")

        current_mean = current_embeddings.mean(axis=0)

        # Cosine distance
        cosine_sim = np.dot(self.baseline_mean, current_mean) / (
            np.linalg.norm(self.baseline_mean) *
            np.linalg.norm(current_mean)
        )
        cosine_distance = 1 - cosine_sim

        return cosine_distance


class LLMProductionMonitor:
    """
    Comprehensive LLM monitoring system.
    """

    def __init__(
        self,
        baseline_win_rate: float = 0.60,
        baseline_sharpe: float = 1.5,
        psi_threshold: float = 0.20,
        cusum_threshold: float = 5.0,
        semantic_threshold: float = 0.25
    ):
        self.baseline_win_rate = baseline_win_rate
        self.baseline_sharpe = baseline_sharpe
        self.psi_threshold = psi_threshold
        self.semantic_threshold = semantic_threshold

        # Drift detectors
        self.signal_psi = PopulationStabilityIndex(n_bins=10)
        self.confidence_psi = PopulationStabilityIndex(n_bins=10)
        self.win_rate_cusum = CUSUMDetector(
            baseline_mean=baseline_win_rate,
            threshold=cusum_threshold
        )
        self.semantic_detector = SemanticDriftDetector()

        # Metrics storage
        self.metrics_history = []
        self.recent_signals = deque(maxlen=1000)
        self.recent_confidences = deque(maxlen=1000)

        # Alert log
        self.alerts = []

    def fit_baseline(
        self,
        baseline_signals: np.ndarray,
        baseline_confidences: np.ndarray,
        baseline_embeddings: Optional[np.ndarray] = None
    ):
        """
        Fit baseline distributions from training/validation data.

        Args:
            baseline_signals: Signal strengths (e.g., -1 to 1)
            baseline_confidences: Confidence scores (0 to 1)
            baseline_embeddings: Optional embeddings
        """
        self.signal_psi.fit(baseline_signals)
        self.confidence_psi.fit(baseline_confidences)

        if baseline_embeddings is not None:
            self.semantic_detector.fit(baseline_embeddings)

    def log_signal(
        self,
        signal: Dict,
        embedding: Optional[np.ndarray] = None
    ):
        """
        Log new signal for monitoring.

        Args:
            signal: Signal dictionary with direction, strength, confidence
            embedding: Optional embedding of LLM output
        """
        # Convert direction to numeric
        direction_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
        signal_value = direction_map.get(signal.get('direction', 'neutral'), 0)
        signal_value *= signal.get('strength', 0.5)

        confidence = signal.get('confidence', 0.5)

        # Store
        self.recent_signals.append(signal_value)
        self.recent_confidences.append(confidence)

    def log_outcome(self, outcome: bool):
        """
        Log trade outcome for win rate tracking.

        Args:
            outcome: True if winning trade, False if losing
        """
        alert = self.win_rate_cusum.update(float(outcome))

        if alert:
            self.alerts.append({
                'type': 'performance_drift',
                'message': 'Win rate CUSUM alert triggered',
                'timestamp': datetime.now()
            })

    def check_drift(self) -> Dict[str, DriftMetrics]:
        """
        Check all drift metrics.

        Returns:
            Dictionary of drift metrics by type
        """
        results = {}

        # Signal distribution drift
        if len(self.recent_signals) >= 50:
            signal_psi = self.signal_psi.calculate(
                np.array(self.recent_signals)
            )
            results['signal_drift'] = DriftMetrics(
                psi=signal_psi,
                kl_divergence=0,  # Could add KL divergence
                wasserstein_distance=0,
                alert_triggered=signal_psi > self.psi_threshold
            )

        # Confidence distribution drift
        if len(self.recent_confidences) >= 50:
            confidence_psi = self.confidence_psi.calculate(
                np.array(self.recent_confidences)
            )
            results['confidence_drift'] = DriftMetrics(
                psi=confidence_psi,
                kl_divergence=0,
                wasserstein_distance=0,
                alert_triggered=confidence_psi > self.psi_threshold
            )

        return results

    def get_status(self) -> Dict:
        """
        Get current monitoring status.

        Returns:
            Status dictionary with all metrics
        """
        drift_metrics = self.check_drift()

        status = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'recent_signals_count': len(self.recent_signals),
                'recent_confidences_mean': np.mean(self.recent_confidences) if self.recent_confidences else 0,
                'recent_confidences_std': np.std(self.recent_confidences) if self.recent_confidences else 0
            },
            'drift': {
                name: {
                    'psi': metrics.psi,
                    'alert': metrics.alert_triggered
                }
                for name, metrics in drift_metrics.items()
            },
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'health': self._compute_health_score(drift_metrics)
        }

        return status

    def _compute_health_score(self, drift_metrics: Dict[str, DriftMetrics]) -> str:
        """
        Compute overall health score.

        Returns:
            'healthy', 'warning', or 'critical'
        """
        alerts = sum(1 for m in drift_metrics.values() if m.alert_triggered)

        if alerts == 0:
            return 'healthy'
        elif alerts <= 1:
            return 'warning'
        else:
            return 'critical'

    def plot_monitoring_dashboard(self):
        """
        Plot monitoring dashboard.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Signal distribution
        if self.recent_signals:
            axes[0, 0].hist(self.recent_signals, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Recent Signal Distribution')
            axes[0, 0].set_xlabel('Signal Strength')
            axes[0, 0].set_ylabel('Count')

        # Confidence over time
        if self.recent_confidences:
            axes[0, 1].plot(list(self.recent_confidences))
            axes[0, 1].axhline(
                y=np.mean(self.recent_confidences),
                color='r',
                linestyle='--',
                label='Mean'
            )
            axes[0, 1].set_title('Confidence Over Time')
            axes[0, 1].set_xlabel('Signal Index')
            axes[0, 1].set_ylabel('Confidence')
            axes[0, 1].legend()

        # CUSUM chart
        cusum_values = [a['cusum_neg'] for a in self.win_rate_cusum.alerts[-100:]]
        if cusum_values:
            axes[1, 0].plot(cusum_values)
            axes[1, 0].axhline(
                y=self.win_rate_cusum.threshold,
                color='r',
                linestyle='--',
                label='Threshold'
            )
            axes[1, 0].set_title('Win Rate CUSUM')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('CUSUM')
            axes[1, 0].legend()

        # Alert timeline
        if self.alerts:
            alert_times = [a['timestamp'] for a in self.alerts[-20:]]
            axes[1, 1].scatter(
                range(len(alert_times)),
                [1] * len(alert_times),
                marker='x',
                s=100,
                color='red'
            )
            axes[1, 1].set_title('Recent Alerts')
            axes[1, 1].set_xlabel('Alert Index')
            axes[1, 1].set_yticks([])

        plt.tight_layout()
        return fig


# Example Usage

print("=" * 70)
print("LLM PRODUCTION MONITORING")
print("=" * 70)

# Create monitor
monitor = LLMProductionMonitor(
    baseline_win_rate=0.62,
    baseline_sharpe=1.8,
    psi_threshold=0.20
)

# Generate baseline data (training period)
np.random.seed(42)
baseline_signals = np.random.randn(500) * 0.5
baseline_confidences = np.random.beta(5, 2, 500)

monitor.fit_baseline(baseline_signals, baseline_confidences)

print("\nBaseline established:")
print(f"  Signals: mean={baseline_signals.mean():.3f}, std={baseline_signals.std():.3f}")
print(f"  Confidences: mean={baseline_confidences.mean():.3f}")

# Simulate production signals (with drift)
print("\n" + "-" * 70)
print("Simulating production signals...")
print("-" * 70)

for i in range(200):
    # Introduce drift after 100 signals
    if i < 100:
        # Normal signals
        signal_value = np.random.randn() * 0.5
        confidence = np.random.beta(5, 2)
        outcome = np.random.random() < 0.62  # Baseline win rate
    else:
        # Drifted signals (lower confidence, worse win rate)
        signal_value = np.random.randn() * 0.5 - 0.2  # Shifted mean
        confidence = np.random.beta(3, 2)  # Lower confidence
        outcome = np.random.random() < 0.52  # Degraded win rate

    # Log signal
    direction = 'bullish' if signal_value > 0 else ('bearish' if signal_value < 0 else 'neutral')
    monitor.log_signal({
        'direction': direction,
        'strength': abs(signal_value),
        'confidence': confidence
    })

    # Log outcome
    monitor.log_outcome(outcome)

    # Check drift every 20 signals
    if (i + 1) % 50 == 0:
        status = monitor.get_status()
        print(f"\nSignal {i+1} Status: {status['health'].upper()}")
        if 'signal_drift' in status['drift']:
            print(f"  Signal PSI: {status['drift']['signal_drift']['psi']:.3f}")
        if 'confidence_drift' in status['drift']:
            print(f"  Confidence PSI: {status['drift']['confidence_drift']['psi']:.3f}")

# Final status
print("\n" + "=" * 70)
print("FINAL STATUS")
print("=" * 70)

final_status = monitor.get_status()
print(f"\nOverall Health: {final_status['health'].upper()}")
print(f"Total Alerts: {len(monitor.alerts)}")
print(f"\nRecent Metrics:")
print(f"  Mean Confidence: {final_status['metrics']['recent_confidences_mean']:.3f}")
print(f"  Confidence Std: {final_status['metrics']['recent_confidences_std']:.3f}")

if final_status['drift']:
    print(f"\nDrift Detection:")
    for name, metrics in final_status['drift'].items():
        status_icon = "⚠" if metrics['alert'] else "✓"
        print(f"  {status_icon} {name}: PSI = {metrics['psi']:.3f}")
```

</div>
</div>

## Common Pitfalls

**1. Monitoring Lag**
- Problem: Daily metrics not caught until week of poor performance
- Symptom: Significant losses before drift detected
- Solution: Multi-timescale monitoring (hourly, daily, weekly), early warning thresholds

**2. Alert Fatigue**
- Problem: Too many false positive alerts
- Symptom: Team ignores alerts, misses real drift
- Solution: Tune thresholds on historical data, use alert severity levels

**3. Insufficient Baseline Data**
- Problem: PSI calculated on 50 baseline samples
- Symptom: High false positive rate
- Solution: Collect 200+ baseline samples, use rolling baselines

**4. Ignoring Market Regimes**
- Problem: Single baseline for all market conditions
- Symptom: False alerts during normal volatility spikes
- Solution: Regime-conditional baselines (bull/bear/high vol)

**5. No Semantic Monitoring**
- Problem: Only tracking numeric metrics, missing reasoning quality
- Symptom: Signals technically correct but reasoning deteriorates
- Solution: Sample and manually review LLM reasoning weekly

## Connections

**Builds on:**
- Module 5.2: Confidence scoring (monitor calibration drift)
- Module 5.3: Backtesting (establish baseline metrics)
- Module 6.1: Production deployment (monitor what's deployed)

**Leads to:**
- Module 6.3: Optimization (monitoring identifies optimization opportunities)
- Incident response (alerts trigger manual review)
- Model retraining (drift detection triggers retraining pipeline)

**Related concepts:**
- A/B testing (compare current vs baseline models)
- Anomaly detection (outlier signals, unusual reasoning)
- MLOps (model versioning, experiment tracking)

## Practice Problems

1. **PSI Calculation**
   Baseline: 60% bullish, 20% neutral, 20% bearish
   Current: 45% bullish, 25% neutral, 30% bearish
   Calculate PSI. Is drift significant (threshold 0.20)?

2. **CUSUM Design**
   Baseline win rate: 62%
   Want to detect drop to 55%
   Acceptable delay: 20 trades
   Calculate: allowance (k), threshold (h)?

3. **Alert Prioritization**
   Multiple alerts fire simultaneously:
   - Signal PSI = 0.22
   - Confidence PSI = 0.15
   - Win rate CUSUM = 6.0 (threshold 5.0)
   - API latency p99 = 8s (threshold 5s)

   Which to investigate first? Why?

4. **Rolling Baseline**
   Should baseline be fixed or rolling window?
   If rolling, what window size for daily signals?
   Trade-offs?

5. **Multi-Model Monitoring**
   Running 3 LLM variants in production (A/B/C).
   Model A: PSI = 0.10, Win rate = 60%
   Model B: PSI = 0.25, Win rate = 63%
   Model C: PSI = 0.18, Win rate = 58%

   Which model to promote? Consider drift vs performance.

<div class="callout-insight">

**Insight:** Understanding production llm monitoring and drift detection is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

**Drift Detection:**
<div class="flow">
<div class="flow-step mint">1. "A Survey on Concept...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. "Learning under Conc...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. "Detecting and Corre...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. "Monitoring Machine ...</div>
</div>


1. **"A Survey on Concept Drift Adaptation"** by Gama et al. - Comprehensive drift survey
2. **"Learning under Concept Drift: A Review"** by Lu et al. - Drift detection methods
3. **"Detecting and Correcting for Label Shift with Black Box Predictors"** - Label shift

**Monitoring ML Systems:**
4. **"Monitoring Machine Learning Models in Production"** by Google - Production best practices
5. **"Hidden Technical Debt in Machine Learning Systems"** - Monitoring debt
6. **"Machine Learning Operations (MLOps)"** - End-to-end monitoring

**Statistical Process Control:**
7. **"Introduction to Statistical Quality Control"** by Montgomery - CUSUM, control charts
8. **"Sequential Analysis"** by Wald - Sequential testing theory

**LLM-Specific:**
9. **"Emergent and Predictable Memorization in Large Language Models"** - LLM behavior changes
10. **"On the Dangers of Stochastic Parrots"** - LLM failure modes

**Financial Monitoring:**
11. **"Model Risk Management"** by OCC - Regulatory guidance
12. **"Algorithmic Trading: A Practitioner's Guide"** - Production monitoring for trading

*"Monitor early, monitor often. Silent failures are the most expensive."*

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---



## Cross-References

<a class="link-card" href="./02_monitoring_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_pipeline_build.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_commodity_agents.md">
  <div class="link-card-title">01 Commodity Agents</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_production_deployment.md">
  <div class="link-card-title">01 Production Deployment</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

