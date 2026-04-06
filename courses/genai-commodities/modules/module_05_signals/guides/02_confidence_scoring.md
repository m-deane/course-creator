# LLM Confidence Calibration and Uncertainty Quantification

> **Reading time:** ~12 min | **Module:** Module 5: Signals | **Prerequisites:** Modules 0-4

<div class="callout-key">

**Key Concept Summary:** Confidence scoring quantifies the reliability of LLM-generated signals by measuring how well stated confidence levels match actual prediction accuracy. Well-calibrated confidence enables optimal position sizing, risk management, and systematic evaluation of when to trust LLM outputs versus defaul...

</div>

## In Brief

Confidence scoring quantifies the reliability of LLM-generated signals by measuring how well stated confidence levels match actual prediction accuracy. Well-calibrated confidence enables optimal position sizing, risk management, and systematic evaluation of when to trust LLM outputs versus defaulting to safer alternatives.

<div class="callout-insight">

**Insight:** LLMs are notoriously overconfident—claiming 90% certainty on predictions that are correct only 60% of the time. Calibration transforms subjective LLM confidence into objective probability estimates through empirical validation. A calibrated confidence score of 0.75 means "historically, signals with this confidence were correct 75% of the time," enabling quantitative risk assessment and Kelly-optimal position sizing.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

## Formal Definition

### Confidence Calibration

**Expected Calibration Error (ECE):**

Given N predictions with confidence scores $\{(c_i, y_i, \hat{y}_i)\}_{i=1}^N$:

1. Partition predictions into M bins $B_m = \{i : c_i \in [\frac{m-1}{M}, \frac{m}{M})\}$

2. Compute bin accuracy: $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{1}[\hat{y}_i = y_i]$

3. Compute bin confidence: $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} c_i$

4. ECE: $\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$

**Perfect calibration:** ECE = 0 (confidence equals accuracy in all bins)

### Calibration Methods

**1. Platt Scaling (Logistic Calibration)**

$$P(y=1|c) = \frac{1}{1 + \exp(A \cdot c + B)}$$

Fit A, B on validation set to map raw confidence to calibrated probability.

**2. Isotonic Regression**

Non-parametric monotonic mapping:
$$f: [0,1] \rightarrow [0,1]$$

Learns arbitrary monotonic relationship between confidence and accuracy.

**3. Temperature Scaling**

For logit-based confidence:
$$\text{conf}_{\text{calibrated}} = \text{softmax}(\frac{\text{logit}}{T})$$

Single temperature parameter T learned on validation set.

### Uncertainty Quantification

**Epistemic Uncertainty** (model uncertainty):
- Generate multiple predictions with different prompts/temperatures
- Measure variance: $\sigma^2_{\text{epistemic}} = \text{Var}[\hat{y}_1, \hat{y}_2, ..., \hat{y}_K]$

**Aleatoric Uncertainty** (data uncertainty):
- Irreducible uncertainty in the data
- LLM should output confidence intervals, not just point estimates

**Total Uncertainty:**
$$\sigma^2_{\text{total}} = \sigma^2_{\text{epistemic}} + \sigma^2_{\text{aleatoric}}$$

### The Overconfidence Problem

**Uncalibrated LLM:**
- Prediction: "Bullish"
- LLM Confidence: 0.95
- Historical Accuracy: 0.63 (signal was right 63% of the time)
- Problem: Over-sizing position based on false confidence

**Calibrated System:**
- Same prediction: "Bullish"
- Raw LLM Confidence: 0.95
- Calibrated Confidence: 0.67 (after empirical correction)
- Action: Appropriately sized position reflecting true reliability

### Why Calibration Matters

**Position Sizing Example:**

| Signal | Raw Confidence | Calibrated | Position Size | Outcome |
|--------|----------------|------------|---------------|---------|
| A | 0.90 | 0.60 | 60% of base | Correct sizing |
| B | 0.95 | 0.65 | 65% of base | Correct sizing |
| C | 0.70 | 0.55 | 55% of base | Avoid over-risk |

Without calibration, all signals appear highly confident → over-leveraged portfolio.

### Ensemble Disagreement as Confidence

Generate 5 signals for the same context:
- 4 say "Bullish", 1 says "Bearish"
- Agreement rate: 80%
- Ensemble confidence: 0.80
- Interpretation: Moderate confidence, conflicting views exist

## Code Implementation

### Confidence Calibration Framework


<span class="filename">confidencecalibrator.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from anthropic import Anthropic
from typing import List, Tuple, Dict
import json

class ConfidenceCalibrator:
    """
    Calibrate LLM confidence scores to match empirical accuracy.
    """

    def __init__(self, method: str = 'platt'):
        """
        Initialize calibrator.

        Args:
            method: 'platt' (logistic), 'isotonic', or 'histogram'
        """
        self.method = method
        self.calibrator = None
        self.calibrated = False

    def fit(self, confidences: np.ndarray, outcomes: np.ndarray):
        """
        Fit calibration model.

        Args:
            confidences: Raw confidence scores [0, 1]
            outcomes: Binary outcomes (1=correct, 0=incorrect)
        """
        confidences = np.array(confidences).reshape(-1, 1)
        outcomes = np.array(outcomes)

        if self.method == 'platt':
            # Logistic regression calibration
            self.calibrator = LogisticRegression()
            self.calibrator.fit(confidences, outcomes)

        elif self.method == 'isotonic':
            # Isotonic regression calibration
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(confidences.ravel(), outcomes)

        elif self.method == 'histogram':
            # Histogram binning
            n_bins = 10
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(confidences.ravel(), bins) - 1

            self.bin_accuracies = {}
            for i in range(n_bins):
                mask = bin_indices == i
                if mask.sum() > 0:
                    self.bin_accuracies[i] = outcomes[mask].mean()
                else:
                    self.bin_accuracies[i] = 0.5

            self.bins = bins

        self.calibrated = True

    def predict(self, confidences: np.ndarray) -> np.ndarray:
        """
        Return calibrated confidence scores.

        Args:
            confidences: Raw confidence scores

        Returns:
            Calibrated probabilities
        """
        if not self.calibrated:
            raise ValueError("Calibrator not fitted")

        confidences = np.array(confidences).reshape(-1, 1)

        if self.method == 'platt':
            return self.calibrator.predict_proba(confidences)[:, 1]

        elif self.method == 'isotonic':
            return self.calibrator.predict(confidences.ravel())

        elif self.method == 'histogram':
            bin_indices = np.digitize(confidences.ravel(), self.bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(self.bin_accuracies) - 1)
            return np.array([self.bin_accuracies[i] for i in bin_indices])

    def evaluate(
        self,
        confidences: np.ndarray,
        outcomes: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute calibration metrics.

        Returns:
            Dictionary with ECE, MCE, and Brier score
        """
        confidences = np.array(confidences)
        outcomes = np.array(outcomes)

        # Expected Calibration Error (ECE)
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1

        ece = 0
        mce = 0

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = outcomes[mask].mean()
                bin_weight = mask.sum() / len(confidences)

                ece += bin_weight * abs(bin_acc - bin_conf)
                mce = max(mce, abs(bin_acc - bin_conf))

        # Brier score (lower is better)
        brier = brier_score_loss(outcomes, confidences)

        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'mean_confidence': confidences.mean(),
            'accuracy': outcomes.mean()
        }

    def plot_calibration(
        self,
        confidences: np.ndarray,
        outcomes: np.ndarray,
        title: str = "Calibration Curve"
    ):
        """
        Plot calibration diagram.
        """
        # Bin predictions
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1

        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_confidences.append(confidences[mask].mean())
                bin_accuracies.append(outcomes[mask].mean())
                bin_counts.append(mask.sum())

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Calibration curve
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.scatter(bin_confidences, bin_accuracies,
                   s=[c*3 for c in bin_counts], alpha=0.6)
        ax1.set_xlabel('Mean Predicted Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{title}\nECE: {self.evaluate(confidences, outcomes)["ece"]:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Confidence histogram
        ax2.hist(confidences, bins=20, alpha=0.6, edgecolor='black')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class EnsembleConfidence:
    """
    Use ensemble disagreement as uncertainty measure.
    """

    def __init__(self, anthropic_api_key: str, n_samples: int = 5):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.n_samples = n_samples

    def generate_ensemble(
        self,
        prompt: str,
        temperature: float = 0.8
    ) -> List[Dict]:
        """
        Generate multiple predictions for same prompt.

        Returns:
            List of predictions with parsed direction and strength
        """
        predictions = []

        for i in range(self.n_samples):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            try:
                # Parse JSON response
                pred = json.loads(response.content[0].text)
                predictions.append(pred)
            except json.JSONDecodeError:
                # Handle non-JSON responses
                text = response.content[0].text.lower()
                direction = 'bullish' if 'bullish' in text else (
                    'bearish' if 'bearish' in text else 'neutral'
                )
                predictions.append({
                    'direction': direction,
                    'strength': 0.5
                })

        return predictions

    def compute_confidence(
        self,
        predictions: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute ensemble confidence metrics.

        Returns:
            Dictionary with agreement rate, mean strength, std, entropy
        """
        directions = [p['direction'] for p in predictions]
        strengths = [p.get('strength', 0.5) for p in predictions]

        # Agreement rate (most common direction)
        from collections import Counter
        direction_counts = Counter(directions)
        most_common = direction_counts.most_common(1)[0]
        agreement_rate = most_common[1] / len(directions)

        # Direction consensus
        consensus_direction = most_common[0]

        # Strength statistics
        mean_strength = np.mean(strengths)
        std_strength = np.std(strengths)

        # Entropy (uncertainty measure)
        probs = np.array(list(direction_counts.values())) / len(directions)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return {
            'consensus_direction': consensus_direction,
            'agreement_rate': agreement_rate,
            'mean_strength': mean_strength,
            'std_strength': std_strength,
            'entropy': entropy,
            'ensemble_confidence': agreement_rate * mean_strength,
            'n_samples': len(predictions)
        }


# Example Usage

# Simulate historical LLM predictions
np.random.seed(42)
n_predictions = 500

# Generate overconfident predictions
raw_confidences = np.random.beta(8, 2, n_predictions)  # Skewed toward high confidence

# Actual outcomes (not as good as confidence suggests)
true_skill = 0.65
outcomes = (np.random.random(n_predictions) < (raw_confidences * true_skill)).astype(int)

print("=" * 70)
print("CONFIDENCE CALIBRATION EXAMPLE")
print("=" * 70)

# Evaluate uncalibrated
calibrator = ConfidenceCalibrator(method='platt')
uncalibrated_metrics = calibrator.evaluate(raw_confidences, outcomes)

print("\nUncalibrated Metrics:")
print(f"  Mean Confidence: {uncalibrated_metrics['mean_confidence']:.3f}")
print(f"  Actual Accuracy: {uncalibrated_metrics['accuracy']:.3f}")
print(f"  ECE (calibration error): {uncalibrated_metrics['ece']:.3f}")
print(f"  Brier Score: {uncalibrated_metrics['brier_score']:.3f}")

# Split data
train_size = int(0.7 * n_predictions)
train_conf, test_conf = raw_confidences[:train_size], raw_confidences[train_size:]
train_out, test_out = outcomes[:train_size], outcomes[train_size:]

# Fit calibrator
calibrator.fit(train_conf, train_out)

# Calibrate test set
calibrated_conf = calibrator.predict(test_conf)

# Evaluate calibrated
calibrated_metrics = calibrator.evaluate(calibrated_conf, test_out)

print("\nCalibrated Metrics (Test Set):")
print(f"  Mean Confidence: {calibrated_metrics['mean_confidence']:.3f}")
print(f"  Actual Accuracy: {calibrated_metrics['accuracy']:.3f}")
print(f"  ECE (calibration error): {calibrated_metrics['ece']:.3f}")
print(f"  Brier Score: {calibrated_metrics['brier_score']:.3f}")

print("\nCalibration Improvement:")
print(f"  ECE Reduction: {(1 - calibrated_metrics['ece']/uncalibrated_metrics['ece'])*100:.1f}%")

# Plot calibration curves
fig = calibrator.plot_calibration(test_conf, test_out, "Before Calibration")
plt.savefig('uncalibrated_curve.png', dpi=150, bbox_inches='tight')

fig2 = calibrator.plot_calibration(calibrated_conf, test_out, "After Calibration")
plt.savefig('calibrated_curve.png', dpi=150, bbox_inches='tight')

print("\nCalibration curves saved to uncalibrated_curve.png and calibrated_curve.png")


# Ensemble Confidence Example

print("\n" + "=" * 70)
print("ENSEMBLE CONFIDENCE EXAMPLE")
print("=" * 70)

# Simulated ensemble for illustration
ensemble_prompt = """
Analyze crude oil outlook and provide signal as JSON:
{
  "direction": "bullish" | "bearish" | "neutral",
  "strength": 0.0-1.0
}

Context: OPEC production cuts announced, but recession fears growing.
"""

# Simulate ensemble predictions
simulated_ensemble = [
    {'direction': 'bullish', 'strength': 0.75},
    {'direction': 'bullish', 'strength': 0.70},
    {'direction': 'bullish', 'strength': 0.80},
    {'direction': 'bearish', 'strength': 0.60},
    {'direction': 'neutral', 'strength': 0.50}
]

ensemble_conf = EnsembleConfidence(anthropic_api_key="dummy", n_samples=5)
confidence_metrics = ensemble_conf.compute_confidence(simulated_ensemble)

print("\nEnsemble Analysis:")
print(f"  Consensus Direction: {confidence_metrics['consensus_direction']}")
print(f"  Agreement Rate: {confidence_metrics['agreement_rate']:.2%}")
print(f"  Mean Strength: {confidence_metrics['mean_strength']:.2f}")
print(f"  Std Strength: {confidence_metrics['std_strength']:.2f}")
print(f"  Entropy (uncertainty): {confidence_metrics['entropy']:.3f}")
print(f"  Final Ensemble Confidence: {confidence_metrics['ensemble_confidence']:.2f}")

print("\nInterpretation:")
if confidence_metrics['agreement_rate'] > 0.7:
    print("  HIGH AGREEMENT - Strong consensus among ensemble members")
elif confidence_metrics['agreement_rate'] > 0.5:
    print("  MODERATE AGREEMENT - Some disagreement, reduce position size")
else:
    print("  LOW AGREEMENT - Conflicting signals, consider staying flat")
```

</div>

### Position Sizing with Calibrated Confidence


<span class="filename">calibratedpositionsizer.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
class CalibratedPositionSizer:
    """
    Size positions using calibrated confidence scores.
    """

    def __init__(
        self,
        account_value: float,
        max_risk_per_trade: float = 0.02,
        kelly_fraction: float = 0.25  # Use 1/4 Kelly for safety
    ):
        self.account_value = account_value
        self.max_risk_per_trade = max_risk_per_trade
        self.kelly_fraction = kelly_fraction

    def kelly_size(
        self,
        win_prob: float,
        win_loss_ratio: float = 2.0
    ) -> float:
        """
        Kelly criterion position size.

        Args:
            win_prob: Calibrated probability of winning
            win_loss_ratio: Average win / average loss

        Returns:
            Optimal position size as fraction of account
        """
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        return max(0, kelly * self.kelly_fraction)

    def size_from_confidence(
        self,
        calibrated_confidence: float,
        entry_price: float,
        stop_loss: float
    ) -> Dict[str, float]:
        """
        Calculate position size from calibrated confidence.

        Args:
            calibrated_confidence: Calibrated win probability
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Position details
        """
        # Risk per unit
        risk_per_unit = abs(entry_price - stop_loss)

        # Kelly-based sizing
        kelly_pct = self.kelly_size(calibrated_confidence)
        kelly_dollars = self.account_value * kelly_pct

        # Risk-based constraint
        max_risk_dollars = self.account_value * self.max_risk_per_trade
        max_units_by_risk = max_risk_dollars / risk_per_unit

        # Position size
        target_units = kelly_dollars / entry_price
        final_units = min(target_units, max_units_by_risk)

        return {
            'units': final_units,
            'dollar_size': final_units * entry_price,
            'risk_amount': final_units * risk_per_unit,
            'risk_pct': (final_units * risk_per_unit) / self.account_value,
            'position_pct': (final_units * entry_price) / self.account_value,
            'kelly_fraction_used': kelly_pct,
            'confidence': calibrated_confidence
        }


# Example position sizing
sizer = CalibratedPositionSizer(account_value=1_000_000)

# Compare uncalibrated vs calibrated
scenarios = [
    {'name': 'High Confidence', 'raw': 0.95, 'calibrated': 0.68, 'entry': 75, 'stop': 72},
    {'name': 'Medium Confidence', 'raw': 0.75, 'calibrated': 0.60, 'entry': 75, 'stop': 72},
    {'name': 'Low Confidence', 'raw': 0.60, 'calibrated': 0.52, 'entry': 75, 'stop': 72}
]

print("\n" + "=" * 70)
print("POSITION SIZING COMPARISON")
print("=" * 70)

for scenario in scenarios:
    uncal_pos = sizer.size_from_confidence(scenario['raw'], scenario['entry'], scenario['stop'])
    cal_pos = sizer.size_from_confidence(scenario['calibrated'], scenario['entry'], scenario['stop'])

    print(f"\n{scenario['name']}:")
    print(f"  Raw Confidence: {scenario['raw']:.2f} → Calibrated: {scenario['calibrated']:.2f}")
    print(f"  Uncalibrated Position: ${uncal_pos['dollar_size']:,.0f} ({uncal_pos['position_pct']*100:.1f}%)")
    print(f"  Calibrated Position: ${cal_pos['dollar_size']:,.0f} ({cal_pos['position_pct']*100:.1f}%)")
    print(f"  Size Reduction: {(1 - cal_pos['position_pct']/uncal_pos['position_pct'])*100:.1f}%")
```

</div>

## Common Pitfalls

**1. Insufficient Calibration Data**
- Problem: Calibrating on 50 samples yields unstable mapping
- Symptom: Wildly different calibrations across validation folds
- Solution: Collect at least 200-500 historical predictions before calibrating

**2. Non-Stationary Confidence**
- Problem: LLM confidence drift over time as market regime changes
- Symptom: Previously calibrated model becomes uncalibrated
- Solution: Recalibrate monthly, use rolling window validation

**3. Ignoring Epistemic Uncertainty**
- Problem: Using single LLM call without measuring model uncertainty
- Symptom: Overconfidence in edge cases where model is genuinely uncertain
- Solution: Always generate ensemble predictions for critical decisions

**4. Overfitting Calibration**
- Problem: Complex calibration method (e.g., deep neural network) on small dataset
- Symptom: Perfect calibration on validation, poor on test
- Solution: Use simple methods (Platt scaling, isotonic regression) unless data abundant

**5. Conflating Calibration with Discrimination**
- Problem: Well-calibrated model can still have poor accuracy
- Symptom: ECE is low but Brier score is high
- Solution: Monitor both calibration (ECE) and discrimination (accuracy, AUC)

## Connections

**Builds on:**
- Module 5.1: Signal frameworks (confidence scores are inputs to position sizing)
- Probability theory (Bayesian calibration, uncertainty quantification)
- Statistical learning (isotonic regression, logistic regression)

**Leads to:**
- Module 5.3: Backtesting (use calibrated confidence in historical simulations)
- Module 6: Production monitoring (track calibration drift in live system)
- Risk management (Kelly criterion, portfolio allocation)

**Related concepts:**
- Bayesian neural networks (epistemic uncertainty via dropout)
- Conformal prediction (distribution-free confidence intervals)
- Model ensembling (variance reduction, uncertainty quantification)

## Practice Problems

1. **Calibration Evaluation**
   Given 100 predictions with mean confidence 0.82 and accuracy 0.64:
   - What is the minimum possible ECE?
   - If binned into 10 bins, what's the maximum ECE?
   - How many samples needed for stable calibration?

2. **Ensemble Design**
   Design an ensemble system with:
   - 5 LLM calls with different temperatures
   - Agreement threshold: 80%
   - Action: LONG if 4/5 agree on bullish

   What is the false positive rate if each LLM has 70% accuracy independently?

3. **Kelly Sizing**
   - Calibrated win probability: 0.65
   - Average win/loss ratio: 1.8
   - Account: $1M, max risk: 2%

   Calculate: Full Kelly size? Quarter Kelly size? Risk per trade?

4. **Calibration Stability**
   You have 300 historical predictions. How many for calibration training vs validation?
   What if you split by time (70% train, 30% test) and calibration degrades?
   Diagnose: overfitting or non-stationarity?

5. **Multi-Horizon Calibration**
   Short-term signals (1-week): ECE = 0.12
   Long-term signals (3-month): ECE = 0.08

   Why might long-term be better calibrated?
   Should you use separate calibrators for each horizon?

<div class="callout-insight">

**Insight:** Understanding llm confidence calibration and uncertainty quantification is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

**Confidence Calibration:**
<div class="flow">
<div class="flow-step mint">1. "On Calibration of M...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. "Beyond temperature ...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. "Verified Uncertaint...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. "Dropout as a Bayesi...</div>
</div>


1. **"On Calibration of Modern Neural Networks"** by Guo et al. (2017) - Temperature scaling, ECE
2. **"Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration"** - Advanced methods
3. **"Verified Uncertainty Calibration"** by Kumar et al. - Theoretical guarantees

**Uncertainty Quantification:**
4. **"Dropout as a Bayesian Approximation"** by Gal & Ghahramani - Epistemic uncertainty
5. **"Predictive Uncertainty Estimation via Prior Networks"** - Aleatoric vs epistemic
6. **"What Uncertainties Do We Need in Bayesian Deep Learning?"** by Kendall & Gal

**Financial Applications:**
7. **"Kelly Capital Growth Investment Criterion"** by MacLean et al. - Position sizing
8. **"Fortune's Formula"** by Poundstone - Kelly criterion in practice
9. **"Enhancing Trading Strategies with Order Book Signals"** - Confidence in trading

**LLM-Specific:**
10. **"Language Models (Mostly) Know What They Know"** - LLM calibration research
11. **"Teaching Models to Express Their Uncertainty in Words"** - Verbalized confidence
12. **"Calibrating Large Language Models"** - Current best practices (2024)

*"Confidence without calibration is noise. Calibrated confidence is signal."*

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./02_confidence_scoring_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_signal_generation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_signal_frameworks.md">
  <div class="link-card-title">01 Signal Frameworks</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_signal_generation.md">
  <div class="link-card-title">01 Signal Generation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

