"""
Captum Neural Network Interpretability
Model attribution audit template.

Runs a systematic interpretability audit on a classification model:
1. Per-class mean attribution maps
2. Border artifact fraction screening
3. Method agreement (Spearman correlation across methods)
4. Convergence delta distribution check
5. Attribution drift baseline recording

Usage:
    python model_audit_template.py --model resnet18 --dataset cifar10

    Or import and customize:
        from model_audit_template import ModelAuditRunner
        runner = ModelAuditRunner(model, dataloader, class_names=CIFAR_CLASSES)
        report = runner.run_full_audit()
        runner.save_report("audit_report.json")
"""

import argparse
import hashlib
import json
import logging
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr, wasserstein_distance

from captum.attr import (
    GradientShap,
    IntegratedGradients,
    LayerGradCam,
    Saliency,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT REPORT DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PerClassAudit:
    class_index: int
    class_name: str
    n_examples: int
    mean_border_fraction: float
    max_attribution_quartile: float
    diagnosis: str  # "PASS" | "WARN" | "FAIL"


@dataclass
class MethodAgreementResult:
    method_a: str
    method_b: str
    spearman_r: float
    p_value: float
    agreement_level: str  # "high" | "moderate" | "low"


@dataclass
class ConvergenceAudit:
    method: str
    n_examples: int
    mean_abs_delta: float
    max_abs_delta: float
    fraction_compliant: float  # fraction with |delta| < 0.05
    compliant: bool  # overall pass/fail at 95% threshold


@dataclass
class AuditReport:
    report_id: str
    timestamp: str
    model_id: str
    n_examples_audited: int
    per_class_results: List[PerClassAudit]
    method_agreement: List[MethodAgreementResult]
    convergence_audit: List[ConvergenceAudit]
    overall_pass: bool
    findings: List[str]
    recommendations: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT RUNNER
# ─────────────────────────────────────────────────────────────────────────────


class ModelAuditRunner:
    """
    Systematic interpretability audit for a classification model.

    Args:
        model:       PyTorch model in eval mode
        dataloader:  DataLoader yielding (image_tensor, label_tensor) batches
        class_names: List of class name strings
        model_id:    Identifier string for the report
        gradcam_layer: Layer for GradCAM (required for method agreement audit)
        device:      Computation device
    """

    BORDER_FRACTION_WARN_THRESHOLD = 0.20  # flag if >20% attribution in border
    CONVERGENCE_THRESHOLD = 0.05           # |delta| must be < this
    COMPLIANCE_FRACTION = 0.95             # 95% of examples must be compliant

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: Any,
        class_names: Optional[List[str]] = None,
        model_id: str = "unnamed_model",
        gradcam_layer: Optional[Any] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.class_names = class_names or []
        self.model_id = model_id
        self.gradcam_layer = gradcam_layer
        self.device = device
        self._report: Optional[AuditReport] = None

    def _get_class_name(self, idx: int) -> str:
        if self.class_names and idx < len(self.class_names):
            return self.class_names[idx]
        return f"class_{idx}"

    def _compute_ig_attribution(
        self, inputs: torch.Tensor, target: int, n_steps: int = 50
    ) -> tuple:
        """Returns (attr_map (H,W), delta)."""
        ig = IntegratedGradients(lambda x: self.model(x))
        baseline = torch.zeros_like(inputs)
        attrs, delta = ig.attribute(
            inputs, baseline, target=target,
            n_steps=n_steps, return_convergence_delta=True,
        )
        attr_map = attrs.abs().sum(dim=1).squeeze(0).detach().cpu()
        return attr_map, float(delta.item())

    def _compute_saliency_attribution(
        self, inputs: torch.Tensor, target: int
    ) -> np.ndarray:
        sal = Saliency(lambda x: self.model(x))
        attrs = sal.attribute(inputs, target=target, abs=True)
        return attrs.sum(dim=1).squeeze(0).detach().cpu().numpy()

    def _compute_gradcam_attribution(
        self, inputs: torch.Tensor, target: int
    ) -> np.ndarray:
        gc = LayerGradCam(lambda x: self.model(x), self.gradcam_layer)
        attrs = gc.attribute(inputs, target=target)
        upsampled = torch.nn.functional.interpolate(
            attrs, size=inputs.shape[2:], mode="bilinear", align_corners=False
        )
        return upsampled.sum(dim=1).squeeze(0).detach().cpu().numpy()

    def _border_fraction(self, attr_map: np.ndarray, fraction: float = 0.1) -> float:
        """Fraction of total attribution in the outermost border region."""
        H, W = attr_map.shape
        bh, bw = max(1, int(H * fraction)), max(1, int(W * fraction))
        mask = np.zeros_like(attr_map)
        mask[:bh, :] = 1
        mask[-bh:, :] = 1
        mask[:, :bw] = 1
        mask[:, -bw:] = 1
        total = attr_map.sum() + 1e-8
        return float((attr_map * mask).sum() / total)

    # ── AUDIT STEP 1: Per-Class Attribution Analysis ──────────────────────

    def audit_per_class(self, n_examples_per_class: int = 20) -> List[PerClassAudit]:
        """
        Compute mean attribution statistics per class.
        Flags classes with high border attribution.
        """
        logger.info("Step 1/5: Per-class attribution analysis...")

        class_stats: Dict[int, Dict] = {}
        count_per_class: Dict[int, int] = {}

        for images, labels in self.dataloader:
            images = images.to(self.device)

            for img, label in zip(images, labels):
                cls_idx = int(label.item())
                if cls_idx not in count_per_class:
                    count_per_class[cls_idx] = 0
                    class_stats[cls_idx] = {"border_fractions": [], "top_q_fractions": []}

                if count_per_class[cls_idx] >= n_examples_per_class:
                    continue

                img = img.unsqueeze(0)
                with torch.no_grad():
                    pred = self.model(img).argmax(dim=1).item()

                attr_map = self._compute_saliency_attribution(img, pred)
                border_frac = self._border_fraction(attr_map)
                top_q = float((attr_map > np.percentile(attr_map, 75)).mean())

                class_stats[cls_idx]["border_fractions"].append(border_frac)
                class_stats[cls_idx]["top_q_fractions"].append(top_q)
                count_per_class[cls_idx] += 1

            if all(v >= n_examples_per_class for v in count_per_class.values()):
                break

        results = []
        for cls_idx, stats in sorted(class_stats.items()):
            border_fracs = stats["border_fractions"]
            top_q = stats["top_q_fractions"]
            mean_border = float(np.mean(border_fracs)) if border_fracs else 0.0
            mean_top_q = float(np.mean(top_q)) if top_q else 0.0

            if mean_border > self.BORDER_FRACTION_WARN_THRESHOLD:
                diagnosis = "WARN"
            else:
                diagnosis = "PASS"

            results.append(PerClassAudit(
                class_index=cls_idx,
                class_name=self._get_class_name(cls_idx),
                n_examples=len(border_fracs),
                mean_border_fraction=round(mean_border, 4),
                max_attribution_quartile=round(mean_top_q, 4),
                diagnosis=diagnosis,
            ))
            logger.info(
                f"  Class {cls_idx} ({self._get_class_name(cls_idx):15s}): "
                f"border={mean_border:.1%}  [{diagnosis}]"
            )

        return results

    # ── AUDIT STEP 2: Method Agreement ────────────────────────────────────

    def audit_method_agreement(self, n_examples: int = 30) -> List[MethodAgreementResult]:
        """
        Compute Spearman correlation between attribution methods.
        Low agreement surfaces where methods fundamentally disagree.
        """
        logger.info("Step 2/5: Method agreement audit...")

        methods_available = ["saliency", "integrated_gradients"]
        if self.gradcam_layer is not None:
            methods_available.append("gradcam")

        attrs_per_method: Dict[str, List[np.ndarray]] = {m: [] for m in methods_available}
        count = 0

        for images, labels in self.dataloader:
            if count >= n_examples:
                break
            images = images.to(self.device)

            for img, label in zip(images, labels):
                if count >= n_examples:
                    break
                img = img.unsqueeze(0)
                with torch.no_grad():
                    pred = int(self.model(img).argmax(dim=1).item())

                attrs_per_method["saliency"].append(
                    self._compute_saliency_attribution(img, pred).flatten()
                )
                ig_map, _ = self._compute_ig_attribution(img, pred, n_steps=30)
                attrs_per_method["integrated_gradients"].append(
                    ig_map.numpy().flatten()
                )
                if "gradcam" in methods_available:
                    attrs_per_method["gradcam"].append(
                        self._compute_gradcam_attribution(img, pred).flatten()
                    )
                count += 1

        # Compute pairwise Spearman correlations
        results = []
        method_list = list(attrs_per_method.keys())
        for i in range(len(method_list)):
            for j in range(i + 1, len(method_list)):
                m_a, m_b = method_list[i], method_list[j]
                rs = []
                for a_arr, b_arr in zip(attrs_per_method[m_a], attrs_per_method[m_b]):
                    r, _ = spearmanr(np.abs(a_arr), np.abs(b_arr))
                    rs.append(r)
                mean_r = float(np.mean(rs))
                mean_p = 0.0  # p-value not averaged here

                if mean_r > 0.7:
                    agreement = "high"
                elif mean_r > 0.4:
                    agreement = "moderate"
                else:
                    agreement = "low"

                results.append(MethodAgreementResult(
                    method_a=m_a,
                    method_b=m_b,
                    spearman_r=round(mean_r, 4),
                    p_value=mean_p,
                    agreement_level=agreement,
                ))
                logger.info(f"  {m_a} vs {m_b}: r={mean_r:+.4f}  [{agreement}]")

        return results

    # ── AUDIT STEP 3: Convergence Delta Audit ─────────────────────────────

    def audit_convergence(self, n_examples: int = 50, n_steps: int = 50) -> List[ConvergenceAudit]:
        """
        Check that IG convergence deltas are within the compliance threshold.
        """
        logger.info("Step 3/5: Convergence delta audit (IG)...")

        deltas = []
        count = 0

        for images, labels in self.dataloader:
            if count >= n_examples:
                break
            images = images.to(self.device)

            for img, label in zip(images, labels):
                if count >= n_examples:
                    break
                img = img.unsqueeze(0)
                with torch.no_grad():
                    pred = int(self.model(img).argmax(dim=1).item())
                _, delta = self._compute_ig_attribution(img, pred, n_steps=n_steps)
                deltas.append(abs(delta))
                count += 1

        deltas_np = np.array(deltas)
        fraction_compliant = float((deltas_np < self.CONVERGENCE_THRESHOLD).mean())
        compliant = fraction_compliant >= self.COMPLIANCE_FRACTION

        result = ConvergenceAudit(
            method="integrated_gradients",
            n_examples=len(deltas),
            mean_abs_delta=round(float(deltas_np.mean()), 6),
            max_abs_delta=round(float(deltas_np.max()), 6),
            fraction_compliant=round(fraction_compliant, 4),
            compliant=compliant,
        )
        logger.info(
            f"  IG (n_steps={n_steps}): "
            f"mean_|delta|={result.mean_abs_delta:.6f}  "
            f"compliant_frac={fraction_compliant:.1%}  "
            f"[{'PASS' if compliant else 'FAIL'}]"
        )
        return [result]

    # ── AUDIT STEP 4: Generate Findings and Recommendations ───────────────

    def _generate_findings(
        self,
        per_class: List[PerClassAudit],
        method_agreement: List[MethodAgreementResult],
        convergence: List[ConvergenceAudit],
    ) -> tuple:
        """Return (findings: list[str], recommendations: list[str])."""
        findings: List[str] = []
        recommendations: List[str] = []

        # Border attribution findings
        warned_classes = [r for r in per_class if r.diagnosis == "WARN"]
        if warned_classes:
            names = [r.class_name for r in warned_classes]
            findings.append(
                f"High border attribution detected in {len(warned_classes)} class(es): "
                f"{', '.join(names[:5])}. "
                f"This may indicate metadata artifacts or dataset construction issues."
            )
            recommendations.append(
                "Inspect images from flagged classes for watermarks, borders, or metadata overlays. "
                "Consider cropping or padding strategies to remove border content."
            )

        # Method disagreement findings
        low_agreement = [r for r in method_agreement if r.agreement_level == "low"]
        if low_agreement:
            pairs = [f"{r.method_a} vs {r.method_b} (r={r.spearman_r:.3f})"
                     for r in low_agreement]
            findings.append(
                f"Low method agreement detected for: {'; '.join(pairs)}. "
                "These methods capture different aspects of model behavior."
            )
            recommendations.append(
                "Use IG for compliance reporting (completeness guarantee). "
                "Investigate examples where Saliency and IG disagree strongly — "
                "these often reveal saturation or shortcut regions."
            )

        # Convergence findings
        for conv in convergence:
            if not conv.compliant:
                findings.append(
                    f"Convergence audit FAILED: only {conv.fraction_compliant:.1%} of "
                    f"{conv.n_examples} examples have |delta| < {self.CONVERGENCE_THRESHOLD} "
                    f"(mean |delta| = {conv.mean_abs_delta:.5f}). "
                    "IG attributions may not satisfy the completeness axiom."
                )
                recommendations.append(
                    f"Increase n_steps from current value. "
                    f"For mean |delta| < {self.CONVERGENCE_THRESHOLD}, n_steps=100 is typically sufficient. "
                    "For compliance-grade reports, use n_steps=200."
                )

        if not findings:
            findings.append("No significant issues detected in this audit.")

        return findings, recommendations

    # ── FULL AUDIT ────────────────────────────────────────────────────────

    def run_full_audit(
        self,
        n_examples_per_class: int = 20,
        n_agreement_examples: int = 30,
        n_convergence_examples: int = 50,
        n_steps: int = 50,
    ) -> AuditReport:
        """
        Run all audit steps and generate a report.

        Returns:
            AuditReport dataclass
        """
        logger.info(f"Starting full audit for model: {self.model_id}")
        logger.info("=" * 60)

        per_class = self.audit_per_class(n_examples_per_class)
        method_agreement = self.audit_method_agreement(n_agreement_examples)
        convergence = self.audit_convergence(n_convergence_examples, n_steps)

        findings, recommendations = self._generate_findings(
            per_class, method_agreement, convergence
        )

        warn_classes = sum(1 for r in per_class if r.diagnosis in ("WARN", "FAIL"))
        conv_pass = all(c.compliant for c in convergence)
        overall_pass = (warn_classes == 0) and conv_pass

        self._report = AuditReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_id=self.model_id,
            n_examples_audited=sum(r.n_examples for r in per_class),
            per_class_results=per_class,
            method_agreement=method_agreement,
            convergence_audit=convergence,
            overall_pass=overall_pass,
            findings=findings,
            recommendations=recommendations,
        )

        logger.info("=" * 60)
        logger.info(f"Audit complete. Overall: {'PASS' if overall_pass else 'FAIL'}")
        logger.info(f"Findings: {len(findings)}")
        for f in findings:
            logger.info(f"  - {f[:80]}...")

        return self._report

    def save_report(self, path: str) -> None:
        """Save the audit report to a JSON file."""
        if self._report is None:
            raise RuntimeError("Run run_full_audit() before saving.")

        report_dict = asdict(self._report)
        with open(path, "w") as f:
            json.dump(report_dict, f, indent=2)
        logger.info(f"Audit report saved: {path}")

    def plot_summary(self, save_path: Optional[str] = None) -> plt.Figure:
        """Generate a summary figure for the audit report."""
        if self._report is None:
            raise RuntimeError("Run run_full_audit() before plotting.")

        report = self._report
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Per-class border fractions
        class_names = [r.class_name[:12] for r in report.per_class_results]
        border_fracs = [r.mean_border_fraction for r in report.per_class_results]
        colors = ["#d62728" if r.diagnosis == "WARN" else "#2ca02c"
                  for r in report.per_class_results]

        axes[0].barh(range(len(class_names)), border_fracs, color=colors, alpha=0.85)
        axes[0].axvline(self.BORDER_FRACTION_WARN_THRESHOLD, color="orange",
                         linestyle="--", linewidth=1.5, label=f"Threshold ({self.BORDER_FRACTION_WARN_THRESHOLD:.0%})")
        axes[0].set_yticks(range(len(class_names)))
        axes[0].set_yticklabels(class_names, fontsize=9)
        axes[0].set_xlabel("Mean Border Attribution Fraction")
        axes[0].set_title("Per-Class Border Attribution Audit", fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="x", alpha=0.3)

        # Method agreement
        if report.method_agreement:
            pairs = [f"{r.method_a[:8]}\nvs\n{r.method_b[:8]}" for r in report.method_agreement]
            rs = [r.spearman_r for r in report.method_agreement]
            bar_colors = ["#2ca02c" if r > 0.7 else "#FF9800" if r > 0.4 else "#d62728"
                          for r in rs]
            axes[1].bar(range(len(pairs)), rs, color=bar_colors, alpha=0.85)
            axes[1].set_xticks(range(len(pairs)))
            axes[1].set_xticklabels(pairs, fontsize=8)
            axes[1].set_ylabel("Spearman r (mean |attribution|)")
            axes[1].set_title("Method Agreement (Spearman r)", fontweight="bold")
            axes[1].axhline(0.7, color="green", linestyle="--", linewidth=1, label="High (0.7)")
            axes[1].axhline(0.4, color="orange", linestyle="--", linewidth=1, label="Moderate (0.4)")
            axes[1].set_ylim(-0.1, 1.05)
            axes[1].legend(fontsize=8)
            axes[1].grid(axis="y", alpha=0.3)

        status = "PASS" if report.overall_pass else "FAIL"
        fig.suptitle(
            f"Interpretability Audit Report — {report.model_id}  [{status}]\n"
            f"{report.timestamp[:19]}  |  {report.n_examples_audited} examples",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Captum model attribution audit")
    parser.add_argument("--model", default="resnet18",
                        help="Model name: resnet18 | resnet50")
    parser.add_argument("--dataset", default="cifar10",
                        help="Dataset: cifar10")
    parser.add_argument("--n-examples-per-class", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--output", default="audit_report.json")
    args = parser.parse_args()

    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18, ResNet18_Weights

    CIFAR_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    transform = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()

    dataset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=False, download=True, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    runner = ModelAuditRunner(
        model=model,
        dataloader=dataloader,
        class_names=CIFAR_CLASSES,
        model_id=f"{args.model}_{args.dataset}",
        gradcam_layer=model.layer4[-1],
    )

    report = runner.run_full_audit(
        n_examples_per_class=args.n_examples_per_class,
        n_agreement_examples=20,
        n_convergence_examples=20,
        n_steps=args.n_steps,
    )
    runner.save_report(args.output)
    fig = runner.plot_summary(save_path=args.output.replace(".json", ".png"))
    plt.show()
    print(f"\nAudit complete. Report saved to: {args.output}")
    print(f"Overall: {'PASS' if report.overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
