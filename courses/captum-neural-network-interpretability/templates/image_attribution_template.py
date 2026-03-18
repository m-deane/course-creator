"""
Captum Neural Network Interpretability
Production image attribution pipeline template.

Covers: CNN and ViT image models, multiple attribution methods,
baseline strategies, visualization, and batch processing.

Usage:
    from image_attribution_template import ImageAttributionPipeline

    pipeline = ImageAttributionPipeline(model, input_size=224)
    results = pipeline.attribute_image(image_path, target_class=281)
    pipeline.plot_results(results, save_path="attribution.png")
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from captum.attr import (
    GradientShap,
    IntegratedGradients,
    LayerGradCam,
    Saliency,
)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_imagenet_transform(input_size: int = 224) -> T.Compose:
    """Standard ImageNet preprocessing pipeline."""
    return T.Compose([
        T.Resize(int(input_size * 256 / 224)),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize_imagenet(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized ImageNet tensor to a displayable RGB numpy array.

    Args:
        tensor: (C, H, W) normalized tensor

    Returns:
        (H, W, 3) float32 array in [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE FACTORY
# ─────────────────────────────────────────────────────────────────────────────


def build_image_baseline(
    inputs: torch.Tensor,
    baseline_type: Literal["black", "white", "blur", "noise", "mean"] = "black",
    blur_sigma: float = 20.0,
    background_samples: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Construct a baseline for image attribution.

    Args:
        inputs:            (1, C, H, W) input tensor
        baseline_type:     One of 'black', 'white', 'blur', 'noise', 'mean'
        blur_sigma:        Gaussian blur sigma for 'blur' baseline
        background_samples: (N, C, H, W) tensor for 'mean' baseline

    Returns:
        (1, C, H, W) baseline tensor
    """
    if baseline_type == "black":
        return torch.zeros_like(inputs)

    elif baseline_type == "white":
        return torch.ones_like(inputs)

    elif baseline_type == "blur":
        # Gaussian blur in PIL then re-normalize
        from torchvision.transforms.functional import gaussian_blur
        # Approximate blur via repeated averaging
        blurred = inputs.clone()
        kernel_size = max(3, int(blur_sigma) | 1)  # ensure odd
        for _ in range(3):
            blurred = torch.nn.functional.avg_pool2d(
                blurred,
                kernel_size=min(kernel_size, 31),
                stride=1,
                padding=min(kernel_size, 31) // 2,
            )
        return blurred

    elif baseline_type == "noise":
        return torch.randn_like(inputs) * 0.1

    elif baseline_type == "mean":
        if background_samples is not None:
            return background_samples.mean(dim=0, keepdim=True).expand_as(inputs)
        # Fall back to black if no background provided
        return torch.zeros_like(inputs)

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTION METHODS
# ─────────────────────────────────────────────────────────────────────────────


class ImageAttributionPipeline:
    """
    Production image attribution pipeline supporting multiple Captum methods.

    Attributes:
        model:          PyTorch model in eval mode
        input_size:     Expected input side length (default 224)
        transform:      Preprocessing transform applied to loaded images
        class_names:    Optional list of class name strings
        gradcam_layer:  Layer module for GradCAM (required for 'gradcam' method)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_size: int = 224,
        transform: Optional[T.Compose] = None,
        class_names: Optional[List[str]] = None,
        gradcam_layer: Optional[Any] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.input_size = input_size
        self.transform = transform or build_imagenet_transform(input_size)
        self.class_names = class_names
        self.gradcam_layer = gradcam_layer
        self.device = device

    def load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess an image from disk.

        Returns:
            (1, C, H, W) tensor on self.device
        """
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        Run the model and return prediction details.

        Returns dict with: class_index, class_name, confidence, probs
        """
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = int(probs.argmax().item())
            confidence = float(probs[pred_class].item())

        class_name = (
            self.class_names[pred_class]
            if self.class_names and pred_class < len(self.class_names)
            else str(pred_class)
        )

        return {
            "class_index": pred_class,
            "class_name": class_name,
            "confidence": confidence,
            "probs": probs.cpu(),
        }

    def attribute(
        self,
        inputs: torch.Tensor,
        target_class: int,
        method: Literal[
            "integrated_gradients",
            "saliency",
            "gradient_shap",
            "gradcam",
        ] = "integrated_gradients",
        baseline_type: Literal["black", "white", "blur", "noise", "mean"] = "black",
        n_steps: int = 50,
        background_samples: Optional[torch.Tensor] = None,
        return_delta: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute attribution for inputs toward target_class.

        Returns dict with: attr_map (H, W), attrs_signed (H, W), delta
        """

        def forward(x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        baseline = build_image_baseline(inputs, baseline_type, background_samples=background_samples)

        delta_val = None

        if method == "integrated_gradients":
            ig = IntegratedGradients(forward)
            attrs, delta = ig.attribute(
                inputs,
                baseline,
                target=target_class,
                n_steps=n_steps,
                return_convergence_delta=True,
            )
            delta_val = float(delta.item())

        elif method == "saliency":
            sal = Saliency(forward)
            attrs = sal.attribute(inputs, target=target_class, abs=False)

        elif method == "gradient_shap":
            gs = GradientShap(forward)
            bg = baseline.expand(8, *baseline.shape[1:])
            attrs, delta = gs.attribute(
                inputs,
                bg,
                n_samples=50,
                target=target_class,
                return_convergence_delta=True,
            )
            delta_val = float(delta.item())

        elif method == "gradcam":
            if self.gradcam_layer is None:
                raise ValueError(
                    "Set gradcam_layer when constructing ImageAttributionPipeline to use GradCAM"
                )
            gc = LayerGradCam(forward, self.gradcam_layer)
            attrs = gc.attribute(inputs, target=target_class)
            # Upsample GradCAM to input resolution
            attrs = torch.nn.functional.interpolate(
                attrs,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        else:
            raise ValueError(f"Unknown method: {method!r}")

        # Aggregate over channel dimension → (H, W)
        attr_map = attrs.abs().sum(dim=1).squeeze(0).detach().cpu()
        attrs_signed = attrs.sum(dim=1).squeeze(0).detach().cpu()

        return {
            "method": method,
            "attr_map": attr_map,  # unsigned, (H, W)
            "attrs_signed": attrs_signed,  # signed, (H, W)
            "attrs_full": attrs.detach().cpu(),  # (1, C, H, W)
            "delta": delta_val,
        }

    def attribute_image(
        self,
        image_path: Union[str, Path],
        target_class: Optional[int] = None,
        method: str = "integrated_gradients",
        baseline_type: str = "black",
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        End-to-end: load image → predict → attribute.

        Args:
            image_path:   Path to input image
            target_class: Class to attribute toward. If None, uses predicted class.
            method:       Attribution method name
            baseline_type: Baseline type
            n_steps:      IG integration steps

        Returns:
            Dict with: inputs, prediction, attribution results
        """
        inputs = self.load_image(image_path)
        prediction = self.predict(inputs)

        if target_class is None:
            target_class = prediction["class_index"]

        attr_result = self.attribute(
            inputs,
            target_class=target_class,
            method=method,
            baseline_type=baseline_type,
            n_steps=n_steps,
        )

        return {
            "image_path": str(image_path),
            "inputs": inputs,
            "prediction": prediction,
            "attribution": attr_result,
            "target_class": target_class,
        }

    def plot_results(
        self,
        result: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 4),
    ) -> plt.Figure:
        """
        Visualize attribution results: input + heatmap + overlay.

        Args:
            result:    Output of attribute_image()
            save_path: Optional path to save the figure
            figsize:   Figure dimensions

        Returns:
            matplotlib Figure
        """
        inputs = result["inputs"]
        prediction = result["prediction"]
        attr_map = result["attribution"]["attr_map"].numpy()

        # Denormalize image for display
        img_np = denormalize_imagenet(inputs.squeeze(0))

        # Normalize attribution to [0, 1]
        attr_norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

        # Overlay
        heatmap_rgb = cm.hot(attr_norm)[:, :, :3]
        overlay = (0.6 * img_np + 0.4 * heatmap_rgb).clip(0, 1)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(img_np)
        axes[0].set_title("Input Image", fontsize=10)
        axes[0].axis("off")

        im = axes[1].imshow(attr_norm, cmap="hot")
        axes[1].set_title(
            f"Attribution Map\n({result['attribution']['method']})", fontsize=10
        )
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=10)
        axes[2].axis("off")

        delta = result["attribution"]["delta"]
        delta_str = f"δ = {delta:.5f}" if delta is not None else ""
        fig.suptitle(
            f"Prediction: {prediction['class_name']} ({prediction['confidence']:.1%})  {delta_str}",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def batch_attribute(
        self,
        image_paths: List[Union[str, Path]],
        method: str = "saliency",
        n_steps: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Compute attributions for a list of image paths.

        Args:
            image_paths: List of paths to images
            method:      Attribution method (saliency is fastest)
            n_steps:     IG steps (unused for saliency)

        Returns:
            List of result dicts from attribute_image()
        """
        results = []
        for i, path in enumerate(image_paths):
            result = self.attribute_image(path, method=method, n_steps=n_steps)
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torchvision.models import resnet18, ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.eval()

    pipeline = ImageAttributionPipeline(
        model=model,
        input_size=224,
        gradcam_layer=model.layer4[-1],
    )

    print("ImageAttributionPipeline ready.")
    print("Usage:")
    print("  result = pipeline.attribute_image('cat.jpg', method='integrated_gradients')")
    print("  pipeline.plot_results(result, save_path='attribution.png')")
    print()
    print("Methods: integrated_gradients | saliency | gradient_shap | gradcam")
    print("Baselines: black | white | blur | noise | mean")
