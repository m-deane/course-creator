"""
Recipe: GradCAM on any CNN model.

Pattern: identify the last convolutional layer, apply LayerGradCam,
upsample to input resolution, and overlay on the original image.

Copy-paste ready. Requires: captum, torch, torchvision, matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from captum.attr import LayerGradCam


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECT LAST CONVOLUTIONAL LAYER
# ─────────────────────────────────────────────────────────────────────────────

def find_last_conv_layer(model):
    """
    Traverse model modules in reverse to find the last Conv2d layer.

    Returns:
        The last torch.nn.Conv2d module found, or None.

    Usage:
        layer = find_last_conv_layer(model)
        print(f"GradCAM target layer: {layer}")
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def get_layer_by_name(model, layer_name):
    """
    Access a named submodule (e.g., 'layer4.1.conv2').

    Usage:
        layer = get_layer_by_name(model, 'layer4.1.conv2')
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise AttributeError(f"Layer '{layer_name}' not found in model")


# ─────────────────────────────────────────────────────────────────────────────
# CORE GRADCAM FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam(model, inputs, target_class, layer=None):
    """
    Compute GradCAM for a CNN model.

    Args:
        model:        torch.nn.Module in eval mode
        inputs:       (1, C, H, W) input tensor
        target_class: int — class index to attribute toward
        layer:        Conv layer to compute CAM from. Defaults to last conv layer.

    Returns:
        cam_map: (H, W) numpy array, upsampled to input resolution
        cam_raw: (1, 1, h, w) raw GradCAM output before upsampling

    Usage:
        cam, _ = compute_gradcam(model, inputs, target_class=281)
        # cam is (224, 224) by default
    """
    model.eval()

    if layer is None:
        layer = find_last_conv_layer(model)
        if layer is None:
            raise ValueError(
                "No Conv2d layer found. Pass layer= explicitly."
            )

    gc = LayerGradCam(lambda x: model(x), layer)
    cam_raw = gc.attribute(inputs, target=target_class)  # (1, 1, h, w)

    # Upsample to input spatial resolution
    H, W = inputs.shape[2], inputs.shape[3]
    cam_upsampled = torch.nn.functional.interpolate(
        cam_raw,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    # ReLU: keep only positive activations
    cam_map = cam_upsampled.squeeze().detach().clamp(min=0).numpy()

    # Normalize to [0, 1]
    if cam_map.max() > 0:
        cam_map = cam_map / cam_map.max()

    return cam_map, cam_raw


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Convert normalized tensor to displayable [0,1] numpy array."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    img = (tensor.squeeze(0) * s + m).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay_gradcam(image_np, cam_map, alpha=0.4, colormap="jet"):
    """
    Blend original image with GradCAM heatmap.

    Args:
        image_np: (H, W, 3) float array in [0, 1]
        cam_map:  (H, W) float array in [0, 1]
        alpha:    Heatmap opacity (0=image only, 1=heatmap only)
        colormap: Matplotlib colormap name

    Returns:
        (H, W, 3) float array blended image
    """
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam_map)[:, :, :3]
    return (1 - alpha) * image_np + alpha * heatmap


def plot_gradcam(inputs, cam_map, class_name="", confidence=None, save_path=None):
    """
    Three-panel GradCAM visualization: input / heatmap / overlay.

    Usage:
        cam, _ = compute_gradcam(model, inputs, target_class=281)
        plot_gradcam(inputs, cam, class_name="tabby cat", confidence=0.94)
    """
    img_np = denormalize(inputs)
    overlay = overlay_gradcam(img_np, cam_map)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np);                     axes[0].set_title("Input Image")
    axes[1].imshow(cam_map, cmap="jet");         axes[1].set_title("GradCAM")
    axes[2].imshow(overlay.clip(0, 1));          axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    conf_str = f" ({confidence:.1%})" if confidence is not None else ""
    fig.suptitle(f"GradCAM — {class_name}{conf_str}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-LAYER GRADCAM (compare different depths)
# ─────────────────────────────────────────────────────────────────────────────

def compute_multilayer_gradcam(model, inputs, target_class, layer_dict):
    """
    Compute GradCAM from multiple layers simultaneously.

    Args:
        model:       CNN model in eval mode
        inputs:      (1, C, H, W) tensor
        target_class: int
        layer_dict:  {'layer_name': layer_module} dict

    Returns:
        Dict of {layer_name: cam_map (H, W)} for each layer

    Usage:
        layers = {
            'layer1': model.layer1[-1],
            'layer2': model.layer2[-1],
            'layer4': model.layer4[-1],
        }
        cams = compute_multilayer_gradcam(model, inputs, target_class=281, layer_dict=layers)
    """
    cams = {}
    for name, layer in layer_dict.items():
        cam_map, _ = compute_gradcam(model, inputs, target_class, layer=layer)
        cams[name] = cam_map
    return cams


def plot_multilayer_gradcam(inputs, cams_dict, class_name="", save_path=None):
    """
    Visualize GradCAM from multiple layers side by side.
    """
    img_np = denormalize(inputs)
    n = len(cams_dict) + 1

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    for i, (name, cam) in enumerate(cams_dict.items(), start=1):
        overlay = overlay_gradcam(img_np, cam)
        axes[i].imshow(overlay.clip(0, 1))
        axes[i].set_title(f"GradCAM: {name}")
        axes[i].axis("off")

    fig.suptitle(f"Multi-Layer GradCAM — {class_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    inputs = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        probs = torch.softmax(model(inputs), dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    print(f"Predicted class: {pred_class} ({confidence:.1%})")

    # Single-layer GradCAM
    cam, _ = compute_gradcam(model, inputs, target_class=pred_class, layer=model.layer4[-1])
    print(f"GradCAM shape: {cam.shape}")
    fig = plot_gradcam(inputs, cam, class_name=f"class_{pred_class}", confidence=confidence)
    fig.savefig("gradcam_demo.png", dpi=120, bbox_inches="tight")
    print("Saved: gradcam_demo.png")

    # Multi-layer GradCAM
    layer_dict = {
        "layer1": model.layer1[-1],
        "layer2": model.layer2[-1],
        "layer3": model.layer3[-1],
        "layer4": model.layer4[-1],
    }
    cams = compute_multilayer_gradcam(model, inputs, pred_class, layer_dict)
    fig2 = plot_multilayer_gradcam(inputs, cams, class_name=f"class_{pred_class}")
    fig2.savefig("gradcam_multilayer.png", dpi=120, bbox_inches="tight")
    print("Saved: gradcam_multilayer.png")
