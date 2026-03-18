"""
Module 03 — Exercise 01: Layer & Neuron Attribution Self-Check

Self-check covering GradCAM theory, LayerGradCam API, Layer Conductance,
Neuron Conductance, and the completeness property at intermediate layers.

Run with: python 01_layer_attribution_self_check.py
"""

# ============================================================
# PART 1: GradCAM Theory
# ============================================================

# Q1: GradCAM computes importance weights by taking the:
#     Global average of gradients over the spatial dimensions
#     of the target convolutional feature maps.
#
# What is the shape of alpha_k for a ResNet-50 layer4 that has
# 2048 feature maps?
gradcam_alpha_shape = None  # e.g., "(2048,)" or "(2048, 7, 7)" or "(7, 7)"

# Q2: After computing the weighted sum of feature maps, GradCAM
#     applies which operation to remove negative contributions?
gradcam_nonlinearity = None  # "ReLU", "Softmax", "Sigmoid", or "Tanh"

# Q3: ResNet-50's layer4 has spatial resolution 7×7.
#     After computing GradCAM at layer4, what must you do to
#     overlay the heatmap on the 224×224 input image?
gradcam_upsample = None
# Options:
# "a" = Tile the 7×7 map 32 times in each dimension
# "b" = Upsample (e.g., bilinear interpolation) to 224×224
# "c" = Pad the 7×7 map with zeros to 224×224
# "d" = Crop the 224×224 image to 7×7

# Q4: GradCAM is class-discriminative. For the same input image,
#     computing GradCAM for class A vs class B will:
gradcam_class_discriminative = None
# Options:
# "a" = Always produce the same heatmap
# "b" = Produce different heatmaps if the model distinguishes the classes
# "c" = Produce heatmaps that sum to the same total
# "d" = Only work for the top-1 predicted class

# Q5: In Captum, which class implements GradCAM?
captum_gradcam_class = None  # "GradCAM", "LayerGradCam", "FeatureGradCam", "ClassActivationMap"

# Q6: In Captum's LayerGradCam, which parameter controls
#     whether the internal ReLU is applied?
gradcam_relu_param = None  # "apply_relu", "relu_attributions", "use_relu", "remove_relu"

# Q7: Guided GradCAM = (Guided Backpropagation) × (upsampled GradCAM).
#     Guided Backpropagation fails which interpretability axiom?
guided_gbp_fails = None  # "Completeness", "Sensitivity", "Implementation Invariance", "Linearity"

# ============================================================
# PART 2: Layer Conductance
# ============================================================

# Q8: Layer Conductance is mathematically equivalent to:
layer_conductance_equiv = None
# Options:
# "a" = GradCAM applied to a hidden layer
# "b" = Integrated Gradients applied at an intermediate layer
# "c" = The gradient of the output w.r.t. the hidden layer (no integration)
# "d" = The activation value at the hidden layer

# Q9: The completeness property of Layer Conductance states:
#     sum_i(Cond_i^l) = f(x) - f(x')
#
# This holds for:
layer_completeness_holds = None
# Options:
# "a" = Only the final layer
# "b" = Only layers with convolutional structure
# "c" = Every layer in the network
# "d" = Only layers after global average pooling

# Q10: For ResNet-50, model.layer4[-1] refers to:
resnet_layer4_last = None
# Options:
# "a" = The first Bottleneck block in ResNet-50's fourth residual stage
# "b" = The last Bottleneck block in ResNet-50's fourth residual stage
# "c" = The final fully connected layer of ResNet-50
# "d" = The global average pooling layer

# Q11: You compute Layer Conductance at layer3 and layer4 for the same prediction.
#      layer3_sum = 0.62, layer4_sum = 0.63, f(x) - f(baseline) = 0.65.
#      What can you conclude?
layer_conductance_comparison = None
# Options:
# "a" = Layer4 contributes more because its sum is closer to 0.65
# "b" = Both layers are approximately complete; which "contributes more"
#       is answered by absolute conductance, not signed sum
# "c" = There is a bug — the sums should be identical
# "d" = Layer3 contributes more because it comes before layer4

# Q12: The LayerConductance output for ResNet-50 layer3 has shape:
layer_cond_shape = None  # "(1, 256, 14, 14)", "(1, 1024, 14, 14)",
                          # "(1, 2048, 7, 7)", or "(1, 1000)"

# ============================================================
# PART 3: Neuron Conductance
# ============================================================

# Q13: Neuron Conductance for neuron (channel=42, h=3, w=3) in layer4
#      returns a tensor of shape:
neuron_conductance_output_shape = None
# Options:
# "a" = (1,) — a single scalar
# "b" = (1, 3, 224, 224) — input attribution for that neuron
# "c" = (1, 2048, 7, 7) — full layer attribution
# "d" = (3, 3) — the spatial neighborhood of the neuron

# Q14: Neuron Conductance answers the question:
neuron_conductance_question = None
# Options:
# "a" = "Which layer in the network is most important?"
# "b" = "Which input features caused this specific neuron to activate?"
# "c" = "How many neurons are active in this layer?"
# "d" = "What is the gradient magnitude at this neuron?"

# Q15: In Captum's NeuronConductance, the neuron_selector parameter for
#      a convolutional layer neuron at channel 5, row 2, column 4 is:
neuron_selector_example = None  # (5, 2, 4), [5, 2, 4], 5, or (2, 4)

# Q16: You apply Neuron Conductance to 10 different neurons in layer4.
#      Neurons 1-8 show diffuse attribution maps covering the whole image.
#      Neuron 9 shows a concentrated map on the dog's ears.
#      Neuron 10 shows a concentrated map on the background sky.
#      What does Neuron 9 most likely represent?
neuron_9_represents = None
# Options:
# "a" = A class-agnostic edge detector
# "b" = A feature detector specialized for dog ear shapes
# "c" = A corrupted neuron with random attribution
# "d" = A neuron responding to the blue color of the sky

# ============================================================
# PART 4: API and Practice
# ============================================================

# Q17: The correct Captum API call for LayerGradCam on ResNet-50's
#      last convolutional layer is:
#      lg = LayerGradCam(model, ???)
#      which of these fills in ???
gradcam_target_layer = None
# Options:
# "a" = model.fc              (fully connected layer)
# "b" = model.layer4[-1]      (last Bottleneck in layer4)
# "c" = model.avgpool         (global average pooling)
# "d" = model.layer4[0]       (first Bottleneck in layer4)

# Q18: After getting LayerGradCam output of shape (1, 2048, 7, 7),
#      to produce a single 2D heatmap, you should:
gradcam_aggregation = None
# Options:
# "a" = Take the first channel: attr[0, 0, :, :]
# "b" = Sum across channels: attr.sum(dim=1)
# "c" = Average across the batch dimension: attr.mean(dim=0)
# "d" = Flatten to a 1D vector

# Q19: LayerAttribution.interpolate is called with interpolate_dims=(224, 224).
#      What does this parameter specify?
interpolate_dims_meaning = None
# Options:
# "a" = The number of interpolation steps
# "b" = The target output spatial size (height, width)
# "c" = The interpolation kernel size
# "d" = The input spatial size to verify

# Q20: Which of these produces the correct Captum NeuronConductance call?
correct_neuron_conductance_call = None
# Options:
# "a" = nc.attribute(input, neuron_selector=(42, 3, 3), target=cls, baselines=bl, n_steps=50)
# "b" = nc.attribute(input, neuron_index=42, spatial=(3,3), target=cls, n_steps=50)
# "c" = nc.attribute(input, channel=42, h=3, w=3, target=cls, n_steps=50)
# "d" = nc.attribute(input, layer=model.layer4, neuron=(42,3,3), target=cls)

# ============================================================
# PART 5: Interpretation and Debugging
# ============================================================

# Q21: You compute LayerGradCam on ResNet-50 and get a completely
#      uniform heatmap (all the same value). The most likely cause is:
uniform_heatmap_cause = None
# Options:
# "a" = The model is perfectly calibrated
# "b" = The wrong target layer was selected (e.g., a fully connected layer)
# "c" = n_steps is too high
# "d" = The image has no high-frequency content

# Q22: You compare GradCAM heatmaps for "golden retriever" and
#      "tabby cat" on the same dog image. They have a spatial IoU
#      of 0.85 (very high overlap). This suggests:
high_iou_interpretation = None
# Options:
# "a" = The model is extremely well-calibrated
# "b" = The model may be using background or texture features
#       rather than class-specific object features
# "c" = GradCAM is not working correctly for this model
# "d" = The dog and cat share all visual features

# Q23: Layer Conductance for 3 images returns these signed sums for layer4:
#      Image A: 0.72, Image B: -0.03, Image C: 0.61
#      f(x)-f(baseline) for each: 0.73, 0.45, 0.62
#      Which image's conductance has poor convergence quality?
poor_convergence_image = None  # "A", "B", or "C"

# Q24: True or False:
#      Layer Conductance requires the same baseline considerations
#      as Integrated Gradients (e.g., zero baseline for images).
layer_cond_needs_baseline = None  # "true" or "false"


# ============================================================
# SELF-CHECK ENGINE
# ============================================================

def check(name, got, expected, hint):
    if got is None:
        print(f"[ TODO ] {name}")
        return False
    if isinstance(expected, list):
        passed = got in expected
    elif isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        passed = abs(got - expected) < 1e-6
    else:
        passed = got == expected
    status = "[ PASS ]" if passed else "[ FAIL ]"
    print(f"{status} {name}")
    if not passed:
        print(f"         Got:      {got!r}")
        print(f"         Expected: {expected!r}")
        print(f"         Hint: {hint}")
    return passed


def run():
    results = []
    print("=" * 65)
    print("PART 1: GradCAM Theory")
    print("=" * 65)

    results.append(check(
        "GradCAM alpha shape",
        gradcam_alpha_shape, "(2048,)",
        "Alpha_k is one scalar per feature map k. For 2048 feature maps, "
        "shape is (2048,). The spatial dimensions are pooled out by the average."
    ))
    results.append(check(
        "GradCAM nonlinearity",
        gradcam_nonlinearity, "ReLU",
        "ReLU removes negative contributions (regions that suppress the class). "
        "Formula: ReLU(sum_k alpha_k * A^k)."
    ))
    results.append(check(
        "GradCAM upsampling step",
        gradcam_upsample, "b",
        "Bilinear interpolation from 7×7 to 224×224 is the standard approach. "
        "Tiling or padding would not produce smooth heatmaps."
    ))
    results.append(check(
        "GradCAM class discriminativeness",
        gradcam_class_discriminative, "b",
        "Different classes have different gradient directions in the feature maps, "
        "producing different heatmaps for the same input."
    ))
    results.append(check(
        "Captum GradCAM class name",
        captum_gradcam_class, "LayerGradCam",
        "Captum uses LayerGradCam (with camelCase) — it targets a specific layer."
    ))
    results.append(check(
        "GradCAM ReLU parameter",
        gradcam_relu_param, "relu_attributions",
        "lg.attribute(..., relu_attributions=False) gives signed heatmap."
    ))
    results.append(check(
        "Guided GBP axiom failure",
        guided_gbp_fails, "Implementation Invariance",
        "Guided Backpropagation modifies the backward pass based on architecture "
        "(clamping negative gradients at ReLU), violating implementation invariance."
    ))

    print("\n" + "=" * 65)
    print("PART 2: Layer Conductance")
    print("=" * 65)

    results.append(check(
        "Layer Conductance mathematical equivalence",
        layer_conductance_equiv, "b",
        "Layer Conductance is IG (Integrated Gradients) applied at an intermediate "
        "layer rather than the input. It integrates gradients along the path from "
        "baseline to input, but measures changes at the hidden layer."
    ))
    results.append(check(
        "Layer completeness: which layers?",
        layer_completeness_holds, "c",
        "The completeness property holds for EVERY layer independently. "
        "This is the mathematical foundation of Layer Conductance."
    ))
    results.append(check(
        "ResNet-50 layer4[-1] meaning",
        resnet_layer4_last, "b",
        "model.layer4[-1] = last Bottleneck in the fourth residual stage. "
        "This is the layer just before global average pooling."
    ))
    results.append(check(
        "Layer conductance comparison interpretation",
        layer_conductance_comparison, "b",
        "Both signed sums should be approximately equal to f(x)-f(baseline). "
        "The 'which layer contributes more' question is answered by absolute "
        "conductance (how much activity), not signed sum (which approaches total_diff)."
    ))
    results.append(check(
        "LayerConductance output shape for layer3",
        layer_cond_shape, "(1, 1024, 14, 14)",
        "ResNet-50 layer3 has 1024 channels (not 256 or 2048) and "
        "14×14 spatial resolution. layer1=56×56, layer2=28×28, "
        "layer3=14×14, layer4=7×7."
    ))

    print("\n" + "=" * 65)
    print("PART 3: Neuron Conductance")
    print("=" * 65)

    results.append(check(
        "NeuronConductance output shape",
        neuron_conductance_output_shape, "b",
        "NeuronConductance returns INPUT attribution: which input features activated "
        "this neuron. Shape = (1, 3, 224, 224) — same as the input."
    ))
    results.append(check(
        "NeuronConductance question",
        neuron_conductance_question, "b",
        "NeuronConductance answers: 'which input features caused this specific "
        "neuron to activate as it did?' It's input attribution from a neuron's perspective."
    ))
    results.append(check(
        "neuron_selector format for ch=5, h=2, w=4",
        neuron_selector_example, (5, 2, 4),
        "neuron_selector takes a tuple (channel, height, width) for conv layers."
    ))
    results.append(check(
        "Neuron 9 interpretation (concentrated on dog ears)",
        neuron_9_represents, "b",
        "A neuron with Neuron Conductance concentrated on dog ears is a feature "
        "detector specialized for dog ear shapes — a class-specific detector."
    ))

    print("\n" + "=" * 65)
    print("PART 4: API and Practice")
    print("=" * 65)

    results.append(check(
        "Correct LayerGradCam target layer",
        gradcam_target_layer, "b",
        "model.layer4[-1] is the last Bottleneck in layer4 — the last spatial "
        "conv before global avg pooling. model.fc has no spatial structure."
    ))
    results.append(check(
        "GradCAM aggregation for 2D heatmap",
        gradcam_aggregation, "b",
        "attr.sum(dim=1) sums across the channel dimension (2048 channels) "
        "to produce a single 2D (7×7) heatmap."
    ))
    results.append(check(
        "interpolate_dims meaning",
        interpolate_dims_meaning, "b",
        "interpolate_dims=(224, 224) specifies the target output resolution "
        "(height, width). The 7×7 feature map will be upsampled to 224×224."
    ))
    results.append(check(
        "Correct NeuronConductance call",
        correct_neuron_conductance_call, "a",
        "nc.attribute(input, neuron_selector=(42, 3, 3), target=cls, "
        "baselines=bl, n_steps=50) is the correct signature."
    ))

    print("\n" + "=" * 65)
    print("PART 5: Interpretation and Debugging")
    print("=" * 65)

    results.append(check(
        "Uniform heatmap cause",
        uniform_heatmap_cause, "b",
        "A uniform heatmap almost always means the target layer has no spatial "
        "structure (e.g., a fully connected layer or global pooling output). "
        "Always verify the target layer outputs a 4D tensor."
    ))
    results.append(check(
        "High IoU (0.85) between dog/cat heatmaps interpretation",
        high_iou_interpretation, "b",
        "If dog and cat heatmaps are nearly identical on a dog image, "
        "the model is likely using background or texture shortcuts rather "
        "than class-specific object features — a spurious correlation warning."
    ))
    results.append(check(
        "Poor convergence image (signed sum far from f(x)-f(x'))",
        poor_convergence_image, "B",
        "Image B: layer4 sum = -0.03, but f(x)-f(x') = 0.45. "
        "The error is 0.48, which is very large. "
        "Images A and C are close to their respective total_diffs."
    ))
    results.append(check(
        "Layer Conductance needs baseline (true/false)",
        layer_cond_needs_baseline, "true",
        "Layer Conductance integrates along the same path x' → x as IG. "
        "It requires the same baseline: zero for images, PAD tokens for text, "
        "training mean for tabular data."
    ))

    print("\n" + "=" * 65)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"RESULTS: {passed}/{total} checks passed")
    if passed == total:
        print("All checks passed! You have mastered layer and neuron attribution.")
    else:
        print(f"{total - passed} checks need review.")
    print("=" * 65)


if __name__ == "__main__":
    run()
