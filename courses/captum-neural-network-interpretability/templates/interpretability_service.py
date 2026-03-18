"""
Captum Neural Network Interpretability
Production FastAPI interpretability service template.

Usage:
    pip install fastapi uvicorn captum torch torchvision
    uvicorn interpretability_service:app --host 0.0.0.0 --port 8080

Endpoints:
    POST /attribute          — single-example attribution
    POST /attribute/batch    — concurrent batch attribution
    GET  /health             — liveness probe
    GET  /models             — list registered models
    GET  /cache/stats        — cache statistics
    DELETE /cache            — clear cache

Configuration:
    Register models in the startup lifespan function below.
    Each model needs a RegisteredModel entry in the registry.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

from captum.attr import (
    GradientShap,
    IntegratedGradients,
    LayerGradCam,
    Saliency,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RegisteredModel:
    """Wraps a model with its metadata for the registry."""

    model: torch.nn.Module
    input_shape: tuple  # (C, H, W) for images, (n_features,) for tabular
    n_classes: int
    class_names: Optional[List[str]] = None
    device: str = "cpu"
    attribution_layer: Optional[Any] = None  # for GradCAM


class AttributionRequest(BaseModel):
    """Schema for a single attribution request."""

    model_id: str
    method: Literal[
        "integrated_gradients",
        "gradient_shap",
        "saliency",
        "gradcam",
    ]
    inputs: List[float]  # flattened input tensor
    input_shape: List[int]  # shape to reconstruct
    target: int
    baseline: Literal["zero", "mean", "noise"] = "zero"
    n_steps: int = 50
    return_delta: bool = False

    @validator("n_steps")
    def validate_n_steps(cls, v: int) -> int:
        if not 10 <= v <= 500:
            raise ValueError("n_steps must be between 10 and 500")
        return v

    @validator("inputs")
    def validate_inputs_not_empty(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ValueError("inputs must not be empty")
        return v


class BatchAttributionRequest(BaseModel):
    """Schema for a batch of attribution requests."""

    requests: List[AttributionRequest]
    max_parallel: int = 4

    @validator("max_parallel")
    def validate_max_parallel(cls, v: int) -> int:
        if not 1 <= v <= 16:
            raise ValueError("max_parallel must be between 1 and 16")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegistry:
    """Thread-safe model registry with lazy loading."""

    def __init__(self) -> None:
        self._models: Dict[str, RegisteredModel] = {}
        self._lock = threading.Lock()

    def register(self, model_id: str, registered: RegisteredModel) -> None:
        with self._lock:
            self._models[model_id] = registered
        logger.info(f"Registered model: {model_id}")

    def get(self, model_id: str) -> RegisteredModel:
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not registered")
            return self._models[model_id]

    def list_models(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())


registry = ModelRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTION CACHE
# ─────────────────────────────────────────────────────────────────────────────


class AttributionCache:
    """LRU in-memory attribution cache keyed by input hash."""

    def __init__(self, max_size: int = 512) -> None:
        self._store: Dict[str, dict] = {}
        self._order: List[str] = []
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def _key(
        self,
        model_id: str,
        method: str,
        inputs_np: np.ndarray,
        target: int,
        baseline_type: str,
        n_steps: int,
    ) -> str:
        payload = {
            "model_id": model_id,
            "method": method,
            "inputs_hash": hashlib.md5(inputs_np.tobytes()).hexdigest(),
            "target": target,
            "baseline": baseline_type,
            "n_steps": n_steps,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            if key in self._store:
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            if len(self._order) >= self._max_size:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)
            self._store[key] = value
            self._order.append(key)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._order.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self._hits / total:.1%}" if total > 0 else "N/A",
            }


attribution_cache = AttributionCache(max_size=512)


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────


def build_baseline(
    inputs: torch.Tensor,
    baseline_type: str,
) -> torch.Tensor:
    """Construct a baseline tensor matching the input shape."""
    if baseline_type == "zero":
        return torch.zeros_like(inputs)
    elif baseline_type == "mean":
        return inputs.mean(dim=0, keepdim=True).expand_as(inputs)
    elif baseline_type == "noise":
        return torch.randn_like(inputs) * 0.1
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTION COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────


def compute_attribution(
    model: torch.nn.Module,
    method: str,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    target: int,
    n_steps: int,
    return_delta: bool,
    attribution_layer: Optional[Any] = None,
) -> dict:
    """Dispatch to the correct Captum attribution method."""

    def forward(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    if method == "integrated_gradients":
        ig = IntegratedGradients(forward)
        result = ig.attribute(
            inputs,
            baseline,
            target=target,
            n_steps=n_steps,
            return_convergence_delta=return_delta,
        )

    elif method == "gradient_shap":
        gs = GradientShap(forward)
        bg = baseline.expand(8, *baseline.shape[1:])
        result = gs.attribute(
            inputs,
            bg,
            n_samples=50,
            target=target,
            return_convergence_delta=return_delta,
        )

    elif method == "saliency":
        sal = Saliency(forward)
        attrs = sal.attribute(inputs, target=target, abs=False)
        result = (attrs, torch.tensor(0.0)) if return_delta else attrs

    elif method == "gradcam":
        if attribution_layer is None:
            raise ValueError(
                "GradCAM requires 'attribution_layer' to be set in RegisteredModel"
            )
        gc = LayerGradCam(forward, attribution_layer)
        attrs = gc.attribute(inputs, target=target)
        result = (attrs, torch.tensor(0.0)) if return_delta else attrs

    else:
        raise ValueError(f"Unsupported method: {method!r}")

    if return_delta and isinstance(result, tuple):
        attrs, delta = result
        return {"attributions": attrs, "delta": float(delta.item())}
    else:
        if isinstance(result, tuple):
            attrs = result[0]
        else:
            attrs = result
        return {"attributions": attrs, "delta": None}


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup and clean up at shutdown."""
    # ── STARTUP ──────────────────────────────────────────────────────────────
    logger.info("Starting interpretability service...")

    # ── Register your models here ────────────────────────────────────────────
    # Example: ResNet-18 for ImageNet
    #
    # from torchvision.models import resnet18, ResNet18_Weights
    # weights = ResNet18_Weights.IMAGENET1K_V1
    # resnet = resnet18(weights=weights).eval()
    # with open("imagenet_classes.txt") as f:
    #     imagenet_classes = [l.strip() for l in f]
    # registry.register("resnet18_imagenet", RegisteredModel(
    #     model=resnet,
    #     input_shape=(3, 224, 224),
    #     n_classes=1000,
    #     class_names=imagenet_classes,
    #     device="cpu",
    #     attribution_layer=resnet.layer4[-1],
    # ))
    #
    # Example: BERT sentiment
    #
    # from transformers import AutoModelForSequenceClassification
    # bert = AutoModelForSequenceClassification.from_pretrained(
    #     "textattack/bert-base-uncased-SST-2").eval()
    # registry.register("bert_sentiment", RegisteredModel(
    #     model=bert,
    #     input_shape=(128,),  # max sequence length
    #     n_classes=2,
    #     class_names=["NEGATIVE", "POSITIVE"],
    # ))

    logger.info(f"Registered models: {registry.list_models()}")
    yield
    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("Shutting down interpretability service.")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Captum Interpretability Service",
    version="1.0.0",
    description="RESTful API for neural network attribution using Captum",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Liveness probe for infrastructure."""
    return {"status": "ok", "models": registry.list_models()}


@app.get("/models")
async def list_models():
    """List all registered models."""
    return {"models": registry.list_models()}


@app.get("/cache/stats")
async def cache_stats():
    """Attribution cache statistics."""
    return attribution_cache.stats()


@app.delete("/cache")
async def clear_cache():
    """Clear the attribution cache."""
    attribution_cache.clear()
    return {"status": "cleared"}


@app.post("/attribute")
async def attribute(request: AttributionRequest):
    """
    Compute attribution for a single example.

    Returns attribution scores in the same shape as the input,
    along with prediction details and optional convergence delta.
    """
    t_start = time.perf_counter()

    # 1. Resolve model
    try:
        registered = registry.get(request.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    device = registered.device
    model = registered.model.to(device).eval()

    # 2. Reconstruct input tensor
    try:
        inputs_np = np.array(request.inputs, dtype=np.float32).reshape(
            request.input_shape
        )
        inputs_t = torch.from_numpy(inputs_np).unsqueeze(0).to(device)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Input reconstruction failed: {exc}"
        )

    # 3. Cache lookup
    cache_key = attribution_cache._key(
        request.model_id,
        request.method,
        inputs_np,
        request.target,
        request.baseline,
        request.n_steps,
    )
    cached = attribution_cache.get(cache_key)
    if cached is not None:
        return {**cached, "metadata": {**cached["metadata"], "cache_hit": True}}

    # 4. Model prediction
    with torch.no_grad():
        logits = model(inputs_t)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(probs.argmax().item())
        confidence = float(probs[pred_class].item())

    # 5. Build baseline and compute attribution
    baseline_t = build_baseline(inputs_t, request.baseline).to(device)
    try:
        attr_result = compute_attribution(
            model,
            request.method,
            inputs_t,
            baseline_t,
            request.target,
            request.n_steps,
            request.return_delta,
            attribution_layer=registered.attribution_layer,
        )
    except Exception as exc:
        logger.exception("Attribution computation failed")
        raise HTTPException(
            status_code=500, detail=f"Attribution failed: {exc}"
        )

    # 6. Serialize
    attrs_np = attr_result["attributions"].squeeze(0).detach().cpu().numpy()
    class_name = (
        registered.class_names[pred_class]
        if registered.class_names and pred_class < len(registered.class_names)
        else str(pred_class)
    )

    t_end = time.perf_counter()
    response = {
        "request_id": str(uuid.uuid4()),
        "model_id": request.model_id,
        "method": request.method,
        "attributions": attrs_np.tolist(),
        "attribution_shape": list(attrs_np.shape),
        "delta": attr_result["delta"],
        "prediction": {
            "class_index": pred_class,
            "class_name": class_name,
            "confidence": round(confidence, 4),
        },
        "metadata": {
            "n_steps": request.n_steps,
            "baseline": request.baseline,
            "compute_time_ms": round((t_end - t_start) * 1000, 1),
            "cache_hit": False,
        },
    }

    attribution_cache.set(cache_key, response)
    return response


@app.post("/attribute/batch")
async def attribute_batch(batch_request: BatchAttributionRequest):
    """
    Compute attributions for multiple examples concurrently.

    Uses asyncio.gather with a semaphore to limit concurrent computations.
    Failed requests return error messages without cancelling the batch.
    """
    semaphore = asyncio.Semaphore(batch_request.max_parallel)

    async def bounded_attribute(req: AttributionRequest) -> dict:
        async def _call():
            return await attribute(req)

        async with semaphore:
            return await _call()

    results = await asyncio.gather(
        *[bounded_attribute(r) for r in batch_request.requests],
        return_exceptions=True,
    )

    processed = [
        r if not isinstance(r, Exception) else {"error": str(r), "request": None}
        for r in results
    ]
    error_count = sum(1 for r in results if isinstance(r, Exception))

    return {
        "results": processed,
        "total": len(processed),
        "errors": error_count,
        "success": len(processed) - error_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "interpretability_service:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=False,
    )
