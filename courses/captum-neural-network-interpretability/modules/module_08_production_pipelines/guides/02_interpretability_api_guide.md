# Building an Interpretability API with FastAPI and Captum

> **Reading time:** ~8 min | **Module:** 8 — Production Pipelines | **Prerequisites:** Modules 1-7


## Learning Objectives

By the end of this guide, you will be able to:
1. Design a RESTful API that serves Captum attributions on demand
2. Implement model registry, caching, and batched attribution in production
3. Handle input validation and error responses for attribution endpoints
4. Structure attribution responses as JSON for downstream consumers
5. Deploy an interpretability service with health checks and logging


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of building an interpretability api with fastapi and captum.

</div>

---

## 1. Why a Dedicated Interpretability API?

Captum Insights is a development tool. When you need attributions accessible to other systems — dashboards, compliance tools, monitoring pipelines — you need a proper API.

**Requirements that an API must satisfy:**
- Stateless request handling
- Input validation and sanitization
- Structured JSON responses (not images)
- Caching to avoid recomputing identical attributions
- Horizontal scalability (multiple workers)
- Health and readiness endpoints for infrastructure

**What Insights lacks:**
- Programmatic access to attribution values
- Request batching
- Authentication
- SLA guarantees
- Integration with monitoring systems

---

## 2. API Design

### Attribution Request

```

POST /attribute
Content-Type: application/json

{
  "model_id": "resnet18_imagenet",
  "method": "integrated_gradients",
  "inputs": [[...]],         // float array, model's input shape
  "target": 281,             // class index to attribute
  "baseline": "zero",        // "zero" | "mean" | "noise"
  "n_steps": 50,             // IG steps
  "return_delta": false
}
```

### Attribution Response

```json
{
  "model_id": "resnet18_imagenet",
  "method": "integrated_gradients",
  "attributions": [[...]],   // same shape as inputs
  "delta": 0.00012,
  "prediction": {
    "class_index": 281,
    "class_name": "tabby cat",
    "confidence": 0.94
  },
  "metadata": {
    "n_steps": 50,
    "baseline": "zero",
    "compute_time_ms": 312
  }
}
```

---

## 3. FastAPI Application Structure

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import torch
import numpy as np
import time
import logging
from captum.attr import (
    IntegratedGradients, GradientShap, Saliency,
    LayerGradCam, LayerIntegratedGradients
)

app = FastAPI(title="Captum Interpretability Service", version="1.0.0")
logger = logging.getLogger(__name__)
```

</div>

</div>

### Pydantic Request Model

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from typing import List, Optional, Literal

class AttributionRequest(BaseModel):
    model_id: str
    method: Literal["integrated_gradients", "gradient_shap",
                    "saliency", "gradcam", "layer_ig"]
    inputs: List[List[float]]           # serialized tensor
    input_shape: List[int]              # actual shape to reconstruct
    target: int
    baseline: Literal["zero", "mean", "noise"] = "zero"
    n_steps: int = 50
    return_delta: bool = False

    @validator("n_steps")
    def validate_n_steps(cls, v):
        if not 10 <= v <= 500:
            raise ValueError("n_steps must be between 10 and 500")
        return v
```

</div>

</div>

---

## 4. Model Registry

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RegisteredModel:
    model: torch.nn.Module
    input_shape: tuple          # (C, H, W) for images
    n_classes: int
    class_names: Optional[list] = None
    device: str = "cpu"
    attribution_layer: Optional[Any] = None  # for layer methods

class ModelRegistry:
    """Thread-safe model registry with lazy loading."""

    def __init__(self):
        self._models: Dict[str, RegisteredModel] = {}
        self._lock = threading.Lock()

    def register(self, model_id: str, registered: RegisteredModel):
        with self._lock:
            self._models[model_id] = registered
        logger.info(f"Registered model: {model_id}")

    def get(self, model_id: str) -> RegisteredModel:
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not registered")
            return self._models[model_id]

    def list_models(self) -> list:
        return list(self._models.keys())


# Global registry (populated at startup)
registry = ModelRegistry()
```

</div>

</div>

---

## 5. Baseline Construction

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
def build_baseline(
    inputs: torch.Tensor,
    baseline_type: str,
    background_samples: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Construct a baseline tensor matching the input shape."""
    if baseline_type == "zero":
        return torch.zeros_like(inputs)

    elif baseline_type == "mean":
        if background_samples is not None:
            # Mean over background dataset
            return background_samples.mean(dim=0, keepdim=True).expand_as(inputs)
        else:
            # Mean of the input itself (degenerate but safe)
            return inputs.mean(dim=0, keepdim=True).expand_as(inputs)

    elif baseline_type == "noise":
        # Gaussian noise baseline (mean=0, std=0.1)
        return torch.randn_like(inputs) * 0.1

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
```

</div>

</div>

---

## 6. Attribution Computation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
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
    """Dispatch to the correct Captum method."""

    def forward(x):
        return model(x)

    if method == "integrated_gradients":
        ig = IntegratedGradients(forward)
        result = ig.attribute(
            inputs, baseline, target=target, n_steps=n_steps,
            return_convergence_delta=return_delta
        )

    elif method == "gradient_shap":
        gs = GradientShap(forward)
        # GradientShap expects multiple baselines
        bg = baseline.expand(8, *baseline.shape[1:])
        result = gs.attribute(
            inputs, bg, n_samples=50, target=target,
            return_convergence_delta=return_delta
        )

    elif method == "saliency":
        sal = Saliency(forward)
        result = sal.attribute(inputs, target=target, abs=False)
        if return_delta:
            result = (result, torch.tensor(0.0))

    elif method == "gradcam":
        if attribution_layer is None:
            raise ValueError("gradcam requires attribution_layer to be registered")
        gc = LayerGradCam(forward, attribution_layer)
        result = gc.attribute(inputs, target=target)
        if return_delta:
            result = (result, torch.tensor(0.0))

    else:
        raise ValueError(f"Method '{method}' not supported")

    if return_delta and isinstance(result, tuple):
        attrs, delta = result
        return {"attributions": attrs, "delta": delta.item()}
    else:
        if isinstance(result, tuple):
            attrs = result[0]
        else:
            attrs = result
        return {"attributions": attrs, "delta": None}
```

</div>

</div>

---

## 7. Attribution Caching

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
import hashlib
import json
from functools import lru_cache

class AttributionCache:
    """Simple in-memory LRU cache for attribution results."""

    def __init__(self, max_size: int = 256):
        self._cache: Dict[str, dict] = {}
        self._keys: list = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def _cache_key(self, model_id: str, method: str,
                   inputs: np.ndarray, target: int,
                   baseline_type: str, n_steps: int) -> str:
        payload = {
            "model_id": model_id,
            "method": method,
            "inputs_hash": hashlib.md5(inputs.tobytes()).hexdigest(),
            "target": target,
            "baseline": baseline_type,
            "n_steps": n_steps,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: dict):
        with self._lock:
            if len(self._keys) >= self._max_size:
                oldest = self._keys.pop(0)
                self._cache.pop(oldest, None)
            self._cache[key] = value
            self._keys.append(key)

    def stats(self) -> dict:
        return {"size": len(self._cache), "max_size": self._max_size}


attribution_cache = AttributionCache(max_size=512)
```

</div>

</div>

---

## 8. The Attribution Endpoint

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
@app.post("/attribute")
async def attribute(request: AttributionRequest):
    t_start = time.perf_counter()

    # 1. Resolve model
    try:
        registered = registry.get(request.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model = registered.model.eval()
    device = registered.device

    # 2. Reconstruct tensor
    try:
        inputs_np = np.array(request.inputs, dtype=np.float32)
        inputs_np = inputs_np.reshape(request.input_shape)
        inputs_t = torch.from_numpy(inputs_np).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=422,
                            detail=f"Input reconstruction failed: {e}")

    # 3. Check cache
    inputs_np_flat = inputs_t.cpu().numpy()
    cache_key = attribution_cache._cache_key(
        request.model_id, request.method, inputs_np_flat,
        request.target, request.baseline, request.n_steps
    )
    cached = attribution_cache.get(cache_key)
    if cached:
        cached["metadata"]["cache_hit"] = True
        return cached

    # 4. Build baseline
    baseline_t = build_baseline(inputs_t, request.baseline)

    # 5. Run model for prediction
    with torch.no_grad():
        logits = model(inputs_t)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    # 6. Compute attribution
    try:
        attr_result = compute_attribution(
            model, request.method, inputs_t, baseline_t,
            request.target, request.n_steps, request.return_delta,
            attribution_layer=registered.attribution_layer
        )
    except Exception as e:
        logger.exception("Attribution failed")
        raise HTTPException(status_code=500,
                            detail=f"Attribution computation failed: {e}")

    # 7. Serialize
    attrs_np = attr_result["attributions"].squeeze(0).detach().cpu().numpy()

    class_name = (
        registered.class_names[pred_class]
        if registered.class_names else str(pred_class)
    )

    t_end = time.perf_counter()
    response = {
        "model_id": request.model_id,
        "method": request.method,
        "attributions": attrs_np.tolist(),
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
        }
    }

    attribution_cache.set(cache_key, response)
    return response
```

</div>

</div>

---

## 9. Health and Utility Endpoints

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
@app.get("/health")
async def health():
    return {"status": "ok", "models": registry.list_models()}

@app.get("/models")
async def list_models():
    return {"models": registry.list_models()}

@app.get("/cache/stats")
async def cache_stats():
    return attribution_cache.stats()

@app.delete("/cache")
async def clear_cache():
    attribution_cache._cache.clear()
    attribution_cache._keys.clear()
    return {"status": "cleared"}
```

</div>

</div>

---

## 10. Application Startup

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from contextlib import asynccontextmanager
from torchvision.models import resnet18, ResNet18_Weights

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.eval()

    with open("imagenet_classes.txt") as f:
        class_names = [line.strip() for line in f]

    registry.register("resnet18_imagenet", RegisteredModel(
        model=model,
        input_shape=(3, 224, 224),
        n_classes=1000,
        class_names=class_names,
        device="cpu",
        attribution_layer=model.layer4[-1],  # for GradCAM
    ))
    logger.info("Models loaded and registered")
    yield  # Application runs here
    logger.info("Shutting down")

app = FastAPI(title="Captum Interpretability Service",
              version="1.0.0", lifespan=lifespan)
```

</div>

</div>

---

## 11. Running the Service

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.sh</span>

</div>
<div class="code-body">

```bash
# Development
uvicorn interpretability_service:app --reload --port 8080

# Production (4 workers)
gunicorn interpretability_service:app \
    -w 4 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 \
    --timeout 120

# Docker
docker run -p 8080:8080 \
    -e NUM_WORKERS=4 \
    interpretability-service:latest
```

</div>

</div>

### Test with curl

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.sh</span>

</div>
<div class="code-body">

```bash
curl -X POST http://localhost:8080/attribute \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "resnet18_imagenet",
    "method": "saliency",
    "inputs": [...],
    "input_shape": [3, 224, 224],
    "target": 281,
    "n_steps": 30
  }'
```

</div>

</div>

---

## 12. Batched Attribution Endpoint

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
class BatchAttributionRequest(BaseModel):
    requests: List[AttributionRequest]
    max_parallel: int = 4

@app.post("/attribute/batch")
async def attribute_batch(batch_request: BatchAttributionRequest):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    async def run_single(req: AttributionRequest):
        return await attribute(req)

    # Run up to max_parallel attributions concurrently
    semaphore = asyncio.Semaphore(batch_request.max_parallel)
    async def bounded_attribute(req):
        async with semaphore:
            return await run_single(req)

    results = await asyncio.gather(
        *[bounded_attribute(r) for r in batch_request.requests],
        return_exceptions=True
    )

    return {
        "results": [
            r if not isinstance(r, Exception) else {"error": str(r)}
            for r in results
        ],
        "total": len(results),
        "errors": sum(1 for r in results if isinstance(r, Exception)),
    }
```

</div>

</div>

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Why a Dedicated Interpretability API?" and why it matters in practice.

2. Given a real-world scenario involving building an interpretability api with fastapi and captum, what would be your first three steps to apply the techniques from this guide?

</div>

## Summary

| Component | Responsibility |
|-----------|---------------|
| `ModelRegistry` | Thread-safe model storage, lazy loading |
| `AttributionRequest` | Pydantic validation, safe deserialization |
| `build_baseline()` | Baseline factory: zero, mean, noise |
| `compute_attribution()` | Captum method dispatch |
| `AttributionCache` | LRU in-memory cache keyed by input hash |
| `/attribute` endpoint | Single-example attribution |
| `/attribute/batch` | Concurrent multi-example attribution |
| `/health` | Liveness probe for infrastructure |

---

## Further Reading

- FastAPI documentation: https://fastapi.tiangolo.com
- Pydantic v2 validators: https://docs.pydantic.dev
- Captum attribution methods: https://captum.ai/docs/algorithms
- Uvicorn deployment: https://www.uvicorn.org/deployment/

---

## Cross-References

<a class="link-card" href="../notebooks/01_captum_insights_demo.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
