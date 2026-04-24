"""
generate_weights.py
-------------------
Run ONCE locally before pushing to GitHub:
    python generate_weights.py

Produces initial_weights.json — commit this file alongside app.py.

Architecture (must match your model.tflite exactly):
    Input  : 128-d float32 vector  (TF-IDF / averaged word-embedding)
    Layer 1: Dense(64, relu)
    Layer 2: Dense(32, relu)
    Layer 3: Dense(5,  softmax)   ← 5 topics / classes

If your .tflite has a different shape, edit LAYER_SHAPES below.
"""

import json
import numpy as np

# ── Edit these to match YOUR model.tflite layer shapes ───────────────────────
LAYER_SHAPES = [
    (128, 64),   # W1
    (64,),       # b1
    (64, 32),    # W2
    (32,),       # b2
    (32, 5),     # W3
    (5,),        # b3
]
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
weights = []
for shape in LAYER_SHAPES:
    if len(shape) == 2:
        # Xavier uniform initialisation for weight matrices
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w = np.random.uniform(-limit, limit, shape).astype(np.float32)
    else:
        w = np.zeros(shape, dtype=np.float32)
    weights.append(w.tolist())

with open("initial_weights.json", "w") as f:
    json.dump(weights, f)

print("✅  initial_weights.json written.")
print(f"   {len(weights)} layers:")
for i, (shape, w) in enumerate(zip(LAYER_SHAPES, weights)):
    print(f"   Layer {i}: shape={shape}  min={np.min(w):.4f}  max={np.max(w):.4f}")
