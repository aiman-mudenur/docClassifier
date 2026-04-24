"""
Federated Learning Server — app.py
Supports: Text classification, percentage analysis, topic modeling
Document ingestion: DOCX, PDF, spreadsheet (via client-side preprocessing)
FL: FedAvg aggregation with battery-aware / resource-aware client metadata
Deploy: Render (gunicorn)
"""

from flask import Flask, request, jsonify
import numpy as np
import json
import threading
import copy
import os
import time
import logging
from collections import defaultdict

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Load / initialise global weights ─────────────────────────────────────────
WEIGHTS_FILE = "initial_weights.json"

def load_initial_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE) as f:
            return json.load(f)
    # Auto-generate compatible weights for a 3-layer text-classifier
    # Input: 128-d TF-IDF-style embedding  →  64  →  32  →  5 classes
    log.warning("initial_weights.json not found — generating default weights.")
    layers = [
        np.zeros((128, 64)).tolist(),   # W1
        np.zeros(64).tolist(),          # b1
        np.zeros((64, 32)).tolist(),    # W2
        np.zeros(32).tolist(),          # b2
        np.zeros((32, 5)).tolist(),     # W3  (5 topic/class outputs)
        np.zeros(5).tolist(),           # b3
    ]
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(layers, f)
    return layers

global_weights = load_initial_weights()

# ── FL state ──────────────────────────────────────────────────────────────────
client_updates   = []          # list of dicts from POST /upload_weights
MIN_CLIENTS      = 3           # minimum before aggregation triggers
lock             = threading.Lock()
round_number     = 0
round_history    = []          # store per-round metrics
client_registry  = {}          # device_id → last-seen metadata

# ── Helpers ───────────────────────────────────────────────────────────────────

def fedavg(updates):
    """
    Weighted FedAvg: weight each client's contribution by num_samples.
    Falls back to simple mean if num_samples missing.
    """
    total_samples = sum(u.get("num_samples", 1) for u in updates)
    n_layers = len(updates[0]["weights"])
    averaged = []
    for layer_idx in range(n_layers):
        layer_stack = []
        for u in updates:
            w    = np.array(u["weights"][layer_idx], dtype=np.float32)
            frac = u.get("num_samples", 1) / total_samples
            layer_stack.append(w * frac)
        averaged.append(np.sum(layer_stack, axis=0).tolist())
    return averaged


def resource_score(meta: dict) -> float:
    """
    Battery-aware inclusion score (0-1).
    Clients that trained on low battery get down-weighted slightly
    so their possibly-truncated local epoch contributes less.
    battery_level : 0-100 (int)
    is_charging   : bool
    """
    battery  = meta.get("battery_level", 100)
    charging = meta.get("is_charging", True)
    score    = battery / 100.0
    if charging:
        score = min(1.0, score + 0.15)   # boost charging devices
    return round(score, 3)


def topic_distribution(updates):
    """
    Aggregate topic-distribution histograms sent by clients.
    Each client sends topic_counts: {topic_id: count, ...}
    Returns merged percentage breakdown.
    """
    merged = defaultdict(int)
    for u in updates:
        for tid, cnt in u.get("topic_counts", {}).items():
            merged[tid] += cnt
    total = sum(merged.values()) or 1
    return {k: round(v / total * 100, 2) for k, v in merged.items()}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "round": round_number})


@app.route("/status", methods=["GET"])
def status():
    with lock:
        waiting = len(client_updates)
    return jsonify({
        "round":           round_number,
        "clients_waiting": waiting,
        "min_clients":     MIN_CLIENTS,
        "registered":      len(client_registry),
        "history":         round_history[-5:],   # last 5 rounds
    })


@app.route("/get_global_model", methods=["GET"])
def get_global_model():
    """
    Clients call this on startup or after each round to pull latest weights.
    """
    return jsonify({
        "weights": global_weights,
        "round":   round_number,
        "layers":  len(global_weights),
    })


@app.route("/register", methods=["POST"])
def register():
    """
    Optional: clients register device metadata before training.
    Body: { device_id, battery_level, is_charging, model_version }
    """
    data      = request.get_json(force=True)
    device_id = data.get("device_id", "unknown")
    with lock:
        client_registry[device_id] = {
            **data,
            "registered_at": time.time()
        }
    log.info(f"Registered device: {device_id}")
    return jsonify({"status": "registered", "device_id": device_id})


@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    """
    Clients POST their locally-trained weight deltas after local training.

    Expected body:
    {
        "device_id":     "android-xxxx",
        "weights":       [[...], [...], ...],   // same shape as global_weights
        "num_samples":   420,
        "battery_level": 78,
        "is_charging":   false,
        "topic_counts":  {"0": 120, "1": 85, "2": 215},
        "local_loss":    0.312,
        "doc_types":     ["pdf", "docx"],
        "round":         3                       // client's last known round
    }
    """
    global global_weights, client_updates, round_number

    data      = request.get_json(force=True)
    device_id = data.get("device_id", "unknown")
    weights   = data.get("weights")
    client_round = data.get("round", -1)

    # Basic validation
    if not weights or not isinstance(weights, list):
        return jsonify({"status": "error", "message": "Missing or invalid weights"}), 400

    if len(weights) != len(global_weights):
        return jsonify({
            "status":  "error",
            "message": f"Layer count mismatch: got {len(weights)}, expected {len(global_weights)}"
        }), 400

    # Stale-round guard: ignore updates from 2+ rounds behind
    if client_round >= 0 and round_number - client_round > 2:
        return jsonify({
            "status":  "stale",
            "message": f"Update is from round {client_round}; server is on {round_number}. Re-pull model.",
            "round":   round_number
        }), 409

    resource = resource_score(data)
    log.info(f"[{device_id}] weights received | samples={data.get('num_samples',1)} "
             f"battery={data.get('battery_level','?')}% resource_score={resource}")

    with lock:
        # Deduplicate: one upload per device per round
        existing_ids = {u["device_id"] for u in client_updates}
        if device_id in existing_ids:
            return jsonify({"status": "duplicate", "message": "Already received update from this device this round"}), 200

        client_updates.append({**data, "resource_score": resource})
        collected = len(client_updates)

        if collected >= MIN_CLIENTS:
            log.info(f"Aggregating {collected} client updates → Round {round_number + 1}")

            topics = topic_distribution(client_updates)
            avg_loss = float(np.mean([u.get("local_loss", 0) for u in client_updates]))
            doc_type_counts = defaultdict(int)
            for u in client_updates:
                for dt in u.get("doc_types", []):
                    doc_type_counts[dt] += 1

            global_weights = fedavg(client_updates)

            round_number += 1
            round_history.append({
                "round":           round_number,
                "clients":         collected,
                "avg_loss":        round(avg_loss, 4),
                "topic_pct":       topics,
                "doc_types":       dict(doc_type_counts),
                "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            client_updates = []   # reset for next round

            log.info(f"Round {round_number} complete! avg_loss={avg_loss:.4f} topics={topics}")

            return jsonify({
                "status":    "aggregated",
                "round":     round_number,
                "avg_loss":  avg_loss,
                "topic_pct": topics,
                "doc_types": dict(doc_type_counts),
                "message":   f"Round {round_number} complete — global model updated."
            })

    return jsonify({
        "status":    "waiting",
        "collected": collected,
        "needed":    MIN_CLIENTS,
        "round":     round_number,
    })


@app.route("/round_history", methods=["GET"])
def get_round_history():
    return jsonify({"history": round_history})


@app.route("/analytics", methods=["GET"])
def analytics():
    """
    Aggregated analytics across all completed rounds.
    """
    if not round_history:
        return jsonify({"message": "No rounds completed yet."})

    all_topics = defaultdict(float)
    all_doc_types = defaultdict(int)
    losses = []

    for r in round_history:
        losses.append(r.get("avg_loss", 0))
        for tid, pct in r.get("topic_pct", {}).items():
            all_topics[tid] += pct
        for dt, cnt in r.get("doc_types", {}).items():
            all_doc_types[dt] += cnt

    n = len(round_history)
    avg_topics = {k: round(v / n, 2) for k, v in all_topics.items()}

    return jsonify({
        "total_rounds":      round_number,
        "avg_topic_pct":     avg_topics,
        "doc_type_totals":   dict(all_doc_types),
        "loss_trend":        losses,
        "registered_clients": len(client_registry),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
