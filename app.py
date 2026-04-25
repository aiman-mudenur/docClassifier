"""
Federated Learning Server — app.py
Dataset: DBpedia-14 (14-class text classification)
Model:   Dense(100→64→32→14)
Deploy:  Render (gunicorn)
"""

from flask import Flask, request, jsonify
import numpy as np
import json
import threading
import os
import time
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

CLASS_NAMES = {
    "0":  "Company",
    "1":  "EducationalInstitution",
    "2":  "Artist",
    "3":  "Athlete",
    "4":  "OfficeHolder",
    "5":  "MeanOfTransportation",
    "6":  "Building",
    "7":  "NaturalPlace",
    "8":  "Village",
    "9":  "Animal",
    "10": "Plant",
    "11": "Album",
    "12": "Film",
    "13": "WrittenWork",
}

INPUT_DIM    = 100
HIDDEN1      = 64
HIDDEN2      = 32
NUM_CLASSES  = 14
WEIGHTS_FILE = "initial_weights.json"
MIN_CLIENTS  = 3


def load_initial_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE) as f:
            weights = json.load(f)
        log.info(f"Loaded initial_weights.json — {len(weights)} layers")
        for i, w in enumerate(weights):
            log.info(f"  Layer {i}: shape={np.array(w).shape}")
        return weights

    log.warning("initial_weights.json not found — generating zero weights.")
    layers = [
        np.zeros((INPUT_DIM, HIDDEN1)).tolist(),
        np.zeros(HIDDEN1).tolist(),
        np.zeros((HIDDEN1, HIDDEN2)).tolist(),
        np.zeros(HIDDEN2).tolist(),
        np.zeros((HIDDEN2, NUM_CLASSES)).tolist(),
        np.zeros(NUM_CLASSES).tolist(),
    ]
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(layers, f)
    return layers


global_weights  = load_initial_weights()
client_updates  = []
lock            = threading.Lock()
round_number    = 0
round_history   = []
client_registry = {}


def fedavg(updates):
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
    battery  = meta.get("battery_level", 100)
    charging = meta.get("is_charging", True)
    score    = battery / 100.0
    if charging:
        score = min(1.0, score + 0.15)
    return round(score, 3)


def topic_distribution(updates):
    merged = defaultdict(int)
    for u in updates:
        for tid, cnt in u.get("topic_counts", {}).items():
            merged[tid] += cnt
    total = sum(merged.values()) or 1
    return {CLASS_NAMES.get(k, k): round(v / total * 100, 2) for k, v in merged.items()}


def _try_aggregate():
    global global_weights, client_updates, round_number
    if len(client_updates) < MIN_CLIENTS:
        return False
    log.info(f"Aggregating round {round_number} with {len(client_updates)} clients...")
    global_weights = fedavg(client_updates)
    dist = topic_distribution(client_updates)
    round_history.append({
        "round":       round_number,
        "num_clients": len(client_updates),
        "topic_dist":  dist,
        "timestamp":   time.time(),
    })
    round_number  += 1
    client_updates = []
    log.info(f"Aggregation complete. Now on round {round_number}.")
    return True


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok",
        "round":       round_number,
        "num_classes": NUM_CLASSES,
        "dataset":     "DBpedia-14",
    })


@app.route("/status", methods=["GET"])
def status():
    with lock:
        waiting = len(client_updates)
    return jsonify({
        "round":           round_number,
        "clients_waiting": waiting,
        "min_clients":     MIN_CLIENTS,
        "registered":      len(client_registry),
        "history":         round_history[-5:],
    })


@app.route("/get_global_model", methods=["GET"])
def get_global_model():
    return jsonify({
        "weights":     global_weights,
        "round":       round_number,
        "layers":      len(global_weights),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
    })


@app.route("/register", methods=["POST"])
def register():
    data      = request.get_json(force=True)
    device_id = data.get("device_id", "unknown")
    with lock:
        client_registry[device_id] = {**data, "registered_at": time.time()}
    log.info(f"Registered device: {device_id}  total={len(client_registry)}")
    return jsonify({"status": "registered", "device_id": device_id})


@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    global global_weights, client_updates, round_number

    data         = request.get_json(force=True)
    device_id    = data.get("device_id", "unknown")
    weights      = data.get("weights")
    client_round = data.get("round", -1)

    if not weights or not isinstance(weights, list):
        return jsonify({"status": "error", "message": "Missing or invalid weights"}), 400

    if len(weights) != len(global_weights):
        return jsonify({
            "status":  "error",
            "message": f"Layer count mismatch: got {len(weights)}, expected {len(global_weights)}",
        }), 400

    for i, (client_layer, global_layer) in enumerate(zip(weights, global_weights)):
        cs = np.array(client_layer).shape
        gs = np.array(global_layer).shape
        if cs != gs:
            return jsonify({
                "status":  "error",
                "message": f"Shape mismatch at layer {i}: client={cs} vs server={gs}",
            }), 400

    if client_round >= 0 and round_number - client_round > 2:
        return jsonify({
            "status":  "stale",
            "message": f"Update from round {client_round}; server is on {round_number}.",
            "round":   round_number,
        }), 409

    resource = resource_score(data)
    log.info(
        f"[{device_id}] weights received | "
        f"samples={data.get('num_samples', 1)} "
        f"battery={data.get('battery_level', '?')}% "
        f"resource_score={resource}"
    )

    aggregated = False
    with lock:
        existing_ids = {u["device_id"] for u in client_updates}
        if device_id in existing_ids:
            return jsonify({
                "status":  "duplicate",
                "message": "