"""
Federated Learning Server — app.py
Dataset: DBpedia-14 (14-class text classification)
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
    "0": "Company", "1": "EducationalInstitution", "2": "Artist",
    "3": "Athlete", "4": "OfficeHolder", "5": "MeanOfTransportation",
    "6": "Building", "7": "NaturalPlace", "8": "Village", "9": "Animal",
    "10": "Plant", "11": "Album", "12": "Film", "13": "WrittenWork"
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
        log.info("Loaded initial_weights.json — %d layers", len(weights))
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

def resource_score(meta):
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
    log.info("Aggregating round %d with %d clients...", round_number, len(client_updates))
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
    log.info("Aggregation complete. Now on round %d.", round_number)
    return True

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok",
        "round":       round_number,
        "num_classes": NUM_CLASSES,
        "dataset":     "DBpedia-14"
    })

@app.route("/status", methods=["GET"])
def status():
    with lock:
        waiting    = len(client_updates)
        waiting_ids = [u["device_id"] for u in client_updates]

    # Build detailed history with FedAvg info for each past round
    detailed_history = []
    for h in round_history[-5:]:
        detailed_history.append({
            "round":        h["round"],
            "num_clients":  h["num_clients"],
            "aggregation":  "FedAvg",
            "topic_dist":   h.get("topic_dist", {}),
            "timestamp":    h["timestamp"],
            "completed_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(h["timestamp"])
            ),
        })

    # Current round progress
    current_round_info = {
        "round":            round_number,
        "clients_submitted": waiting,
        "clients_needed":   MIN_CLIENTS - waiting,
        "min_clients":      MIN_CLIENTS,
        "progress_pct":     round(waiting / MIN_CLIENTS * 100),
        "submitted_devices": waiting_ids,
        "aggregation_method": "FedAvg",
        "status": "waiting_for_clients" if waiting < MIN_CLIENTS else "ready_to_aggregate",
    }

    return jsonify({
        "current_round":  current_round_info,
        "registered":     len(client_registry),
        "history":        detailed_history,
        # keep these flat fields so existing app code still works
        "round":          round_number,
        "clients_waiting": waiting,
        "min_clients":    MIN_CLIENTS,
    })

@app.route("/get_global_model", methods=["GET"])
def get_global_model():
    return jsonify({
        "weights":     global_weights,
        "round":       round_number,
        "layers":      len(global_weights),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES
    })

@app.route("/register", methods=["POST"])
def register():
    data      = request.get_json(force=True)
    device_id = data.get("device_id", "unknown")
    with lock:
        client_registry[device_id] = {**data, "registered_at": time.time()}
    log.info("Registered device: %s  total=%d", device_id, len(client_registry))
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
        msg = "Layer count mismatch: got %d, expected %d" % (len(weights), len(global_weights))
        return jsonify({"status": "error", "message": msg}), 400

    for i, (cl, gl) in enumerate(zip(weights, global_weights)):
        cs = np.array(cl).shape
        gs = np.array(gl).shape
        if cs != gs:
            msg = "Shape mismatch at layer %d: client=%s vs server=%s" % (i, cs, gs)
            return jsonify({"status": "error", "message": msg}), 400

    if client_round >= 0 and round_number - client_round > 2:
        return jsonify({
            "status":  "stale",
            "message": "Update from round %d; server is on %d." % (client_round, round_number),
            "round":   round_number
        }), 409

    resource = resource_score(data)
    log.info("[%s] weights received | samples=%s battery=%s%% score=%s",
             device_id, data.get("num_samples", 1),
             data.get("battery_level", "?"), resource)

    aggregated    = False
    clients_before = 0
    with lock:
        existing_ids = {u["device_id"] for u in client_updates}
        if device_id in existing_ids:
            return jsonify({
                "status":  "duplicate",
                "message": "Already received update from this device this round",
                "round":   round_number
            }), 200

        client_updates.append({
            "device_id":      device_id,
            "weights":        weights,
            "num_samples":    data.get("num_samples", 1),
            "topic_counts":   data.get("topic_counts", {}),
            "doc_types":      data.get("doc_types", []),
            "local_loss":     data.get("local_loss", 0.0),
            "battery_level":  data.get("battery_level", 100),
            "resource_score": resource,
            "received_at":    time.time(),
        })
        clients_before = len(client_updates)
        log.info("Collected %d/%d updates for round %d", clients_before, MIN_CLIENTS, round_number)
        aggregated = _try_aggregate()

    if aggregated:
        msg = "Round %d aggregated with %d clients." % (round_number - 1, MIN_CLIENTS)
        status_str = "aggregated"
    else:
        remaining = MIN_CLIENTS - clients_before
        msg = "Waiting for %d more client(s)." % remaining
        status_str = "accepted"

    return jsonify({
        "status":  status_str,
        "round":   round_number,
        "message": msg
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)