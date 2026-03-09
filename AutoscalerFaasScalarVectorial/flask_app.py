"""
Flask API for the vectorial SPSA autoscaler.

Exposes the SPSA algorithm as a REST API for integration with
Kubernetes/OpenFaaS. The real system reports state observations
and receives theta (scaling parameters) in return.

Usage:
    python -m AutoscalerFaasScalarVectorial.flask_app --config algo_config.json
    python -m AutoscalerFaasScalarVectorial.flask_app --config algo_config.json --restore logs/api/autosave.json
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from flask import Flask, request, jsonify
from AutoscalerFaasScalarVectorial.api_algorithm import APIVectorialAutoScaler

import json
import time
import threading


def load_config(path):
    with open(path) as f:
        return json.load(f)


def create_app(config_path, restore_path=None):
    app = Flask(__name__)

    config = load_config(config_path)
    lock = threading.Lock()

    # Initialize or restore algorithm
    if restore_path and os.path.exists(restore_path):
        algo = APIVectorialAutoScaler.from_file(restore_path)
        print(f"Restored algorithm state from {restore_path} (iteration={algo.n}, steps={algo.t})")
    else:
        algo = APIVectorialAutoScaler.from_config(config)
        print(f"Initialized algorithm: theta={algo.theta.tolist()}, N={algo.N}")

    # Ensure log directory exists
    os.makedirs(algo.log_dir, exist_ok=True)

    # Save config alongside logs
    with open(os.path.join(algo.log_dir, 'algo_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # POST /event - Report a system state observation
    # ------------------------------------------------------------------
    @app.route('/event', methods=['POST'])
    def post_event():
        """Process a system state observation.

        Request body:
        {
            "state": [cold, idle_on, busy, initializing, init_reserved],
            "has_rejected_job": false,  // optional
            "timestamp": 1709000000.0   // optional
        }

        Returns current theta and phase info.
        """
        data = request.get_json()
        if not data or 'state' not in data:
            return jsonify({"error": "Missing 'state' in request body"}), 400

        state = data['state']
        if len(state) != 5:
            return jsonify({"error": "State must have 5 elements: [cold, idle_on, busy, init, init_reserved]"}), 400

        has_rejected = data.get('has_rejected_job', False)
        timestamp = data.get('timestamp', None)

        with lock:
            result = algo.process_event(state, has_rejected, timestamp)

        return jsonify(result)

    # ------------------------------------------------------------------
    # GET /theta - Get current theta vector
    # ------------------------------------------------------------------
    @app.route('/theta', methods=['GET'])
    def get_theta():
        """Get current scaling parameters.

        Returns:
        {
            "theta": [theta_stock, theta_idle, theta_exp],
            "theta_step": [pi_theta_stock, pi_theta_idle, theta_exp_applied],
            "expiration_rate": theta_exp / K_exp
        }
        """
        with lock:
            return jsonify({
                "theta": algo.theta.tolist(),
                "theta_step": algo.theta_step.tolist(),
                "expiration_rate": round(float(algo.theta_step[2]) / algo.K_exp, 6),
            })

    # ------------------------------------------------------------------
    # GET /status - Full algorithm status
    # ------------------------------------------------------------------
    @app.route('/status', methods=['GET'])
    def get_status():
        with lock:
            gamma_n, delta_n, tau_n = algo.get_sequences()
            phase_name, step_in_phase, phase_budget = algo.get_phase()

            return jsonify({
                "iteration": algo.n,
                "total_steps": algo.t,
                "phase": phase_name,
                "step_in_phase": step_in_phase,
                "phase_budget": phase_budget,
                "theta": algo.theta.tolist(),
                "theta_step": algo.theta_step.tolist(),
                "theta_history": algo.thetas,
                "cost_history": algo.costs,
                "cost_avg_plus": float(algo.cost_avg_plus),
                "cost_avg_minus": float(algo.cost_avg_minus),
                "current_state": algo.state,
                "perturbations": algo.perturbations.tolist(),
                "sequences": {
                    "gamma_n": gamma_n.tolist(),
                    "delta_n": delta_n,
                    "tau_n": tau_n,
                },
            })

    # ------------------------------------------------------------------
    # POST /snapshot - Save algorithm state
    # ------------------------------------------------------------------
    @app.route('/snapshot', methods=['POST'])
    def post_snapshot():
        """Save algorithm state for recovery or simulator replay.

        Request body (optional):
        {
            "path": "/custom/path/snapshot.json",
            "include_replay_data": true
        }
        """
        data = request.get_json() or {}
        ts = time.strftime("%Y%m%d_%H%M%S")

        snapshot_path = data.get('path', os.path.join(algo.log_dir, f"snapshot_{ts}.json"))

        with lock:
            algo.to_file(snapshot_path)

            replay_path = None
            if data.get('include_replay_data', False):
                replay_path = snapshot_path.replace('.json', '_replay.json')
                replay_data = algo.get_replay_data()
                with open(replay_path, 'w') as f:
                    json.dump(replay_data, f, indent=2)

        result = {"snapshot_path": snapshot_path}
        if replay_path:
            result["replay_path"] = replay_path

        return jsonify(result)

    # ------------------------------------------------------------------
    # POST /force-update - Force gradient update with partial data
    # ------------------------------------------------------------------
    @app.route('/force-update', methods=['POST'])
    def post_force_update():
        """Force a gradient update with accumulated costs so far.

        Useful when the real system has low traffic and phases take
        too long to complete naturally.
        """
        with lock:
            result = algo.force_update()

        if result is None:
            return jsonify({"error": "Not enough data for gradient update (need both plus and minus observations)"}), 400

        return jsonify(result)

    # ------------------------------------------------------------------
    # POST /config - Update configuration at runtime
    # ------------------------------------------------------------------
    @app.route('/config', methods=['POST'])
    def post_config():
        """Update algorithm configuration without restart.

        Request body (partial config - only provided keys are updated):
        {
            "weights": {"w_idle_on": 2, "w_busy": 1, ...},
            "optimization": "adam",
            "learn_mask": [true, true, true]
        }
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        with lock:
            if 'weights' in data:
                w = data['weights']
                from AutoscalerFaasScalarVectorial.utils import SystemState
                if 'w_cold' in w:
                    algo.weights[SystemState.COLD.value] = w['w_cold']
                if 'w_idle_on' in w:
                    algo.weights[SystemState.IDLE_ON.value] = w['w_idle_on']
                if 'w_busy' in w:
                    algo.weights[SystemState.BUSY.value] = w['w_busy']
                if 'w_init' in w:
                    algo.weights[SystemState.INITIALIZING.value] = w['w_init']
                if 'w_reserved' in w:
                    algo.weights[SystemState.INIT_RESERVED.value] = w['w_reserved']
                if 'w_rej' in w:
                    algo.w_rej = w['w_rej']

            if 'optimization' in data:
                algo.optimization = data['optimization'].lower()

            if 'learn_mask' in data:
                import numpy as np
                algo.learn_mask = np.array(data['learn_mask'])

        return jsonify({"status": "ok", "updated_keys": list(data.keys())})

    # ------------------------------------------------------------------
    # POST /reset - Reset algorithm to initial state
    # ------------------------------------------------------------------
    @app.route('/reset', methods=['POST'])
    def post_reset():
        """Reset the algorithm to its initial state (keeping config)."""
        nonlocal algo
        with lock:
            algo = APIVectorialAutoScaler.from_config(config)

        return jsonify({"status": "ok", "theta": algo.theta.tolist()})

    # ------------------------------------------------------------------
    # GET /health - Health check
    # ------------------------------------------------------------------
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok"})

    return app


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Flask API for Vectorial SPSA Autoscaler')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to algo_config.json')
    parser.add_argument('--restore', type=str, default=None,
                        help='Path to a saved snapshot to restore from')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        sys.exit(1)

    app = create_app(args.config, args.restore)
    app.run(host=args.host, port=args.port, debug=args.debug)
