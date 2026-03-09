"""
API-ready vectorial SPSA autoscaler, decoupled from the simulator.

This class implements the same SPSA algorithm as Algorithm.py but without
any dependency on ServerlessSimulator. It receives system state observations
from a real Kubernetes/OpenFaaS platform and returns theta recommendations.

Usage:
    algo = APIVectorialAutoScaler.from_config(config_dict)
    result = algo.process_event(state=[40, 3, 5, 1, 1])
    theta = result["theta_step"]  # use this for scaling decisions
"""

from AutoscalerFaasScalarVectorial.utils import SystemState
import numpy as np
import json
import time
import os


class APIVectorialAutoScaler:

    def __init__(self, N, k_delta, k_gamma, theta_init, tau, K,
                 optimization="sgd", seed=1, log_dir="logs/api", **kwargs):
        self.N = N
        self.d = len(theta_init)
        self.theta = np.array(theta_init, dtype=float)
        self.theta_init = np.array(theta_init, dtype=float)
        self.theta_step = np.array(theta_init, dtype=float)

        self.k_delta = k_delta
        self.k_gamma = np.array(k_gamma, dtype=float)
        self.tau = tau
        self.K = K
        self.optimization = optimization.lower()
        self.seed = seed
        self.log_dir = log_dir

        # Extra params
        self.K_exp = kwargs.get('K_exp', 1000)
        self.gamma_min = kwargs.get('gamma_min', 1)
        self.prtb = kwargs.get('prtb', [[-0.5, 0.5], [-0.5, 0.5], [-1, 1]])
        self.learn_mask = np.array(kwargs.get('learn_mask', [True, True, True]))
        self.accumulate_cost = kwargs.get('accumulate_cost', True)

        # Cost weights
        self.state_elements_count = 5
        self.weights = [0] * self.state_elements_count
        w = kwargs.get('weights', {})
        self.weights[SystemState.COLD.value] = w.get('w_cold', 0)
        self.weights[SystemState.IDLE_ON.value] = w.get('w_idle_on', 2)
        self.weights[SystemState.BUSY.value] = w.get('w_busy', 1.0)
        self.weights[SystemState.INITIALIZING.value] = w.get('w_init', 5.0)
        self.weights[SystemState.INIT_RESERVED.value] = w.get('w_reserved', 100)
        self.w_rej = w.get('w_rej', 200)

        # State
        self.state = [0] * self.state_elements_count
        self.state[SystemState.COLD.value] = self.N

        # Iteration counters
        self.n = 1
        self.t = 0
        self.k = 0

        # Cost accumulators
        self.cost_avg_plus = 0.0
        self.cost_avg_minus = 0.0

        # Perturbation
        self.perturbation = np.zeros(self.d)
        self.perturbations = np.zeros(self.d)

        # Logging
        self.thetas = [self.theta.copy().tolist()]
        self.costs = []
        self.all_states = []
        self.all_costs = []

        # Adam
        self.m = np.zeros(self.d)
        self.v_adam = np.zeros(self.d)
        self.beta1 = np.full(self.d, 0.9)
        self.beta2 = np.full(self.d, 0.999)
        self.epsilon = 1e-8

        # RMSProp
        self.grad_avg_sq = np.zeros(self.d)
        self.beta_rms = np.full(self.d, 0.9)

        # RNGs for stochastic rounding and perturbation
        self.rng_perturb = np.random.default_rng(self.seed)
        self._reset_rounding_rngs()
        self._draw_perturbation()

        # Persistence
        self._auto_save_interval = kwargs.get('auto_save_interval', 0)
        self._auto_save_path = kwargs.get('auto_save_path', '')

    @classmethod
    def from_config(cls, config):
        """Create an instance from a config dictionary (matching algo_config.json)."""
        return cls(
            N=config['max_concurrency'],
            k_delta=config.get('k_delta', 1),
            k_gamma=config.get('k_gamma', [1, 1, 1]),
            theta_init=config['theta_init'],
            tau=config['tau'],
            K=config['K'],
            optimization=config.get('optimization', 'sgd'),
            seed=config.get('seed', 1),
            log_dir=config.get('log_dir', 'logs/api'),
            K_exp=config.get('K_exp', 1000),
            gamma_min=config.get('gamma_min', 1),
            prtb=config.get('prtb', [[-0.5, 0.5], [-0.5, 0.5], [-1, 1]]),
            learn_mask=config.get('learn_mask', [True, True, True]),
            accumulate_cost=config.get('accumulate_cost', True),
            weights=config.get('weights', {}),
            auto_save_interval=config.get('persistence', {}).get('auto_save_interval', 0),
            auto_save_path=config.get('persistence', {}).get('auto_save_path', ''),
        )

    def _reset_rounding_rngs(self):
        self.rng_delta_plus = np.random.default_rng(self.seed)
        self.rng_delta_minus = np.random.default_rng(self.seed)
        self.rng_delta_min_plus = np.random.default_rng(self.seed)
        self.rng_delta_min_minus = np.random.default_rng(self.seed)

    def _draw_perturbation(self):
        """Draw a new SPSA Rademacher perturbation vector."""
        delta_n = self.k_delta / (self.n ** (2.0 / 3.0))
        all_choices = np.array(self.prtb)
        random_indices = self.rng_perturb.integers(0, 2, size=all_choices.shape[0])
        self.perturbation = all_choices[np.arange(all_choices.shape[0]), random_indices]
        self.perturbations = self.perturbation * delta_n
        self.perturbations[~self.learn_mask] = 0.0

    def _stochastic_round(self, theta_real, rng):
        floor_val = np.floor(theta_real)
        p = theta_real - floor_val
        return int(rng.choice([floor_val, floor_val + 1], p=[1 - p, p]))

    def _compute_theta_step(self, opt_delta, rng_stock, rng_idle):
        """Compute rounded theta_step from a perturbed theta vector."""
        theta_stock_step = self._stochastic_round(opt_delta[0], rng_stock)
        theta_stock_step = int(min(max(theta_stock_step, 1), self.N))

        theta_idle_step = self._stochastic_round(opt_delta[1], rng_idle)
        theta_idle_step = int(min(max(theta_idle_step, 1), self.N))

        theta_exp_step = max(opt_delta[2], self.gamma_min)

        return np.array([theta_stock_step, theta_idle_step, theta_exp_step])

    def compute_cost(self, state, has_rejected_job=False):
        return np.dot(state, self.weights) + self.w_rej * has_rejected_job

    def get_sequences(self):
        """Return current SPSA sequences."""
        gamma_n = self.k_gamma / self.n
        delta_n = self.k_delta / (self.n ** (2.0 / 3.0))
        tau_n = int(self.tau * np.log(self.n + 1))
        return gamma_n, delta_n, tau_n

    def get_phase(self):
        """Return current phase name and budget."""
        _, _, tau_n = self.get_sequences()
        phase_budget = self.K * tau_n
        if self.k < phase_budget:
            return "plus", self.k, phase_budget
        else:
            return "minus", self.k - phase_budget, phase_budget

    def process_event(self, state, has_rejected_job=False, timestamp=None):
        """Process one state observation from the real system.

        Parameters
        ----------
        state : list of 5 ints
            [cold, idle_on, busy, initializing, init_reserved]
        has_rejected_job : bool
            Whether a request was rejected since the last event.
        timestamp : float, optional
            Unix timestamp of the event (for logging).

        Returns
        -------
        dict with current theta, phase info, and whether a gradient update occurred.
        """
        self.t += 1
        self.state = list(state)
        cost = self.compute_cost(state, has_rejected_job)

        self.all_states.append(list(state))
        self.all_costs.append(cost)

        gamma_n, delta_n, tau_n = self.get_sequences()
        gradient_update = False

        # === Plus phase ===
        if self.k < self.K * tau_n:
            opt_delta = self.theta + self.perturbations
            self.theta_step = self._compute_theta_step(
                opt_delta, self.rng_delta_plus, self.rng_delta_min_plus)
            self.k += 1

            if self.accumulate_cost:
                self.cost_avg_plus += cost
            else:
                if self.k % tau_n == 0:
                    self.cost_avg_plus += cost

            # Reset minus RNGs at end of plus phase
            if self.k == self.K * tau_n:
                self.rng_delta_minus = np.random.default_rng(self.seed)
                self.rng_delta_min_minus = np.random.default_rng(self.seed)

        # === Minus phase ===
        elif self.k < 2 * self.K * tau_n:
            opt_delta = self.theta - self.perturbations
            self.theta_step = self._compute_theta_step(
                opt_delta, self.rng_delta_minus, self.rng_delta_min_minus)
            self.k += 1

            if self.accumulate_cost:
                self.cost_avg_minus += cost
            else:
                if self.k % tau_n == 0:
                    self.cost_avg_minus += cost

            # === Gradient update ===
            if self.k == 2 * self.K * tau_n:
                gradient_update = True

                if self.accumulate_cost:
                    self.cost_avg_plus /= (self.K * tau_n)
                    self.cost_avg_minus /= (self.K * tau_n)
                else:
                    self.cost_avg_plus /= self.K
                    self.cost_avg_minus /= self.K

                # SPSA gradient
                grad = np.zeros(self.d)
                learned = self.learn_mask
                cost_diff = self.cost_avg_plus - self.cost_avg_minus
                grad[learned] = cost_diff / (2.0 * self.perturbations[learned])

                # Optimizer
                if self.optimization == "adam":
                    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
                    self.v_adam = self.beta2 * self.v_adam + (1 - self.beta2) * (grad ** 2)
                    m_hat = self.m / (1 - self.beta1 ** self.n)
                    v_hat = self.v_adam / (1 - self.beta2 ** self.n)
                    opt = self.theta - gamma_n * m_hat / (np.sqrt(v_hat) + self.epsilon)
                elif self.optimization == "rmsprop":
                    self.grad_avg_sq = self.beta_rms * self.grad_avg_sq + \
                                       (1 - self.beta_rms) * grad ** 2
                    opt = self.theta - gamma_n * grad / (np.sqrt(self.grad_avg_sq) + self.epsilon)
                else:
                    opt = self.theta - gamma_n * grad

                # Clip
                new_theta = np.array([
                    min(max(opt[0], 1), self.N),
                    min(max(opt[1], 1), self.N),
                    max(opt[2], self.gamma_min),
                ])
                new_theta[~self.learn_mask] = self.theta_init[~self.learn_mask]
                self.theta = new_theta
                self.theta_step = self.theta.copy()

                # Log
                avg_cost = (self.cost_avg_plus + self.cost_avg_minus) / 2.0
                self.thetas.append(self.theta.copy().tolist())
                self.costs.append(avg_cost)

                self._log_gradient_update(grad, tau_n)

                # Reset for next iteration
                self.cost_avg_plus = 0.0
                self.cost_avg_minus = 0.0
                self.n += 1
                self.k = 0

                self._reset_rounding_rngs()
                self._draw_perturbation()

        # Log event
        self._log_event(state, cost, timestamp)

        # Auto-save
        if self._auto_save_interval > 0 and self.t % self._auto_save_interval == 0:
            if self._auto_save_path:
                self.to_file(self._auto_save_path)

        phase_name, step_in_phase, phase_budget = self.get_phase()
        return {
            "theta": self.theta.tolist(),
            "theta_step": self.theta_step.tolist(),
            "expiration_rate": round(float(self.theta_step[2]) / self.K_exp, 6),
            "phase": phase_name,
            "iteration": int(self.n),
            "step_in_phase": int(step_in_phase),
            "phase_budget": int(phase_budget),
            "gradient_update_occurred": gradient_update,
            "cost": float(cost),
        }

    def force_update(self):
        """Force a gradient update with whatever costs have been accumulated so far.

        Useful when the real system has low traffic and phases take too long to complete.
        Returns the same dict as process_event, or None if not enough data.
        """
        _, delta_n, tau_n = self.get_sequences()
        gamma_n = self.k_gamma / self.n

        plus_steps = min(self.k, self.K * tau_n)
        minus_steps = max(0, self.k - self.K * tau_n)

        if plus_steps == 0:
            return None

        cost_plus = self.cost_avg_plus / plus_steps if self.accumulate_cost else self.cost_avg_plus / max(plus_steps // tau_n, 1)

        if minus_steps == 0:
            # Only have plus data - can't compute gradient
            return None

        cost_minus = self.cost_avg_minus / minus_steps if self.accumulate_cost else self.cost_avg_minus / max(minus_steps // tau_n, 1)

        # SPSA gradient
        grad = np.zeros(self.d)
        learned = self.learn_mask
        grad[learned] = (cost_plus - cost_minus) / (2.0 * self.perturbations[learned])

        # Optimizer (same logic as process_event)
        if self.optimization == "adam":
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v_adam = self.beta2 * self.v_adam + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.n)
            v_hat = self.v_adam / (1 - self.beta2 ** self.n)
            opt = self.theta - gamma_n * m_hat / (np.sqrt(v_hat) + self.epsilon)
        elif self.optimization == "rmsprop":
            self.grad_avg_sq = self.beta_rms * self.grad_avg_sq + \
                               (1 - self.beta_rms) * grad ** 2
            opt = self.theta - gamma_n * grad / (np.sqrt(self.grad_avg_sq) + self.epsilon)
        else:
            opt = self.theta - gamma_n * grad

        new_theta = np.array([
            min(max(opt[0], 1), self.N),
            min(max(opt[1], 1), self.N),
            max(opt[2], self.gamma_min),
        ])
        new_theta[~self.learn_mask] = self.theta_init[~self.learn_mask]
        self.theta = new_theta
        self.theta_step = self.theta.copy()

        avg_cost = (cost_plus + cost_minus) / 2.0
        self.thetas.append(self.theta.copy().tolist())
        self.costs.append(avg_cost)

        self._log_gradient_update(grad, tau_n=0)

        # Reset
        self.cost_avg_plus = 0.0
        self.cost_avg_minus = 0.0
        self.n += 1
        self.k = 0
        self._reset_rounding_rngs()
        self._draw_perturbation()

        phase_name, step_in_phase, phase_budget = self.get_phase()
        return {
            "theta": self.theta.tolist(),
            "theta_step": self.theta_step.tolist(),
            "expiration_rate": round(float(self.theta_step[2]) / self.K_exp, 6),
            "phase": phase_name,
            "iteration": self.n,
            "step_in_phase": step_in_phase,
            "phase_budget": phase_budget,
            "gradient_update_occurred": True,
            "forced": True,
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _jsonify(obj):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _log_event(self, state, cost, timestamp):
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "api_events.jsonl")
        phase_name, step_in_phase, _ = self.get_phase()
        entry = {
            "timestamp": timestamp or time.time(),
            "step": int(self.t),
            "state": [int(x) for x in state],
            "cost": float(cost),
            "theta": self.theta.tolist(),
            "theta_step": self.theta_step.tolist(),
            "phase": phase_name,
            "iteration": int(self.n),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_gradient_update(self, grad, tau_n):
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "gradient_updates.jsonl")
        entry = {
            "timestamp": time.time(),
            "iteration": int(self.n),
            "theta": self.theta.tolist(),
            "grad": grad.tolist(),
            "cost_plus": float(self.cost_avg_plus),
            "cost_minus": float(self.cost_avg_minus),
            "tau_n": int(tau_n),
            "perturbations": self.perturbations.tolist(),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"n = {self.n + 1}, theta = {self.theta}, grad = {grad}, "
              f"cost+ = {self.cost_avg_plus:.4f}, cost- = {self.cost_avg_minus:.4f}, "
              f"tau_n = {tau_n}")

    # ------------------------------------------------------------------
    # Serialization (for persistence and simulator replay)
    # ------------------------------------------------------------------

    def to_dict(self):
        """Serialize full algorithm state to a JSON-serializable dict."""
        return {
            "N": self.N,
            "d": self.d,
            "theta": self.theta.tolist(),
            "theta_init": self.theta_init.tolist(),
            "theta_step": self.theta_step.tolist(),
            "k_delta": self.k_delta,
            "k_gamma": self.k_gamma.tolist(),
            "tau": self.tau,
            "K": self.K,
            "optimization": self.optimization,
            "seed": self.seed,
            "log_dir": self.log_dir,
            "K_exp": self.K_exp,
            "gamma_min": self.gamma_min,
            "prtb": self.prtb,
            "learn_mask": self.learn_mask.tolist(),
            "accumulate_cost": self.accumulate_cost,
            "weights": self.weights,
            "w_rej": self.w_rej,
            "state": self.state,
            "n": self.n,
            "t": self.t,
            "k": self.k,
            "cost_avg_plus": self.cost_avg_plus,
            "cost_avg_minus": self.cost_avg_minus,
            "perturbation": self.perturbation.tolist(),
            "perturbations": self.perturbations.tolist(),
            "thetas": self.thetas,
            "costs": self.costs,
            # Optimizer state
            "m": self.m.tolist(),
            "v_adam": self.v_adam.tolist(),
            "grad_avg_sq": self.grad_avg_sq.tolist(),
            # RNG states
            "rng_perturb_state": self.rng_perturb.bit_generator.state,
            "rng_delta_plus_state": self.rng_delta_plus.bit_generator.state,
            "rng_delta_minus_state": self.rng_delta_minus.bit_generator.state,
            "rng_delta_min_plus_state": self.rng_delta_min_plus.bit_generator.state,
            "rng_delta_min_minus_state": self.rng_delta_min_minus.bit_generator.state,
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct from a serialized dict."""
        algo = cls(
            N=d["N"],
            k_delta=d["k_delta"],
            k_gamma=d["k_gamma"],
            theta_init=d["theta_init"],
            tau=d["tau"],
            K=d["K"],
            optimization=d["optimization"],
            seed=d["seed"],
            log_dir=d["log_dir"],
            K_exp=d["K_exp"],
            gamma_min=d["gamma_min"],
            prtb=d["prtb"],
            learn_mask=d["learn_mask"],
            accumulate_cost=d["accumulate_cost"],
        )
        algo.theta = np.array(d["theta"])
        algo.theta_step = np.array(d["theta_step"])
        algo.weights = d["weights"]
        algo.w_rej = d["w_rej"]
        algo.state = d["state"]
        algo.n = d["n"]
        algo.t = d["t"]
        algo.k = d["k"]
        algo.cost_avg_plus = d["cost_avg_plus"]
        algo.cost_avg_minus = d["cost_avg_minus"]
        algo.perturbation = np.array(d["perturbation"])
        algo.perturbations = np.array(d["perturbations"])
        algo.thetas = d["thetas"]
        algo.costs = d["costs"]
        algo.m = np.array(d["m"])
        algo.v_adam = np.array(d["v_adam"])
        algo.grad_avg_sq = np.array(d["grad_avg_sq"])

        # Restore RNG states
        algo.rng_perturb.bit_generator.state = d["rng_perturb_state"]
        algo.rng_delta_plus.bit_generator.state = d["rng_delta_plus_state"]
        algo.rng_delta_minus.bit_generator.state = d["rng_delta_minus_state"]
        algo.rng_delta_min_plus.bit_generator.state = d["rng_delta_min_plus_state"]
        algo.rng_delta_min_minus.bit_generator.state = d["rng_delta_min_minus_state"]

        return algo

    def to_file(self, path):
        """Save algorithm state to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)

    @classmethod
    def from_file(cls, path):
        """Load algorithm state from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def get_replay_data(self):
        """Return data needed to replay this experiment in the simulator."""
        return {
            "all_states": [[int(x) for x in s] for s in self.all_states],
            "all_costs": [float(c) for c in self.all_costs],
            "thetas": self.thetas,
            "costs": [float(c) for c in self.costs],
            "config": {
                "max_concurrency": self.N,
                "theta_init": self.theta_init.tolist(),
                "k_delta": self.k_delta,
                "k_gamma": self.k_gamma.tolist(),
                "tau": self.tau,
                "K": self.K,
                "optimization": self.optimization,
                "K_exp": self.K_exp,
                "gamma_min": self.gamma_min,
                "prtb": self.prtb,
                "learn_mask": self.learn_mask.tolist(),
                "accumulate_cost": self.accumulate_cost,
            }
        }
