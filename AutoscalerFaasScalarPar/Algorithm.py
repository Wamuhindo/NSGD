"""
Scalar NSKW algorithm - parallel version.

Contains the full SPSA logic for the parallel scalar optimization:
  - State management and cost computation
  - Stochastic rounding (eq. 14)
  - Perturbation sequences (delta_n, gamma_n, tau_n)
  - Gradient estimation from K plus/minus cost averages
  - Optimizer updates (SGD, Adam, RMSProp)

In the parallel version, 2*K simulator processes run concurrently.
Each process calls compute_cost() and stochastic_round() locally.
The leader process (plus_0) uses compute_gradient() and apply_optimizer()
to update theta after all processes synchronize at a Barrier.

Key equations (paper 2502.01284v2):
  - tau_n = tau * ln(n+1)
  - delta_n = k_delta / n^(2/3)
  - gamma_n = k_gamma / n
  - grad = (1/K) * sum_k (c_hat_plus_k - c_hat_minus_k) / (2 * delta_n)
"""

from AutoscalerFaasScalarPar.utils import SystemState
import numpy as np
import time


class ScalarAutoScalingAlgorithm:
    def __init__(self, N, k_delta, k_gamma, theta_init, tau, K, T=1e9, log_dir=""):
        self.N = N
        self.k_delta = k_delta
        self.k_gamma = k_gamma
        self.Tmax = T
        self.theta = float(theta_init)
        self.theta_init = float(theta_init)
        self.theta_step = float(theta_init)
        self.state_elements_count = 5
        self.state = []
        self.tau = tau
        self.K = K
        self.init_state()
        self.init_weights()

        # Iteration counter
        self.n = 1
        self.t = 0

        # Logging
        self.costs = []
        self.thetas = []
        self.states = []
        self.all_costs = []
        self.all_states = []
        self.has_rejected_job = False
        self.last_checkpoint = time.time()
        self.log_dir = log_dir

        # Adam optimizer state (scalar)
        self.m = 0.0
        self.v_adam = 0.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        # RMSProp optimizer state (scalar)
        self.grad_avg_sq = 0.0
        self.beta_rms = 0.9

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def init_state(self):
        self.state = [0] * self.state_elements_count
        self.state[SystemState.COLD.value] = self.N

    def init_weights(self):
        # Paper eq. 2: w1=2 (idle_on), w2=1 (busy), w3=5 (init), w4=100 (reserved)
        self.weights = [0] * self.state_elements_count
        self.weights[SystemState.COLD.value] = 0
        self.weights[SystemState.IDLE_ON.value] = 2
        self.weights[SystemState.BUSY.value] = 1.0
        self.weights[SystemState.INITIALIZING.value] = 5.0
        self.weights[SystemState.INIT_RESERVED.value] = 100
        self.w_rej = 200

    # ------------------------------------------------------------------
    # Getters / setters
    # ------------------------------------------------------------------

    def set_has_rejected_job(self, has_rejected_job):
        self.has_rejected_job = has_rejected_job

    def set_weights(self, w_cold, w_idle_on, w_init_free, w_busy, w_reserved, w_rej):
        self.weights[SystemState.COLD.value] = w_cold
        self.weights[SystemState.IDLE_ON.value] = w_idle_on
        self.weights[SystemState.INITIALIZING.value] = w_init_free
        self.weights[SystemState.BUSY.value] = w_busy
        self.weights[SystemState.INIT_RESERVED.value] = w_reserved
        self.w_rej = w_rej

    def set_state(self, cold, idle_on, init_free, busy, init_reserved):
        self.state[SystemState.COLD.value] = cold
        self.state[SystemState.IDLE_ON.value] = idle_on
        self.state[SystemState.INITIALIZING.value] = init_free
        self.state[SystemState.BUSY.value] = busy
        self.state[SystemState.INIT_RESERVED.value] = init_reserved

    def get_state(self):
        return self.state

    def get_theta_step(self):
        return self.theta_step

    def running_condition(self):
        return self.t < self.Tmax

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------

    def compute_cost(self, state=None):
        """Compute instantaneous cost C(x) = sum_i w_i*x_i + w_rej * I{rejected}."""
        if state is None:
            raise Exception("State is not provided")
        self.state = state.copy()
        return np.dot(self.state, self.weights) + self.w_rej * self.has_rejected_job

    # ------------------------------------------------------------------
    # SPSA sequences
    # ------------------------------------------------------------------

    def get_delta_n(self, n=None):
        """Perturbation magnitude: delta_n = k_delta / n^(2/3)."""
        if n is None:
            n = self.n
        return round(self.k_delta / n ** (2.0 / 3.0), 6)

    def get_gamma_n(self, n=None):
        """Step size: gamma_n = k_gamma / n."""
        if n is None:
            n = self.n
        return self.k_gamma / n

    def get_tau_n(self, n=None):
        """Mixing time: tau_n = tau * ln(n+1)."""
        if n is None:
            n = self.n
        return int(self.tau * (1 + np.log10(n)))

    # ------------------------------------------------------------------
    # Stochastic rounding (eq. 14)
    # ------------------------------------------------------------------

    @staticmethod
    def stochastic_round(theta_real, rng):
        """Stochastic rounding: floor(theta) + Bernoulli(theta - floor(theta)).

        Ensures E[pi_theta] = theta (unbiased integer rounding).
        """
        floor_val = np.floor(theta_real)
        p = theta_real - floor_val
        return int(rng.choice([floor_val, floor_val + 1], p=[1 - p, p]))

    def get_perturbed_theta_step(self, theta_perturbed, rng):
        """Apply stochastic rounding and clip to [1, N]."""
        theta_step = self.stochastic_round(theta_perturbed, rng)
        return float(min(max(theta_step, 1), self.N))

    # ------------------------------------------------------------------
    # Gradient estimation (called by leader process after barrier sync)
    # ------------------------------------------------------------------

    def compute_gradient(self, avg_costs_plus, avg_costs_minus, delta_n):
        """Compute SPSA gradient estimate from K plus/minus cost averages.

        grad = (1/K) * sum_k (c_hat_plus_k - c_hat_minus_k) / (2 * delta_n)

        Parameters
        ----------
        avg_costs_plus : list of float
            Average costs from K plus-perturbation processes.
        avg_costs_minus : list of float
            Average costs from K minus-perturbation processes.
        delta_n : float
            Current perturbation magnitude.

        Returns
        -------
        float
            Estimated gradient.
        """
        grad = 0.0
        K = len(avg_costs_plus)
        for k in range(K):
            grad += (avg_costs_plus[k] - avg_costs_minus[k]) / (2.0 * delta_n)
        return grad / K

    # ------------------------------------------------------------------
    # Optimizer update (called by leader process)
    # ------------------------------------------------------------------

    def apply_optimizer(self, grad, optimization="SGD"):
        """Apply optimizer to update theta.

        Parameters
        ----------
        grad : float
            Estimated gradient.
        optimization : str
            Optimizer type: "SGD", "adam", or "RMSProp".

        Returns
        -------
        float
            New theta value (clipped to [1, N]).
        """
        gamma_n = self.get_gamma_n()

        if optimization == "adam":
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v_adam = self.beta2 * self.v_adam + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.n)
            v_hat = self.v_adam / (1 - self.beta2 ** self.n)
            new_theta = self.theta - gamma_n * m_hat / (v_hat ** 0.5 + self.epsilon)

        elif optimization == "RMSProp":
            self.grad_avg_sq = self.beta_rms * self.grad_avg_sq + (1 - self.beta_rms) * grad ** 2
            new_theta = self.theta - gamma_n * grad / (np.sqrt(self.grad_avg_sq) + self.epsilon)

        else:  # SGD
            new_theta = self.theta - gamma_n * grad

        # Clip to [1, N]
        new_theta = min(max(new_theta, 1.0), float(self.N))
        self.theta = new_theta
        self.theta_step = new_theta
        return new_theta

    def advance_iteration(self):
        """Advance to next outer iteration n."""
        self.n += 1
