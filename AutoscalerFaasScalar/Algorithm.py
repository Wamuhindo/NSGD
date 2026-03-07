"""
Clean scalar implementation of Algorithm 2 from paper 2502.01284v2.

Single-parameter Non-Stationary Kiefer-Wolfowitz (NSKW) algorithm.
Optimizes theta (number of extra servers to spawn) using a scalar perturbation.

Key equations from the paper:
  - tau_n = tau * ln(n+1)                          (Algorithm 2, line 3)
  - delta_n = k_delta / n^(2/3)                    (perturbation size)
  - gamma_n = k_gamma / n                          (step size)
  - grad = (f_hat_plus - f_hat_minus) / (2*delta_n)  (eq. 4)
  - theta_{n+1} = theta_n - gamma_n * grad           (eq. 4, SGD case)

Cost function (eq. 2):
  C(x) = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w_rej * I{x2+x4=N}
  with w1=w2=1, w3=5, w4=100, w_rej=1000

Scale-up rule pi_theta(x) (eq. 14):
  min((floor(theta) + I{V < theta - floor(theta)} - x3 + x4)+, N - x2 - x3 - 1)
  where x3 = #initializing, x4 = #reserved, V ~ Uniform(0,1)
"""

from AutoscalerFaasScalar.utils import SystemState
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

        # Iteration counters
        self.n = 1       # outer iteration (gradient step index)
        self.t = 0       # total simulation steps
        self.k = 0       # inner counter within current gradient estimation

        # Cost accumulators for + and - phases
        self.cost_avg_plus = 0.0
        self.cost_avg_minus = 0.0

        # Logging
        self.thetas = []
        self.costs = []
        self.states = []
        self.all_costs = []
        self.all_states = []
        self.has_rejected_job = False
        self.last_checkpoint = time.time()
        self.log_dir = log_dir

        # Adam parameters (scalar)
        self.m = 0.0
        self.v_adam = 0.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        # RMSProp parameters (scalar)
        self.grad_avg_sq = 0.0
        self.beta_rms = 0.9

        # Perturbation RNG
        self.rang_perturb = None

    def init_state(self):
        self.state = [0] * self.state_elements_count
        self.state[SystemState.COLD.value] = self.N
        self.state[SystemState.IDLE_ON.value] = 0
        self.state[SystemState.INITIALIZING.value] = 0
        self.state[SystemState.BUSY.value] = 0
        self.state[SystemState.INIT_RESERVED.value] = 0

    def init_weights(self):
        # Paper eq. 2: w1=w2=1 (idle_on, busy), w3=5 (initializing), w4=100 (reserved)
        self.weights = [0] * self.state_elements_count
        self.weights[SystemState.COLD.value] = 0
        self.weights[SystemState.IDLE_ON.value] = 2
        self.weights[SystemState.BUSY.value] = 1.0
        self.weights[SystemState.INITIALIZING.value] = 5.0
        self.weights[SystemState.INIT_RESERVED.value] = 100
        self.w_rej = 200

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

    def compute_cost(self, state=None):
        if state is None:
            raise Exception("State is not provided")
        self.state = state.copy()
        return np.dot(self.state, self.weights) + self.w_rej * self.has_rejected_job

    def _stochastic_round(self, theta_real, rng):
        """Stochastic rounding: floor(theta) + Bernoulli(theta - floor(theta))
        This implements the randomized policy from eq. 14."""
        floor_val = np.floor(theta_real)
        p = theta_real - floor_val
        return int(rng.choice([floor_val, floor_val + 1], p=[1 - p, p]))

    def simulate_step(self, state, simulator):
        self.t += 1
        cost = self.compute_cost(state)
        simulator.job_rejected = False

        self.all_states.append(state)
        self.all_costs.append(cost)

        # Algorithm 2 sequences
        gamma_n = self.k_gamma / self.n
        delta_n = self.k_delta / (self.n ** (2.0 / 3.0))
        tau_n = int(self.tau * np.log(self.n + 1))

        # === Phase 1: theta + delta_n (plus perturbation) ===
        if self.k < self.K * tau_n:
            if self.k == 0 and self.n == 1:
                self.thetas.append(self.theta)
                self.costs.append(0.0)
                # Initialize RNGs for stochastic rounding (same seed for + and - phases)
                simulator.rang_delta_plus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_minus = np.random.default_rng(simulator.seed)

            # Deterministic perturbation: theta + delta_n
            theta_plus = self.theta + delta_n
            theta_step = min(self._stochastic_round(theta_plus, simulator.rang_delta_plus), self.N)
            theta_step = max(theta_step, 1)
            self.theta_step = float(theta_step)
            self.k += 1

            # Accumulate ALL costs for time averaging (eq. 3: f̂ = (1/τ_n) Σ C(X_t))
            self.cost_avg_plus += cost

            # When + phase ends, reset RNG for - phase
            if self.k == self.K * tau_n:
                simulator.rang_delta_minus = np.random.default_rng(simulator.seed)

        # === Phase 2: theta - delta_n (minus perturbation) ===
        elif self.k < 2 * self.K * tau_n:
            # Deterministic perturbation: theta - delta_n
            theta_minus = self.theta - delta_n
            theta_step = max(self._stochastic_round(theta_minus, simulator.rang_delta_minus), 1)
            theta_step = min(theta_step, self.N)
            self.theta_step = float(theta_step)
            self.k += 1

            # Accumulate ALL costs for time averaging (eq. 3: f̂ = (1/τ_n) Σ C(X_t))
            self.cost_avg_minus += cost

            # === Gradient update at end of both phases ===
            if self.k == 2 * self.K * tau_n:
                # Time average: divide by total steps in each phase (K * tau_n)
                self.cost_avg_plus /= (self.K * tau_n)
                self.cost_avg_minus /= (self.K * tau_n)

                # Gradient estimate (eq. 4)
                grad = (self.cost_avg_plus - self.cost_avg_minus) / (2.0 * delta_n)

                # Apply optimizer
                if simulator.optimization == "adam":
                    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
                    self.v_adam = self.beta2 * self.v_adam + (1 - self.beta2) * (grad ** 2)
                    m_hat = self.m / (1 - self.beta1 ** self.n)
                    v_hat = self.v_adam / (1 - self.beta2 ** self.n)
                    new_theta = self.theta - gamma_n * m_hat / (np.sqrt(v_hat) + self.epsilon)

                elif simulator.optimization == "RMSProp":
                    self.grad_avg_sq = self.beta_rms * self.grad_avg_sq + (1 - self.beta_rms) * grad ** 2
                    new_theta = self.theta - gamma_n * grad / (np.sqrt(self.grad_avg_sq) + self.epsilon)

                else:  # SGD (vanilla, as in paper)
                    new_theta = self.theta - gamma_n * grad

                # Clip to [1, N]
                new_theta = min(max(new_theta, 1.0), float(self.N))
                self.theta = new_theta
                self.theta_step = self.theta

                # Log
                self.thetas.append(self.theta)
                avg_cost = (self.cost_avg_plus + self.cost_avg_minus) / 2.0
                self.costs.append(avg_cost)
                self.states.append(self.state.copy())

                total_req, served_req = simulator.get_request_stats_between(simulator.last_t, simulator.t)
                served = simulator.total_finished - simulator.last_total_finished
                simulator.last_total_finished = simulator.total_finished
                simulator.last_t = simulator.t

                print(f"n = {self.n + 1}, theta = {self.theta:.4f}, grad = {grad:.6f}, "
                      f"cost+ = {self.cost_avg_plus:.4f}, cost- = {self.cost_avg_minus:.4f}, "
                      f"tau_n = {tau_n}, delta_n = {delta_n:.6f}")

                with open(f"{self.log_dir}/theta_costs_v.txt", "a") as file:
                    file.write(f"{self.theta};{self.cost_avg_plus};{self.cost_avg_minus};{avg_cost};{total_req};{served_req};{served};{self.state}\n")

                # Reset for next outer iteration
                self.cost_avg_plus = 0.0
                self.cost_avg_minus = 0.0
                self.n += 1
                self.k = 0

                # Reset stochastic rounding RNGs
                simulator.rang_delta_plus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_minus = np.random.default_rng(simulator.seed)

        self.has_rejected_job = False
