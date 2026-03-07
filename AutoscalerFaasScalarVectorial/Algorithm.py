"""
Vectorial SPSA implementation of Algorithm 2 from Autoscaling.pdf (NSGD paper).

Jointly optimizes 3 parameters: theta = (theta_stock, theta_idle, theta_exp)
using Simultaneous Perturbation Stochastic Approximation (SPSA).

Key equations from the paper:
  - tau_n = tau * ln(n+1)                                  (Algorithm 2, line 3)
  - delta_n = k_delta / n^(2/3)                            (perturbation magnitude)
  - gamma_n = k_gamma / n                                  (step size, per-component)
  - u_n ~ Rademacher{-1,+1}^d                              (random direction, d=3)
  - perturbation = prtb_scale * u_n * delta_n               (element-wise scaling)
  - grad_j = [c_hat(theta+u*delta) - c_hat(theta-u*delta)] / (2*u_j*delta_n)  (eq. 2)

Cost function (eq. 7):
  C(theta, x) = sum_i w_i * x_i + I{x2+x4=N} * w_rej
  Paper weights: w1=2 (idle-on), w2=1 (busy), w3=5 (init), w4=100 (reserved), w_rej=200

Scale-up policy (Policy 1):
  - Cold start:  spawn 1 init-reserved + pi_theta_stock init-free
  - Warm start:  if #idle-on < pi_theta_idle, spawn up to pi_theta_stock - (#idle-on + #init-free)
  - Expiration:  rate = theta_exp / K_exp

Stochastic rounding (Section VI-A, item 4):
  pi_theta = floor(theta) w.p. ceil(theta)-theta, else ceil(theta)
  so that E[pi_theta] = theta.

Cost estimation modes:
  - accumulate=True  (default): time-average over all K*tau_n steps per phase (eq. 3 style)
  - accumulate=False: endpoint sampling — sample cost every tau_n steps, average K samples (eq. 3a/3b)
  The accumulate mode is more stable; endpoint mode matches the paper's literal eq. 3a/3b.

Optimizers supported: SGD (paper default), Adam, RMSProp.
"""

from AutoscalerFaasScalarVectorial.utils import SystemState
import numpy as np
import time


class VectorialAutoScalingAlgorithm:
    """SPSA-based autoscaling algorithm optimizing theta = (theta_stock, theta_idle, theta_exp).

    Parameters
    ----------
    N : int
        Maximum concurrency (number of function slots).
    k_delta : float
        Perturbation scale constant. delta_n = k_delta / n^(2/3).
    k_gamma : np.ndarray of shape (3,)
        Step size constants per dimension. gamma_n[j] = k_gamma[j] / n.
    theta_init : np.ndarray of shape (3,)
        Initial parameter vector [theta_stock, theta_idle, theta_exp].
    tau : float
        Base mixing time. tau_n = tau * ln(n+1).
    K : int
        Number of simulation runs per phase (for variance reduction).
    T : float
        Total simulation budget (number of steps).
    log_dir : str
        Directory for output logs.
    """

    def __init__(self, N, k_delta, k_gamma, theta_init, tau, K, T=1e9, log_dir=""):
        # --- System constants ---
        self.N = N                                # max concurrency
        self.Tmax = T                             # total simulation budget

        # --- Parameter vector: [theta_stock, theta_idle, theta_exp] ---
        self.d = len(theta_init)                  # dimensionality (3)
        self.theta = np.array(theta_init, dtype=float)
        self.theta_init = np.array(theta_init, dtype=float)
        self.theta_step = np.array(theta_init, dtype=float)  # current rounded/applied params

        # --- Sequence constants ---
        self.k_delta = k_delta                    # perturbation scale
        self.k_gamma = np.array(k_gamma, dtype=float)  # step size per dim
        self.tau = tau                            # base mixing time
        self.K = K                                # simulations per phase

        # --- State tracking ---
        self.state_elements_count = 5
        self.state = []
        self.init_state()
        self.init_weights()

        # --- Iteration counters ---
        self.n = 1          # outer iteration index (gradient step number)
        self.t = 0          # total simulation steps consumed
        self.k = 0          # inner counter within current gradient estimation

        # --- Cost accumulators (O(1) memory) ---
        self.cost_avg_plus = 0.0       # running sum for + phase
        self.cost_avg_minus = 0.0      # running sum for - phase

        # --- Current SPSA perturbation ---
        self.perturbation = np.zeros(self.d)   # u_n (Rademacher vector)
        self.perturbations = np.zeros(self.d)  # u_n * prtb_scale * delta_n

        # --- Logging ---
        self.thetas = []             # theta after each gradient update
        self.costs = []              # average cost at each gradient update
        self.states = []             # system state at each gradient update
        self.all_costs = []          # cost at every simulation step
        self.all_states = []         # state at every simulation step
        self.has_rejected_job = False
        self.last_checkpoint = time.time()
        self.log_dir = log_dir

        # --- Adam optimizer state (per-component) ---
        self.m = np.zeros(self.d)           # 1st moment estimate
        self.v_adam = np.zeros(self.d)       # 2nd moment estimate
        self.beta1 = np.full(self.d, 0.9)   # 1st moment decay
        self.beta2 = np.full(self.d, 0.999) # 2nd moment decay
        self.epsilon = 1e-8

        # --- RMSProp optimizer state (per-component) ---
        self.grad_avg_sq = np.zeros(self.d)
        self.beta_rms = np.full(self.d, 0.9)

        # --- Perturbation RNG (seeded per-run for reproducibility) ---
        self.rang_perturb = None

    # ------------------------------------------------------------------
    # Configuration setters (called after __init__ from ServerlessSimulator)
    # ------------------------------------------------------------------

    def set_params(self, **kwargs):
        """Set additional algorithm parameters from config.

        Parameters
        ----------
        K_exp : float
            Scaling factor for expiration rate: actual_rate = theta_exp / K_exp.
        gamma_min : float
            Minimum allowed value for theta_exp (prevents negative expiration).
        prtb : list of [float, float]
            Per-dimension perturbation choices. Each row [a, b] means u_j is chosen
            uniformly from {a, b}. Example: [[0.5, 0.5], [0.5, 0.5], [1, 1]].
            Paper uses [[0.5, -0.5], [0.5, -0.5], [1, -1]] for Rademacher.
        learn_mask : list of bool
            Which dimensions to optimize. [True, True, True] = learn all.
            [True, False, False] = learn only theta_stock (reduces to scalar case).
        accumulate_cost : bool
            If True (default), time-average costs over all steps in each phase.
            If False, use endpoint sampling (sample cost every tau_n steps, average K).
        """
        self.K_exp = kwargs.get('K_exp', 1000)
        self.gamma_min = kwargs.get('gamma_min', 1)
        self.prtb = kwargs.get('prtb', [
            [-0.5, 0.5],   # theta_stock perturbation choices
            [-0.5, 0.5],   # theta_idle perturbation choices
            [-1, 1]         # theta_exp perturbation choices
        ])
        self.learn_mask = np.array(kwargs.get('learn_mask', [True, True, True]))
        self.accumulate_cost = kwargs.get('accumulate_cost', True)

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def init_state(self):
        """Initialize system state: all servers start cold."""
        self.state = [0] * self.state_elements_count
        self.state[SystemState.COLD.value] = self.N
        self.state[SystemState.IDLE_ON.value] = 0
        self.state[SystemState.INITIALIZING.value] = 0
        self.state[SystemState.BUSY.value] = 0
        self.state[SystemState.INIT_RESERVED.value] = 0

    def init_weights(self):
        """Initialize cost function weights (eq. 7 in Autoscaling.pdf).

        Paper defaults: w1=2 (idle-on), w2=1 (busy), w3=5 (init), w4=100 (reserved).
        w_rej=200 (penalty for rejecting a job when x2+x4=N).
        """
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
        """Override cost function weights."""
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
        """Return the current rounded parameter vector [pi_theta_stock, pi_theta_idle, theta_exp]."""
        return self.theta_step

    def running_condition(self):
        """Check if the algorithm should continue (total steps < budget)."""
        return self.t < self.Tmax

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------

    def compute_cost(self, state=None):
        """Compute instantaneous cost C(x) = sum_i w_i*x_i + w_rej * I{rejected}.

        Parameters
        ----------
        state : list
            System state vector [cold, idle_on, busy, initializing, init_reserved].

        Returns
        -------
        float
            Instantaneous cost.
        """
        if state is None:
            raise Exception("State is not provided")
        self.state = state.copy()
        return np.dot(self.state, self.weights) + self.w_rej * self.has_rejected_job

    # ------------------------------------------------------------------
    # Stochastic rounding (Section VI-A, item 4)
    # ------------------------------------------------------------------

    def _stochastic_round(self, theta_real, rng):
        """Stochastic rounding: pi_theta = floor(theta) w.p. ceil(theta)-theta, else ceil(theta).

        Ensures E[pi_theta] = theta (unbiased integer rounding).

        Parameters
        ----------
        theta_real : float
            Real-valued parameter to round.
        rng : numpy.random.Generator
            RNG for the Bernoulli draw.

        Returns
        -------
        int
            Integer-valued rounded parameter.
        """
        floor_val = np.floor(theta_real)
        p = theta_real - floor_val  # probability of rounding up
        return int(rng.choice([floor_val, floor_val + 1], p=[1 - p, p]))

    # ------------------------------------------------------------------
    # Main simulation step (called at every event in the simulator)
    # ------------------------------------------------------------------

    def simulate_step(self, state, simulator):
        """Process one simulation step: accumulate cost and update theta when phases complete.

        This implements Algorithm 2 from Autoscaling.pdf. Each outer iteration n consists of:
          1. Plus phase  (K * tau_n steps): simulate with theta + u_n * delta_n
          2. Minus phase (K * tau_n steps): simulate with theta - u_n * delta_n
          3. Gradient update: theta_{n+1} = theta_n - gamma_n * grad  (SPSA formula, eq. 2)

        Parameters
        ----------
        state : list
            Current system state vector.
        simulator : ServerlessSimulator
            Reference to the simulator (for RNG access and logging).
        """
        self.t += 1
        cost = self.compute_cost(state)
        simulator.job_rejected = False

        # Log every step (for post-hoc analysis)
        self.all_states.append(state)
        self.all_costs.append(cost)

        # --- Compute sequences for current outer iteration n ---
        # gamma_n: per-component step size (k_gamma_j / n)
        gamma_n = self.k_gamma / self.n
        # delta_n: perturbation magnitude (k_delta / n^(2/3))
        delta_n = self.k_delta / (self.n ** (2.0 / 3.0))
        # tau_n: mixing time for this iteration (tau * ln(n+1))
        tau_n = int(self.tau * np.log(self.n + 1))

        # ================================================================
        # PHASE 1: Plus perturbation (theta + u_n * prtb_scale * delta_n)
        # ================================================================
        if self.k < self.K * tau_n:

            # --- First step of first iteration: initialize ---
            if self.k == 0 and self.n == 1:
                self.thetas.append(self.theta.copy())
                self.costs.append(np.zeros(self.d))

                # Initialize perturbation RNG (deterministic per seed)
                self.rang_perturb = np.random.default_rng(simulator.seed + 0 * 10)

                # Draw SPSA perturbation u_n from prtb choices (Rademacher-like)
                all_choices = np.array(self.prtb)
                random_indices = self.rang_perturb.integers(0, 2, size=all_choices.shape[0])
                self.perturbation = all_choices[np.arange(all_choices.shape[0]), random_indices]

                # Scale perturbation by delta_n; zero out fixed dimensions
                self.perturbations = self.perturbation * delta_n
                self.perturbations[~self.learn_mask] = 0.0

                # Initialize stochastic rounding RNGs (same seed for + and - phases)
                simulator.rang_delta_plus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_minus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_min_plus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_min_minus = np.random.default_rng(simulator.seed)

            # --- Apply perturbed theta to simulator ---
            opt_delta = self.theta + self.perturbations  # theta + u*delta

            # Stochastic rounding for theta_stock (integer servers to spawn)
            theta_stock_real = opt_delta[0]
            theta_stock_step = self._stochastic_round(theta_stock_real, simulator.rang_delta_plus)
            theta_stock_step = int(min(max(theta_stock_step, 1), self.N))

            # Stochastic rounding for theta_idle (integer threshold)
            theta_idle_real = opt_delta[1]
            theta_idle_step = self._stochastic_round(theta_idle_real, simulator.rang_delta_min_plus)
            theta_idle_step = int(min(max(theta_idle_step, 1), self.N))

            # theta_exp is continuous, just clip to gamma_min
            theta_exp_step = max(opt_delta[2], self.gamma_min)

            # Update simulator's expiration rate: xi = theta_exp / K_exp
            simulator.expiration_process.rate = round(theta_exp_step / self.K_exp, 6)

            # Store the applied (rounded) parameters
            self.theta_step = np.array([theta_stock_step, theta_idle_step, theta_exp_step])
            self.k += 1

            # --- Cost accumulation ---
            if self.accumulate_cost:
                # Time-average: accumulate cost at every step
                self.cost_avg_plus += cost
            else:
                # Endpoint: sample cost at the end of each tau_n window
                if self.k % tau_n == 0:
                    self.cost_avg_plus += cost

            # When + phase ends, reset RNGs for - phase (same seed = same rounding sequence)
            if self.k == self.K * tau_n:
                simulator.rang_delta_minus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_min_minus = np.random.default_rng(simulator.seed)

        # ================================================================
        # PHASE 2: Minus perturbation (theta - u_n * prtb_scale * delta_n)
        # ================================================================
        elif self.k < 2 * self.K * tau_n:

            # --- Apply negatively perturbed theta ---
            opt_delta = self.theta - self.perturbations  # theta - u*delta

            # Stochastic rounding for theta_stock
            theta_stock_real = opt_delta[0]
            theta_stock_step = self._stochastic_round(theta_stock_real, simulator.rang_delta_minus)
            theta_stock_step = int(min(max(theta_stock_step, 1), self.N))

            # Stochastic rounding for theta_idle
            theta_idle_real = opt_delta[1]
            theta_idle_step = self._stochastic_round(theta_idle_real, simulator.rang_delta_min_minus)
            theta_idle_step = int(min(max(theta_idle_step, 1), self.N))

            # theta_exp clipping
            theta_exp_step = max(opt_delta[2], self.gamma_min)

            # Update simulator's expiration rate
            simulator.expiration_process.rate = round(theta_exp_step / self.K_exp, 6)

            # Store the applied (rounded) parameters
            self.theta_step = np.array([theta_stock_step, theta_idle_step, theta_exp_step])
            self.k += 1

            # --- Cost accumulation ---
            if self.accumulate_cost:
                self.cost_avg_minus += cost
            else:
                if self.k % tau_n == 0:
                    self.cost_avg_minus += cost

            # ============================================================
            # GRADIENT UPDATE (at end of both phases)
            # ============================================================
            if self.k == 2 * self.K * tau_n:

                # --- Compute cost estimates ---
                if self.accumulate_cost:
                    # Time-average: divide by total steps in each phase
                    self.cost_avg_plus /= (self.K * tau_n)
                    self.cost_avg_minus /= (self.K * tau_n)
                else:
                    # Endpoint: divide by K (number of endpoint samples)
                    self.cost_avg_plus /= self.K
                    self.cost_avg_minus /= self.K

                # Broadcast scalar cost to vector for per-component gradient
                cost_plus_vec = np.full(self.d, self.cost_avg_plus)
                cost_minus_vec = np.full(self.d, self.cost_avg_minus)

                # --- SPSA gradient estimate (eq. 2) ---
                # grad_j = (c_hat_plus - c_hat_minus) / (2 * u_j * delta_n)
                # Only compute gradient for learned dimensions; fixed dims get zero
                grad = np.zeros(self.d)
                learned = self.learn_mask
                grad[learned] = (cost_plus_vec[learned] - cost_minus_vec[learned]) / \
                                (2.0 * self.perturbations[learned])

                # --- Apply optimizer ---
                if simulator.optimization == "adam":
                    # Adam: adaptive learning rate with bias correction
                    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
                    self.v_adam = self.beta2 * self.v_adam + (1 - self.beta2) * (grad ** 2)
                    m_hat = self.m / (1 - self.beta1 ** self.n)
                    v_hat = self.v_adam / (1 - self.beta2 ** self.n)
                    opt = self.theta - gamma_n * m_hat / (np.sqrt(v_hat) + self.epsilon)

                elif simulator.optimization == "RMSProp":
                    # RMSProp: running average of squared gradients
                    self.grad_avg_sq = self.beta_rms * self.grad_avg_sq + \
                                       (1 - self.beta_rms) * grad ** 2
                    opt = self.theta - gamma_n * grad / (np.sqrt(self.grad_avg_sq) + self.epsilon)

                else:
                    # SGD (vanilla, as in paper eq. 2)
                    opt = self.theta - gamma_n * grad

                # --- Clip parameters to valid ranges ---
                # theta_stock in [1, N], theta_idle in [0, N], theta_exp in [gamma_min, inf)
                theta_stock_opt = min(max(opt[0], 1), self.N)
                theta_idle_opt = min(max(opt[1], 1), self.N)
                theta_exp_opt = max(opt[2], self.gamma_min)
                new_theta = np.array([theta_stock_opt, theta_idle_opt, theta_exp_opt])

                # Keep fixed dimensions at their initial values
                new_theta[~self.learn_mask] = self.theta_init[~self.learn_mask]
                self.theta = new_theta

                # Update simulator expiration rate with new theta_exp
                simulator.expiration_process.rate = round(theta_exp_opt / self.K_exp, 6)
                self.theta_step = self.theta.copy()

                # --- Logging ---
                avg_cost = (self.cost_avg_plus + self.cost_avg_minus) / 2.0
                self.thetas.append(self.theta.copy())
                self.costs.append(np.full(self.d, avg_cost))
                self.states.append(self.state.copy())

                total_req, served_req = simulator.get_request_stats_between(
                    simulator.last_t, simulator.t)
                served = simulator.total_finished - simulator.last_total_finished
                simulator.last_total_finished = simulator.total_finished
                simulator.last_t = simulator.t

                print(f"n = {self.n + 1}, theta = {self.theta}, grad = {grad}, "
                      f"cost+ = {self.cost_avg_plus:.4f}, cost- = {self.cost_avg_minus:.4f}, "
                      f"tau_n = {tau_n}, perturbations = {self.perturbations}")

                with open(f"{self.log_dir}/theta_costs_v.txt", "a") as file:
                    file.write(f"{self.theta};{self.cost_avg_plus};{self.cost_avg_minus};"
                               f"{avg_cost};{total_req};{served_req};{served};{self.state}\n")

                # --- Reset for next outer iteration ---
                self.cost_avg_plus = 0.0
                self.cost_avg_minus = 0.0
                self.n += 1
                self.k = 0

                # Reset stochastic rounding RNGs
                simulator.rang_delta_plus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_min_plus = np.random.default_rng(simulator.seed)

                # Draw new SPSA perturbation for next iteration
                delta_n_new = self.k_delta / (self.n ** (2.0 / 3.0))
                all_choices = np.array(self.prtb)
                random_indices = self.rang_perturb.integers(0, 2, size=all_choices.shape[0])
                self.perturbation = all_choices[np.arange(all_choices.shape[0]), random_indices]
                self.perturbations = self.perturbation * delta_n_new
                self.perturbations[~self.learn_mask] = 0.0

                simulator.missed_update = 0

        self.has_rejected_job = False
