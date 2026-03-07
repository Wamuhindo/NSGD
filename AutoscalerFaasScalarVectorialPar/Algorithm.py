"""
Vectorial SPSA algorithm - parallel version (Autoscaling.pdf).

Contains the full 3-parameter SPSA logic for the parallel vectorial optimization:
  - State management and cost computation
  - Stochastic rounding (Section VI-A, item 4)
  - SPSA perturbation generation (Rademacher-like from prtb choices)
  - Perturbation sequences (delta_n, gamma_n, tau_n)
  - Gradient estimation from K plus/minus cost averages (eq. 2)
  - Optimizer updates (SGD, Adam, RMSProp)
  - learn_mask support for fixing dimensions
  - Configurable cost accumulation mode

In the parallel version, 2*K simulator processes run concurrently.
Each process calls compute_cost() and stochastic_round() locally.
The leader process (plus_0) uses compute_gradient() and apply_optimizer()
to update theta after all processes synchronize at a Barrier.

Key equations:
  - tau_n = tau * ln(n+1)
  - delta_n = k_delta / n^(2/3)
  - gamma_n = k_gamma / n (per-component)
  - grad_j = (1/K) * sum_k (c_plus_k - c_minus_k) / (2 * u_j * delta_n)

Cost function (eq. 7):
  C(theta, x) = sum_i w_i * x_i + I{rejected} * w_rej
  Paper weights: w1=2 (idle-on), w2=1 (busy), w3=5 (init), w4=100 (reserved), w_rej=200
"""

from AutoscalerFaasScalarVectorialPar.utils import SystemState
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
        # System constants
        self.N = N
        self.Tmax = T

        # Parameter vector: [theta_stock, theta_idle, theta_exp]
        self.d = len(theta_init)
        self.theta = np.array(theta_init, dtype=float)
        self.theta_init = np.array(theta_init, dtype=float)
        self.theta_step = np.array(theta_init, dtype=float)

        # Sequence constants
        self.k_delta = k_delta
        self.k_gamma = np.array(k_gamma, dtype=float)
        self.tau = tau
        self.K = K

        # State tracking
        self.state_elements_count = 5
        self.state = []
        self.init_state()
        self.init_weights()

        # Iteration counters
        self.n = 1
        self.t = 0

        # SPSA perturbation storage
        self.perturbation = np.zeros(self.d)
        self.perturbations = np.zeros(self.d)

        # Logging
        self.thetas = []
        self.costs = []
        self.states = []
        self.all_costs = []
        self.all_states = []
        self.has_rejected_job = False
        self.last_checkpoint = time.time()
        self.log_dir = log_dir

        # Adam optimizer state (per-component)
        self.m = np.zeros(self.d)
        self.v_adam = np.zeros(self.d)
        self.beta1 = np.full(self.d, 0.9)
        self.beta2 = np.full(self.d, 0.999)
        self.epsilon = 1e-8

        # RMSProp optimizer state (per-component)
        self.grad_avg_sq = np.zeros(self.d)
        self.beta_rms = np.full(self.d, 0.9)

        # Perturbation RNG
        self.rang_perturb = None

    # ------------------------------------------------------------------
    # Configuration (called after __init__)
    # ------------------------------------------------------------------

    def set_params(self, **kwargs):
        """Set additional algorithm parameters from config.

        Parameters
        ----------
        K_exp : float
            Scaling factor for expiration rate: actual_rate = theta_exp / K_exp.
        gamma_min : float
            Minimum allowed value for theta_exp.
        prtb : list of [float, float]
            Per-dimension perturbation choices. Each row [a, b] means u_j ~ {a, b}.
        learn_mask : list of bool
            Which dimensions to optimize. [True, True, True] = learn all.
        accumulate_cost : bool
            If True, time-average costs over all steps per phase.
        """
        self.K_exp = kwargs.get('K_exp', 1000)
        self.gamma_min = kwargs.get('gamma_min', 1)
        self.prtb = kwargs.get('prtb', [[-0.5, 0.5], [-0.5, 0.5], [-1, 1]])
        self.learn_mask = np.array(kwargs.get('learn_mask', [True, True, True]))
        self.accumulate_cost = kwargs.get('accumulate_cost', True)

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def init_state(self):
        self.state = [0] * self.state_elements_count
        self.state[SystemState.COLD.value] = self.N

    def init_weights(self):
        # Paper eq. 7: w1=2 (idle-on), w2=1 (busy), w3=5 (init), w4=100 (reserved)
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
        """Step size vector: gamma_n[j] = k_gamma[j] / n."""
        if n is None:
            n = self.n
        return self.k_gamma / n

    def get_tau_n(self, n=None):
        """Mixing time: tau_n = tau * ln(n+1)."""
        if n is None:
            n = self.n
        return int(self.tau * (1 + np.log10(n)))

    # ------------------------------------------------------------------
    # Stochastic rounding (Section VI-A, item 4)
    # ------------------------------------------------------------------

    @staticmethod
    def stochastic_round(theta_real, rng):
        """Stochastic rounding: floor(theta) + Bernoulli(theta - floor(theta))."""
        floor_val = np.floor(theta_real)
        p = theta_real - floor_val
        return int(rng.choice([floor_val, floor_val + 1], p=[1 - p, p]))

    # ------------------------------------------------------------------
    # SPSA perturbation generation
    # ------------------------------------------------------------------

    def generate_perturbation(self, rng, delta_n=None):
        """Generate SPSA perturbation vector from prtb choices.

        Parameters
        ----------
        rng : numpy.random.Generator
            RNG for perturbation draws.
        delta_n : float, optional
            Current perturbation magnitude. If None, computed from self.n.

        Returns
        -------
        np.ndarray
            Scaled perturbation vector (d,). Zero for non-learned dims.
        """
        if delta_n is None:
            delta_n = self.get_delta_n()

        all_choices = np.array(self.prtb)
        random_indices = rng.integers(0, 2, size=all_choices.shape[0])
        self.perturbation = all_choices[np.arange(all_choices.shape[0]), random_indices]

        self.perturbations = self.perturbation * delta_n
        self.perturbations[~self.learn_mask] = 0.0
        return self.perturbations

    def get_perturbed_theta_step(self, opt_delta, rang_stock, rang_idle):
        """Apply stochastic rounding to perturbed theta and compute theta_step.

        Parameters
        ----------
        opt_delta : np.ndarray
            Perturbed parameter vector (theta +/- perturbations).
        rang_stock : numpy.random.Generator
            RNG for theta_stock stochastic rounding.
        rang_idle : numpy.random.Generator
            RNG for theta_idle stochastic rounding.

        Returns
        -------
        np.ndarray
            theta_step = [pi_theta_stock, pi_theta_idle, theta_exp_clipped].
        """
        # Stochastic rounding for theta_stock (integer)
        theta_stock_step = self.stochastic_round(opt_delta[0], rang_stock)
        theta_stock_step = int(min(max(theta_stock_step, 1), self.N))

        # Stochastic rounding for theta_idle (integer)
        theta_idle_step = self.stochastic_round(opt_delta[1], rang_idle)
        theta_idle_step = int(min(max(theta_idle_step, 1), self.N))

        # theta_exp is continuous, clip to gamma_min
        theta_exp_step = max(opt_delta[2], self.gamma_min)

        return np.array([theta_stock_step, theta_idle_step, theta_exp_step])

    # ------------------------------------------------------------------
    # Gradient estimation (called by leader process after barrier sync)
    # ------------------------------------------------------------------

    def compute_gradient(self, avg_costs_plus, avg_costs_minus, perturbations):
        """Compute SPSA gradient estimate from K plus/minus cost averages.

        grad_j = (1/K) * sum_k (c_plus_k - c_minus_k) / (2 * perturbations_j)
        Only computes gradient for learned dimensions.

        Parameters
        ----------
        avg_costs_plus : list of float
            Average costs from K plus-perturbation processes.
        avg_costs_minus : list of float
            Average costs from K minus-perturbation processes.
        perturbations : np.ndarray
            Perturbation vector (d,) used for all K replications.

        Returns
        -------
        np.ndarray
            Estimated gradient vector (d,).
        """
        K = len(avg_costs_plus)
        grad = np.zeros(self.d)

        for k in range(K):
            avg_plus = np.full(self.d, avg_costs_plus[k])
            avg_minus = np.full(self.d, avg_costs_minus[k])
            learned = self.learn_mask
            grad[learned] += (avg_plus[learned] - avg_minus[learned]) / (2.0 * perturbations[learned])

        return grad / K

    # ------------------------------------------------------------------
    # Optimizer update (called by leader process)
    # ------------------------------------------------------------------

    def apply_optimizer(self, grad, optimization="SGD"):
        """Apply optimizer to update theta vector.

        Parameters
        ----------
        grad : np.ndarray
            Estimated gradient vector (d,).
        optimization : str
            Optimizer type: "SGD", "adam", or "RMSProp".

        Returns
        -------
        np.ndarray
            New theta vector (clipped per dimension).
        """
        gamma_n = self.get_gamma_n()

        if optimization == "adam":
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v_adam = self.beta2 * self.v_adam + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.n)
            v_hat = self.v_adam / (1 - self.beta2 ** self.n)
            opt = self.theta - gamma_n * m_hat / (np.sqrt(v_hat) + self.epsilon)

        elif optimization == "RMSProp":
            self.grad_avg_sq = self.beta_rms * self.grad_avg_sq + (1 - self.beta_rms) * grad ** 2
            opt = self.theta - gamma_n * grad / (np.sqrt(self.grad_avg_sq) + self.epsilon)

        else:  # SGD
            opt = self.theta - gamma_n * grad

        # Clip per dimension: theta_stock in [1,N], theta_idle in [1,N], theta_exp >= gamma_min
        theta_stock_opt = min(max(opt[0], 1), self.N)
        theta_idle_opt = min(max(opt[1], 1), self.N)
        theta_exp_opt = max(opt[2], self.gamma_min)
        new_theta = np.array([theta_stock_opt, theta_idle_opt, theta_exp_opt])

        # Keep fixed dimensions at their initial values
        new_theta[~self.learn_mask] = self.theta_init[~self.learn_mask]
        self.theta = new_theta
        self.theta_step = new_theta.copy()
        return new_theta

    def advance_iteration(self):
        """Advance to next outer iteration n."""
        self.n += 1
