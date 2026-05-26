# Simulation and Sanity-Check V2 Usage Report

## 1. Using `simulate.py` Directly

`simulate.py` is the base simulator. It expects two files:

- an input JSON file with simulation parameters
- a DAG JSON file with the workflow graph

Run it from the project root:

```bash
cd /opt/Projects/Soft2/NSGD
.venv/bin/python AutoscalerFaasScalarVectorial/simulate.py \
  --input path/to/input.json \
  --dag path/to/DAG.json
```

The DAG file must have this shape:

```json
{
  "DirectedAcyclicGraph": {
    "A1": {
      "next": ["A2"],
      "transition_probability": [1.0]
    },
    "A2": {
      "next": [],
      "transition_probability": []
    }
  }
}
```

The simulator writes results under the configured `log_dir`, in a timestamped folder named like:

```text
<experiment_name>_arr<arrival_rate>_<timestamp>/
```

Important output files:

```text
aggregated_results.json
all_runs_summary.csv
all_runs_metrics.csv
theta_*/graph_run_*/metrics.csv
```

### Input JSON Parameters

`description`
: Free text description of the experiment.

`experiment_name`
: Used in the output directory name.

`arrival_rate`
: External request arrival rate for the workflow root. This is the main workload intensity parameter.

`nodes`
: Per-DAG-node service configuration. Each DAG node must have a matching entry here.

Example:

```json
"nodes": {
  "A1": {
    "warm_service": {"rate": 1.5, "type": "Exponential"},
    "cold_service": {"rate": 300.0, "type": "Exponential"},
    "cold_start": {"rate": 0.3, "type": "Exponential"},
    "expiration": {"rate": 0.03, "type": "Exponential"}
  }
}
```

`warm_service.rate`
: Rate for warm execution service time. Higher rate means shorter mean service time.

`cold_service.rate`
: Rate for cold execution service time.

`cold_start.rate`
: Rate for cold-start initialization time.

`expiration.rate`
: Initial configured expiration distribution rate. In the vectorial algorithm, runtime expiration is also affected by `theta_exp / K_exp`.

`type`
: Distribution type. Current graph path commonly uses `"Exponential"`. Warm service also has code paths for `"Pareto"`; expiration supports `"Exponential"` or `"Deterministic"`.

`optimization.type`
: Intended optimizer name, for example `"sgd"`, `"adam"`, or `"RMSProp"`.

Note: the current `Algorithm.py` appears to force `simulator.optimization = "adam"` immediately before applying the optimizer update. That means changing this value may not fully affect behavior unless that simulator code is fixed separately.

`theta`
: List of initial theta vectors. Each theta is:

```text
[theta_stock, theta_idle, theta_exp]
```

Example:

```json
"theta": [[0.3333333333, 0.3333333333, 30]]
```

`theta_stock`
: Controls scale-up stock target.

`theta_idle`
: Controls preemptive provisioning when idle warm instances fall below this level.

`theta_exp`
: Controls expiration through `theta_exp / K_exp`.

`tau`
: SPSA phase length / update interval used by the autoscaling algorithm.

`max_concurrency`
: Shared maximum concurrency/resource budget for the graph simulation.

`max_time`
: Algorithm time/event budget used by the simulator.

`log_dir`
: Base output directory for logs and results.

`K`
: Number of cost samples or repetitions used inside the SPSA update logic.

`seeds`
: List of random seeds. The simulator runs each theta for each seed.

`exp_per_run`
: Number of repeated experiments per theta/seed combination.

`k_delta`
: SPSA perturbation decay parameter.

`k_gamma`
: Learning-rate scaling vector for `[theta_stock, theta_idle, theta_exp]`.

Example:

```json
"k_gamma": [1, 1, 1]
```

`k_gamma_per_theta`
: Optional list of `k_gamma` vectors, one per theta entry. If present, it overrides `k_gamma` for matching theta indices.

`prtb`
: Perturbation choices for each theta dimension.

Example:

```json
"prtb": [[-0.5, 0.5], [-0.5, 0.5], [-1, 1]]
```

`learn_mask`
: Boolean mask for which theta dimensions are learned.

Example:

```json
"learn_mask": [true, true, true]
```

Meaning:

```text
[learn theta_stock, learn theta_idle, learn theta_exp]
```

Examples:

```json
[true, true, true]
```

learn all three.

```json
[true, false, false]
```

learn only `theta_stock`.

`accumulate_cost`
: If `true`, costs are accumulated over the phase. If `false`, endpoint sampling is used.

`K_exp`
: Scaling factor for expiration rate. The effective learned expiration rate is based on:

```text
theta_exp / K_exp
```

`gamma_min`
: Minimum allowed value for `theta_exp`.

`theta_stock_min`
: Minimum allowed value for `theta_stock`.

`theta_idle_min`
: Minimum allowed value for `theta_idle`.

`stop_by_simulated_time`
: If `true`, graph simulation can stop by simulated wall-clock time instead of only algorithm budget.

`max_simulated_time`
: Simulated-time stopping threshold used when `stop_by_simulated_time` is `true`.

## 2. Using `run_sanitycheck_suite_v2.py`

`run_sanitycheck_suite_v2.py` is a wrapper around `simulate.py`.

It does four things:

1. Reads a sweep JSON.
2. Generates all simulation input JSON and DAG JSON files.
3. Runs `simulate.py` once per generated configuration.
4. Collects results into `results.csv` and creates 2D/3D plots.

Run it:

```bash
cd /opt/Projects/Soft2/NSGD
.venv/bin/python run_sanitycheck_suite_v2.py \
  --config sanitycheck_sweep_config.example.json \
  --output-prefix sanitycheck_sweep_run
```

Generate configs only:

```bash
.venv/bin/python run_sanitycheck_suite_v2.py \
  --config sanitycheck_sweep_config.example.json \
  --generate-only \
  --output-prefix sanitycheck_sweep_run
```

Stop immediately on the first failed simulation:

```bash
.venv/bin/python run_sanitycheck_suite_v2.py \
  --config sanitycheck_sweep_config.example.json \
  --stop-on-error \
  --output-prefix sanitycheck_sweep_run
```

Default behavior is to continue after failed simulations and record failed rows in `results.csv`.

### V2 JSON Structure

The v2 JSON has five top-level sections:

```json
{
  "base_config": {},
  "node_defaults": {},
  "sweeps": {},
  "z_metrics": [],
  "plot_by_parameters": []
}
```

`base_config`
: Base simulator input. Most fields are copied directly into the generated `simulate.py` input JSON.

`node_defaults`
: Template for each generated DAG node.

`sweeps`
: Parameters to vary. Every list creates a Cartesian product with every other list.

`z_metrics`
: Output metrics to plot on the z-axis in 3D plots and y-axis in 2D plots.

`plot_by_parameters`
: Input parameters to use as the non-`N` axis.

### Sweep Parameters

Use names close to the original simulator JSON.

`N_values`
: Number of DAG nodes. This is not a direct `simulate.py` field; the v2 script uses it to generate the DAG and `nodes` object.

`arrival_rate_values`
: Values for simulator `arrival_rate`.

`warm_service_values`
: Base warm service rates. These become `nodes.A*.warm_service.rate`.

`warm_service_mean_values`
: Alternative to `warm_service_values`. Values are mean times and are converted to rates with `rate = 1 / mean`.

`cold_service_values`
: Base cold service rates.

`cold_service_mean_values`
: Cold service mean times, converted to rates.

`cold_start_values`
: Base cold-start rates.

`cold_start_mean_values`
: Cold-start mean times, converted to rates.

`expiration_values`
: Base expiration rates.

`expiration_mean_values`
: Expiration mean times, converted to rates.

`optimization_values`
: Values for `optimization.type`.

`theta_values`
: Values for `theta`.

`tau_values`
: Values for `tau`.

`max_concurrency_values`
: Values for `max_concurrency`.

`max_time_values`
: Values for `max_time`.

`K_values`
: Values for `K`.

`seeds_values`
: Values for `seeds`.

`k_delta_values`
: Values for `k_delta`.

`k_gamma_values`
: Values for `k_gamma`.

`k_gamma_per_theta_values`
: Values for `k_gamma_per_theta`.

`prtb_values`
: Values for `prtb`.

`learn_mask_values`
: Values for `learn_mask`.

`accumulate_cost_values`
: Values for `accumulate_cost`.

`K_exp_values`
: Values for `K_exp`.

`gamma_min_values`
: Values for `gamma_min`.

`theta_stock_min_values`
: Values for `theta_stock_min`.

`theta_idle_min_values`
: Values for `theta_idle_min`.

`stop_by_simulated_time_values`
: Values for `stop_by_simulated_time`.

`max_simulated_time_values`
: Values for `max_simulated_time`.

`exp_per_run_values`
: Values for `exp_per_run`.

### Node Rate Scaling

In `node_defaults`:

```json
"scale_node_rates_by_N": true
```

means each node rate is multiplied by `N`.

This preserves the old sanity-check convention: as chain length grows, each node becomes proportionally faster, so the total expected serial service time remains comparable.

If you want literal rates exactly as written, use:

```json
"scale_node_rates_by_N": false
```

### Axis Rules

For 3D plots:

```text
x-axis = selected input parameter from plot_by_parameters
y-axis = N
z-axis = selected output metric from z_metrics
```

Example:

```json
"plot_by_parameters": ["arrival_rate"],
"z_metrics": ["response_time"]
```

produces:

```text
x = arrival_rate
y = N
z = response_time_avg
```

For 2D plots:

```text
x-axis = N
y-axis = selected output metric from z_metrics
curve/color = one value of selected input parameter
```

Example:

```json
"plot_by_parameters": ["arrival_rate"],
"z_metrics": ["response_time"]
```

produces curves like:

```text
response_time_avg vs N, one curve for arrival_rate=10
response_time_avg vs N, one curve for arrival_rate=20
```

### Metric Names

The v2 script validates requested metrics against simulator output columns.

Useful aliases:

`response_time`
: Resolves to `response_time_avg`.

`rejection_probability`
: Resolves to `prob_reject`.

`cold_start_probability`
: Resolves to `prob_cold`.

`queued_probability`
: Resolves to `prob_queued`.

`average_cost`
: Resolves to `graph_time_avg_cost`.

You can also use exact output columns from `all_runs_metrics.csv`, such as:

```text
response_time_avg
response_time_p95
response_time_p99
prob_reject
prob_cold
graph_event_avg_cost
graph_time_avg_cost
inst_count_avg
reqs_total
```

### Output Files

For:

```bash
--output-prefix sanitycheck_sweep_run
```

the v2 script writes:

```text
sanitycheck_sweep_run/generated_configs.csv
sanitycheck_sweep_run/results.csv
sanitycheck_sweep_run/configs/
sanitycheck_sweep_run/logs/
sanitycheck_sweep_run/plots/2d/
sanitycheck_sweep_run/plots/3d/
```

Example plot filenames:

```text
plot_2d_response_time_avg_vs_N_by_arrival_rate.png
plot_3d_response_time_avg_N_arrival_rate.png
plot_2d_graph_time_avg_cost_vs_N_by_max_concurrency.png
plot_3d_graph_time_avg_cost_N_max_concurrency.png
```

## 3. Example JSON Files

### Example 1: Direct `simulate.py` Input JSON

Save as `input_direct_example.json`:

```json
{
  "description": "Direct simulate.py example",
  "experiment_name": "direct_example",
  "arrival_rate": 10,
  "nodes": {
    "A1": {
      "warm_service": {
        "rate": 1.5,
        "type": "Exponential"
      },
      "cold_service": {
        "rate": 300.0,
        "type": "Exponential"
      },
      "cold_start": {
        "rate": 0.3,
        "type": "Exponential"
      },
      "expiration": {
        "rate": 0.03,
        "type": "Exponential"
      }
    },
    "A2": {
      "warm_service": {
        "rate": 1.5,
        "type": "Exponential"
      },
      "cold_service": {
        "rate": 300.0,
        "type": "Exponential"
      },
      "cold_start": {
        "rate": 0.3,
        "type": "Exponential"
      },
      "expiration": {
        "rate": 0.03,
        "type": "Exponential"
      }
    }
  },
  "optimization": {
    "type": "sgd"
  },
  "theta": [
    [
      0.5,
      0.5,
      20
    ]
  ],
  "tau": 100,
  "max_concurrency": 50,
  "max_time": 200000,
  "log_dir": "direct_example/logs",
  "K": 2,
  "seeds": [
    1
  ],
  "k_delta": 1,
  "k_gamma": [
    1,
    1,
    1
  ],
  "prtb": [
    [
      -0.5,
      0.5
    ],
    [
      -0.5,
      0.5
    ],
    [
      -1,
      1
    ]
  ],
  "learn_mask": [
    true,
    true,
    true
  ],
  "accumulate_cost": true,
  "K_exp": 1000,
  "gamma_min": 1,
  "theta_stock_min": 0,
  "theta_idle_min": 0
}
```

Save as `DAG_direct_example.json`:

```json
{
  "DirectedAcyclicGraph": {
    "A1": {
      "next": [
        "A2"
      ],
      "transition_probability": [
        1.0
      ]
    },
    "A2": {
      "next": [],
      "transition_probability": []
    }
  }
}
```

Run:

```bash
.venv/bin/python AutoscalerFaasScalarVectorial/simulate.py \
  --input input_direct_example.json \
  --dag DAG_direct_example.json
```

### Example 2: V2 Sweep Over Arrival Rate and N

Save as `sweep_arrival_rate.json`:

```json
{
  "base_config": {
    "description": "Sweep arrival rate and N",
    "experiment_name": "sweep_arrival_rate",
    "arrival_rate": 10,
    "optimization": {
      "type": "sgd"
    },
    "theta": [
      [
        0.3333333333333333,
        0.3333333333333333,
        30
      ]
    ],
    "tau": 100,
    "max_concurrency": 50,
    "max_time": 200000,
    "K": 2,
    "seeds": [
      1
    ],
    "k_delta": 1,
    "k_gamma": [
      1,
      1,
      1
    ],
    "learn_mask": [
      true,
      true,
      true
    ],
    "accumulate_cost": true,
    "K_exp": 1000,
    "gamma_min": 1,
    "theta_stock_min": 0,
    "theta_idle_min": 0
  },
  "node_defaults": {
    "warm_service": {
      "rate": 0.5,
      "type": "Exponential"
    },
    "cold_service": {
      "rate": 100.0,
      "type": "Exponential"
    },
    "cold_start": {
      "rate": 0.1,
      "type": "Exponential"
    },
    "expiration": {
      "rate": 0.01,
      "type": "Exponential"
    },
    "scale_node_rates_by_N": true
  },
  "sweeps": {
    "N_values": [
      1,
      3,
      5
    ],
    "arrival_rate_values": [
      10,
      20,
      30
    ]
  },
  "z_metrics": [
    "response_time",
    "rejection_probability",
    "average_cost"
  ],
  "plot_by_parameters": [
    "arrival_rate"
  ]
}
```

Run:

```bash
.venv/bin/python run_sanitycheck_suite_v2.py \
  --config sweep_arrival_rate.json \
  --output-prefix sweep_arrival_rate_run
```

### Example 3: V2 Sweep Over Max Concurrency

Save as `sweep_max_concurrency.json`:

```json
{
  "base_config": {
    "description": "Sweep max concurrency",
    "experiment_name": "sweep_max_concurrency",
    "arrival_rate": 20,
    "optimization": {
      "type": "sgd"
    },
    "theta": [
      [
        0.3333333333333333,
        0.3333333333333333,
        30
      ]
    ],
    "tau": 100,
    "max_concurrency": 50,
    "max_time": 200000,
    "K": 2,
    "seeds": [
      1
    ],
    "k_delta": 1,
    "k_gamma": [
      1,
      1,
      1
    ],
    "learn_mask": [
      true,
      true,
      true
    ],
    "accumulate_cost": true,
    "K_exp": 1000,
    "gamma_min": 1,
    "theta_stock_min": 0,
    "theta_idle_min": 0
  },
  "node_defaults": {
    "warm_service": {
      "rate": 0.5,
      "type": "Exponential"
    },
    "cold_service": {
      "rate": 100.0,
      "type": "Exponential"
    },
    "cold_start": {
      "rate": 0.1,
      "type": "Exponential"
    },
    "expiration": {
      "rate": 0.01,
      "type": "Exponential"
    },
    "scale_node_rates_by_N": true
  },
  "sweeps": {
    "N_values": [
      1,
      3,
      5
    ],
    "max_concurrency_values": [
      25,
      50,
      100
    ]
  },
  "z_metrics": [
    "response_time",
    "rejection_probability",
    "average_cost"
  ],
  "plot_by_parameters": [
    "max_concurrency"
  ]
}
```

Run:

```bash
.venv/bin/python run_sanitycheck_suite_v2.py \
  --config sweep_max_concurrency.json \
  --output-prefix sweep_max_concurrency_run
```

### Example 4: V2 Sweep Over Learning Mask and Gamma

Save as `sweep_learning.json`:

```json
{
  "base_config": {
    "description": "Sweep learning parameters",
    "experiment_name": "sweep_learning",
    "arrival_rate": 20,
    "optimization": {
      "type": "sgd"
    },
    "theta": [
      [
        0.3333333333333333,
        0.3333333333333333,
        30
      ]
    ],
    "tau": 100,
    "max_concurrency": 50,
    "max_time": 200000,
    "K": 2,
    "seeds": [
      1
    ],
    "k_delta": 1,
    "k_gamma": [
      1,
      1,
      1
    ],
    "learn_mask": [
      true,
      true,
      true
    ],
    "accumulate_cost": true,
    "K_exp": 1000,
    "gamma_min": 1,
    "theta_stock_min": 0,
    "theta_idle_min": 0
  },
  "node_defaults": {
    "warm_service": {
      "rate": 0.5,
      "type": "Exponential"
    },
    "cold_service": {
      "rate": 100.0,
      "type": "Exponential"
    },
    "cold_start": {
      "rate": 0.1,
      "type": "Exponential"
    },
    "expiration": {
      "rate": 0.01,
      "type": "Exponential"
    },
    "scale_node_rates_by_N": true
  },
  "sweeps": {
    "N_values": [
      3,
      5
    ],
    "learn_mask_values": [
      [
        true,
        true,
        true
      ],
      [
        true,
        false,
        false
      ]
    ],
    "k_gamma_values": [
      [
        1,
        1,
        1
      ],
      [
        0.5,
        0.5,
        0.5
      ]
    ],
    "gamma_min_values": [
      1,
      5
    ]
  },
  "z_metrics": [
    "response_time",
    "rejection_probability",
    "average_cost"
  ],
  "plot_by_parameters": [
    "gamma_min",
    "learn_mask",
    "k_gamma"
  ]
}
```

Run:

```bash
.venv/bin/python run_sanitycheck_suite_v2.py \
  --config sweep_learning.json \
  --output-prefix sweep_learning_run
```
