# Serverless Simulator - Enhanced Usage Guide

## Overview

The enhanced ServerlessSimulator now supports:
- Command-line argument parsing with `--input` flag
- Multiple runs with different random seeds
- Comprehensive output files with timing information
- Organized directory structure for results
- Aggregate statistics across multiple runs

## Quick Start

### Basic Usage

```bash
python AutoscalerFaas/ServerlessSimulator.py --input input.json
```

### Configuration File

The `input.json` file contains all experiment parameters. Here's a description of the key fields:

```json
{
  "arrival_rate": 5,              // Request arrival rate
  "warm_service": {
    "rate": 1,                    // Warm service rate
    "type": "Exponential"         // Distribution type
  },
  "cold_service": {
    "rate": 100,
    "type": "Exponential"
  },
  "cold_start": {
    "rate": 0.1,
    "type": "Exponential"
  },
  "expiration": {
    "rate": 0.1,
    "type": "Exponential"         // or "Deterministic"
  },
  "optimization": {
    "type": "adam",               // "adam", "RMSprop", or "SGD"
    "learning_rate": 0.01
  },
  "theta": [[1, 1, 5]],           // [theta, theta_min, gamma_exp]
  "tau": 1000,                    // Time horizon for optimization
  "max_currency": 50,             // Maximum concurrency
  "max_time": 100000,             // Maximum simulation time
  "K": 2,                         // Number of perturbations
  "seeds": [1, 42, 123, 456, 789],// Random seeds for multiple runs
  "exp_per_run": 1,               // Experiments per seed
  "log_dir": "logs/"              // Base output directory
}
```

## Running Multiple Experiments

### Multiple Seeds

To run multiple experiments with different random seeds, simply add them to the `seeds` array:

```json
{
  "seeds": [1, 42, 123, 456, 789],
  "exp_per_run": 1
}
```

This will run 5 independent simulations, one for each seed.

### Multiple Runs per Seed

To run multiple experiments with the same seed:

```json
{
  "seeds": [1],
  "exp_per_run": 3
}
```

This will run 3 experiments with seed 1.

## Output Files

The simulator creates a structured output directory:

```
logs/
└── experiment_arr5_Exponential_Exponential/
    └── theta_1_1_5_20250104_120000/
        ├── experiment_config.json          # Master configuration
        ├── aggregated_results.json         # All runs combined
        ├── all_runs_summary.csv           # CSV summary of all runs
        ├── run_1_seed_1/
        │   ├── config.json                # Run-specific config
        │   ├── results.json               # Detailed results
        │   ├── summary.txt                # Human-readable summary
        │   ├── theta.csv                  # Theta evolution
        │   ├── states.csv                 # State evolution
        │   └── all_costs.csv              # Cost evolution
        ├── run_2_seed_42/
        │   └── ...
        └── ...
```

### Output File Descriptions

#### Per-Run Files (in each `run_X_seed_Y/` directory)

- **config.json**: Complete configuration for this specific run
- **results.json**: Detailed results including:
  - Cold start probability
  - Rejection probability
  - Average server counts
  - Execution times (wall-clock and CPU)
  - Simulated time
- **summary.txt**: Human-readable summary with timing info
- **theta.csv**: Evolution of theta parameters over time
- **states.csv**: System states over time
- **all_costs.csv**: Cost values throughout the simulation

#### Aggregate Files (in the experiment root directory)

- **experiment_config.json**: The original input configuration
- **aggregated_results.json**: Combined results from all runs with statistics
- **all_runs_summary.csv**: CSV file with one row per run, easy for plotting

## Example Workflows

### 1. Quick Single Run

```bash
# Create a simple config
cat > quick_test.json << EOF
{
  "arrival_rate": 5,
  "warm_service": {"rate": 1, "type": "Exponential"},
  "cold_service": {"rate": 100, "type": "Exponential"},
  "cold_start": {"rate": 0.1, "type": "Exponential"},
  "expiration": {"rate": 0.1, "type": "Exponential"},
  "optimization": {"type": "adam", "learning_rate": 0.01},
  "theta": [[1, 1, 5]],
  "tau": 1000,
  "max_currency": 50,
  "max_time": 10000,
  "K": 2,
  "seeds": [1],
  "exp_per_run": 1,
  "log_dir": "logs/"
}
EOF

# Run it
python AutoscalerFaas/ServerlessSimulator.py --input quick_test.json
```

### 2. Multiple Seeds for Statistical Significance

```bash
# Edit input.json to include multiple seeds
# "seeds": [1, 42, 123, 456, 789, 999, 1337, 2048, 3141, 9999]

python AutoscalerFaas/ServerlessSimulator.py --input input.json
```

### 3. Parameter Sweep

Create multiple config files for different parameter values:

```bash
# config_arrival_5.json
# config_arrival_10.json
# config_arrival_15.json

for rate in 5 10 15; do
  python AutoscalerFaas/ServerlessSimulator.py --input config_arrival_${rate}.json
done
```

## Analyzing Results

### Using Python

```python
import pandas as pd
import json

# Load aggregate results
with open('logs/.../aggregated_results.json') as f:
    data = json.load(f)

# Or load CSV for easy analysis
df = pd.read_csv('logs/.../all_runs_summary.csv')

# Calculate statistics
print(f"Mean cold start probability: {df['prob_cold'].mean():.4f} ± {df['prob_cold'].std():.4f}")
print(f"Mean execution time: {df['wall_clock_time_seconds'].mean():.2f} seconds")

# Plot results
import matplotlib.pyplot as plt
df.plot(x='seed', y='prob_cold', kind='bar')
plt.title('Cold Start Probability by Seed')
plt.ylabel('Probability')
plt.show()
```

### Command Line Analysis

```bash
# View summary statistics
cat logs/.../aggregated_results.json | jq '.runs[] | {seed, prob_cold, prob_reject, wall_clock_time_seconds}'

# Extract execution times
cat logs/.../aggregated_results.json | jq '.runs[].wall_clock_time_seconds'

# CSV analysis with awk
awk -F',' 'NR>1 {sum+=$5; count++} END {print "Average cold start prob:", sum/count}' logs/.../all_runs_summary.csv
```

## Performance Tuning

### Reducing Simulation Time

- Decrease `max_time`
- Increase `tau` (less frequent optimization updates)
- Reduce number of seeds for quick tests

### Increasing Accuracy

- Increase `max_time` for longer runs
- Use more seeds for better statistics
- Decrease `tau` for more frequent updates

## Troubleshooting

### Out of Memory

- Reduce `max_time`
- Run fewer seeds in parallel
- Check system resources

### Slow Performance

- Enable progress bar to monitor (`progress=True`)
- Check if `tau` is too small (too many updates)
- Profile with timing information in output

### File Not Found Errors

- Ensure input file path is correct
- Check that all required fields are in config
- Verify JSON syntax is valid

## Advanced Configuration

### Custom Perturbation Patterns

```json
{
  "prtb": [
    [-1.0, 1.0],   // theta perturbation range
    [-0.5, 0.5],   // theta_min perturbation range
    [-2.0, 2.0]    // gamma_exp perturbation range
  ]
}
```

### Custom Learning Rates

```json
{
  "k_gamma": [2.0, 1.0, 0.5],  // Different rates for each parameter
  "exp_lr": [0.8, 0.85, 0.9]   // Exponential decay rates
}
```

## Migration from Old Code

If you have old scripts using hardcoded parameters:

**Old way:**
```python
sim = ServerlessSimulator(arrival_rate=5, warm_service_rate=1, ...)
```

**New way:**
1. Create a config file with your parameters
2. Run: `python AutoscalerFaas/ServerlessSimulator.py --input myconfig.json`

## Support

For issues or questions:
1. Check that input.json is valid JSON
2. Verify all required parameters are present
3. Review the error messages and stack traces
4. Check the generated log files for debugging information
