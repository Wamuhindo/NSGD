# ServerlessSimulator Quick Start Guide

## 30-Second Start

```bash
# Test that everything works (takes ~1 second)
python AutoscalerFaas/ServerlessSimulator.py --input input_test_tiny.json

# Run a full experiment with 5 different seeds
python AutoscalerFaas/ServerlessSimulator.py --input input.json
```

## What You Get

After running, check the `logs/` directory:

```bash
# View the summary CSV
cat logs/experiment_*/theta_*/all_runs_summary.csv

# Or open in Excel/pandas for analysis
```

## Common Tasks

### 1. Change the number of runs
Edit `input.json`:
```json
{
  "seeds": [1, 42, 123, 456, 789, 999, 1337]  // Add more seeds here
}
```

### 2. Change simulation parameters
Edit `input.json`:
```json
{
  "arrival_rate": 10,        // Change arrival rate
  "max_time": 200000,        // Longer simulation
  "theta": [[2, 2, 10]]      // Different initial theta
}
```

### 3. Analyze results in Python
```python
import pandas as pd

# Load results
df = pd.read_csv('logs/experiment_*/theta_*/all_runs_summary.csv')

# Show statistics
print(df[['seed', 'prob_cold', 'prob_reject', 'wall_clock_time_seconds']].describe())

# Plot cold start probability
import matplotlib.pyplot as plt
df.plot(x='seed', y='prob_cold', kind='bar')
plt.title('Cold Start Probability by Seed')
plt.show()
```

## Output Files Explained

For each experiment run, you get:

### Per Run (`run_X_seed_Y/`):
- `results.json` - All metrics in JSON format
- `summary.txt` - Human-readable results
- `theta.csv` - How theta evolved over time
- `all_costs.csv` - Cost function values

### Aggregate (experiment root):
- `all_runs_summary.csv` - One row per run (open in Excel!)
- `aggregated_results.json` - All results combined
- `experiment_config.json` - Your input configuration

## Example Output

```
Statistics across 5 runs:
  Mean cold start probability: 0.0071 ± 0.0006
  Mean rejection probability: 0.0000 ± 0.0000
  Mean execution time: 0.29 ± 0.01 seconds
```

## Need More Help?

- Full usage guide: `SIMULATOR_USAGE.md`
- Enhancement details: `ENHANCEMENT_SUMMARY.md`
- Help command: `python AutoscalerFaas/ServerlessSimulator.py --help`

## Troubleshooting

**"Input file not found"**
```bash
# Make sure you're in the NSGD directory
cd /path/to/NSGD
python AutoscalerFaas/ServerlessSimulator.py --input input.json
```

**"Too slow"**
- Use `input_example_quick.json` for faster testing
- Or reduce `max_time` and number of `seeds` in your config

**"Want more statistical significance"**
- Add more seeds: `"seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`

## That's It!

You're ready to run experiments. Start with:
```bash
python AutoscalerFaas/ServerlessSimulator.py --input input_test_tiny.json
```

Then check the results in `logs/`.
