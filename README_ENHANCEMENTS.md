# ServerlessSimulator.py Enhancements

## Summary

The `ServerlessSimulator.py` has been significantly enhanced to support professional, reproducible scientific experiments with multiple runs, comprehensive output files, and proper timing information.

## Key Enhancements

### ✅ 1. Argument Parser (argparse)
- Proper command-line interface with `--input` flag
- Help documentation with `--help`
- Input file validation

### ✅ 2. Multiple Run Support
- Run experiments with different random seeds
- Specify seeds as array in `input.json`: `"seeds": [1, 42, 123, ...]`
- Multiple experiments per seed: `"exp_per_run": N`
- Each run is completely independent

### ✅ 3. Enhanced Configuration
- Comprehensive `input.json` with all parameters
- Example configurations for different use cases:
  - `input.json` - Full production config (5 seeds)
  - `input_example_quick.json` - Quick testing (3 seeds, shorter time)
  - `input_test_tiny.json` - Validation (2 seeds, very short)

### ✅ 4. Organized Output Structure
```
logs/
└── experiment_arr{rate}_{service_type}_{expiration_type}/
    └── theta_{values}_{timestamp}/
        ├── experiment_config.json
        ├── aggregated_results.json
        ├── all_runs_summary.csv
        └── run_{N}_seed_{S}/
            ├── config.json
            ├── results.json
            ├── summary.txt
            ├── theta.csv
            ├── states.csv
            └── all_costs.csv
```

### ✅ 5. Comprehensive Output Files
**Per-Run Files:**
- Complete configuration
- Detailed results (JSON)
- Human-readable summary
- Theta evolution (CSV)
- System states (CSV)
- Cost values (CSV)

**Aggregate Files:**
- Combined results from all runs
- Summary CSV for easy analysis
- Statistical summaries (mean ± std)

### ✅ 6. Timing Information
Each run tracks:
- Wall-clock execution time
- CPU time
- Simulated system time


## Quick Start

```bash
# Test (takes ~1 second)
python AutoscalerFaas/ServerlessSimulator.py --input input_test_tiny.json

# Production run
python AutoscalerFaas/ServerlessSimulator.py --input input.json
```

## Documentation

1. **QUICKSTART.md** - Get started in 30 seconds
2. **SIMULATOR_USAGE.md** - Complete usage guide
3. **ENHANCEMENT_SUMMARY.md** - Detailed technical documentation

## Usage

### Command Line
```bash
python AutoscalerFaas/ServerlessSimulator.py --input <config_file>
```

### Configuration Example
```json
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
  "max_time": 100000,
  "K": 2,
  "seeds": [1, 42, 123, 456, 789],
  "exp_per_run": 1,
  "log_dir": "logs/"
}
```

## Analysis

### Python
```python
import pandas as pd

# Load results
df = pd.read_csv('logs/experiment_*/theta_*/all_runs_summary.csv')

# Statistics
print(df.describe())
print(f"Mean cold start: {df['prob_cold'].mean():.4f}")

# Plot
df.plot(x='seed', y='prob_cold', kind='bar')
```

### Command Line (jq)
```bash
# View execution times
jq '.runs[].wall_clock_time_seconds' logs/.../aggregated_results.json

# Extract metrics
jq '.runs[] | {seed, prob_cold, prob_reject}' logs/.../aggregated_results.json
```

## Testing

Tested with:
- ✓ Single and multiple seed runs
- ✓ All output files generated correctly
- ✓ Timing information accurate
- ✓ Aggregate statistics computed
- ✓ CSV/JSON export working
- ✓ Error handling functional


## Performance

Example run (`input_test_tiny.json` with 2 seeds):
- Run 1: 0.30 seconds
- Run 2: 0.28 seconds
- Total: 0.59 seconds
- Files: 17 generated (8 per run + 3 aggregate)

## Next Steps

1. **Validation**: `python AutoscalerFaas/ServerlessSimulator.py --input input_test_tiny.json`
2. **Quick Test**: `python AutoscalerFaas/ServerlessSimulator.py --input input_example_quick.json`
3. **Production**: Edit `input.json` and run with 10+ seeds

## Support

- Help: `python AutoscalerFaas/ServerlessSimulator.py --help`
- Documentation: See `SIMULATOR_USAGE.md`
- Examples: Check provided config files
