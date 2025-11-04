# Parallel vs Sequential Simulator Comparison

## Overview

The NSGD project includes two simulator versions with different use cases and interfaces:

1. **Sequential Simulator** (`AutoscalerFaas/ServerlessSimulator.py`) - **Recommended for most users**
2. **Parallel Simulator** (`AutoscalerFaasParallel/ServerlessSimulator.py`) - Advanced multiprocessing version

## Sequential Simulator (Recommended)

### Features

- ✅ **Full argparse support** with `--input` flag
- ✅ **JSON-based configuration** for easy reproducibility
- ✅ **Multiple seed support** for statistical significance
- ✅ **Organized output structure** with comprehensive files
- ✅ **Easy to use** and well-documented
- ✅ **Perfect for research** and parameter studies

### Usage

```bash
# Simple and straightforward
python AutoscalerFaas/ServerlessSimulator.py --input input.json

# Get help
python AutoscalerFaas/ServerlessSimulator.py --help
```

### When to Use

- Running systematic experiments with multiple seeds
- Parameter sweeps and sensitivity analysis
- Reproducible research experiments
- Learning and exploring the simulator
- Most production use cases

### Pros

- Easy configuration via JSON files
- Clean, organized output
- Multiple runs with different seeds
- Comprehensive documentation
- Statistical summaries automatically generated

### Cons

- Single-threaded execution (sequential)
- May be slower for very long simulations

## Parallel Simulator (Advanced)

### Features

- ⚡ **Multiprocessing support** for faster execution
- 🔧 **Low-level control** for advanced users
- 💪 **K-way parallelization** of NSGD perturbations
- 🖥️ **HPC-ready** for cluster environments

### Current Interface

**Note**: The parallel version currently uses hardcoded parameters in the `__main__` block and does **not** have argparse support. It's designed for advanced users who need to modify the code directly for high-performance scenarios.

### Usage

```bash
# Edit the file directly to change parameters
# Then run:
python AutoscalerFaasParallel/ServerlessSimulator.py
```

### When to Use

- Need maximum simulation speed
- Willing to modify code for custom scenarios
- Working with very large-scale experiments
- Advanced research requiring fine-grained control

### Pros

- Much faster for long simulations (uses multiprocessing)
- Can leverage multiple CPU cores
- Parallel perturbation evaluation in NSGD

### Cons

- No argparse/JSON configuration (yet)
- Requires code modification for different parameters
- More complex codebase
- Less user-friendly for beginners

## Comparison Table

| Feature | Sequential | Parallel |
|---------|-----------|----------|
| **Interface** | argparse + JSON | Hardcoded parameters |
| **Execution** | Single-threaded | Multi-process |
| **Speed** | Moderate | Fast |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Reproducibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Documentation** | Comprehensive | Minimal |
| **Output Organization** | Excellent | Basic |
| **Multiple Seeds** | Built-in support | Manual |
| **HPC Support** | No | Yes |
| **Recommended For** | Most users | HPC experts |

## Recommendations

### For Most Users
**Use the Sequential Simulator**

```bash
python AutoscalerFaas/ServerlessSimulator.py --input input.json
```

It's easier to use, better documented, and sufficient for most research needs.

### For HPC Environments
**Consider the Parallel Simulator**

The parallel version is optimized for high-performance computing with multiprocessing support. However, you'll need to:

1. Modify the `__main__` block in the file directly
2. Set up your parameters in code
3. Ensure your HPC environment supports multiprocessing

## Example Workflows

### Sequential: Parameter Sweep

```bash
# Easy parameter sweep
for rate in 5 10 15 20; do
  sed "s/\"arrival_rate\": 5/\"arrival_rate\": $rate/" input.json > config_${rate}.json
  python AutoscalerFaas/ServerlessSimulator.py --input config_${rate}.json
done
```

### Parallel: HPC Batch Job

```python
# Edit AutoscalerFaasParallel/ServerlessSimulator.py
# Modify parameters in __main__ section
# Lines 1516-1560

# Then run with job scheduler
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

python AutoscalerFaasParallel/ServerlessSimulator.py
```

## Future Plans

We plan to add argparse/JSON configuration support to the parallel version in a future update, making it as easy to use as the sequential version while retaining its performance benefits.

## Migration Notes

### From Parallel to Sequential

If you're currently using the parallel version but want the easier interface:

1. Extract your parameters from the `__main__` block
2. Create an `input.json` with those parameters
3. Use the sequential simulator:
   ```bash
   python AutoscalerFaas/ServerlessSimulator.py --input input.json
   ```

### From Sequential to Parallel

If you need the speed benefits:

1. Copy parameters from your `input.json`
2. Edit `AutoscalerFaasParallel/ServerlessSimulator.py`
3. Update the hardcoded values in the `__main__` block
4. Ensure multiprocessing is properly configured for your environment

## Performance Guidelines

### Sequential Simulator
- **Good for**: < 10 million simulated time units
- **Typical runtime**: Minutes to hours
- **Memory usage**: Moderate
- **CPU cores**: 1

### Parallel Simulator
- **Good for**: > 10 million simulated time units
- **Typical runtime**: Hours to days (but much faster than sequential)
- **Memory usage**: High (multiple processes)
- **CPU cores**: Multiple (K * 2 processes where K is perturbation count)

## Support

- **Sequential**: See [SIMULATOR_USAGE.md](SIMULATOR_USAGE.md) and [QUICKSTART.md](QUICKSTART.md)
- **Parallel**: Advanced users - review code directly and documentation comments
- **Questions**: Open a GitHub issue

## Summary

**Bottom line**:
- **🎯 Use Sequential** for 95% of use cases - it's easier and well-documented
- **⚡ Use Parallel** only if you need HPC-level performance and are comfortable modifying code

The sequential simulator is the recommended starting point for all users. Switch to parallel only if you have specific performance requirements.
