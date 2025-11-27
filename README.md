# NSGD: Non-Stationary Stochastic Gradient Descent for Serverless Autoscaling

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A serverless computing platform simulator implementing Non-Stationary Stochastic Gradient Descent (NSGD) algorithms for intelligent autoscaling and resource management.
This simulator was used for the experiment in the paper : **Autoscaling in Serverless Platforms via Online Learning with Convergence Guarantees**.
This simulator is based on the SimFaas simulator provided [here](https://github.com/pacslab/simfaas).

## Overview

NSGD provides a comprehensive framework for simulating serverless function execution with advanced autoscaling capabilities. The simulator models realistic serverless behaviors including cold starts, warm execution, and instance expiration while optimizing resource allocation using NSGD-based algorithms.

### Key Features

- **🚀 Intelligent Autoscaling**: NSGD-based optimization for dynamic resource allocation
- **⚡ Multiple Execution Modes**: Sequential and parallel simulation support
- **📊 Comprehensive Metrics**: Cold start probability, rejection rates, cost analysis, and more
- **🔧 Flexible Configuration**: JSON-based configuration for reproducible experiments
- **🎯 Multi-Seed Support**: Run multiple independent experiments for statistical significance
- **💾 Organized Output**: Hierarchical result structure with JSON, CSV, and text formats

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Output Files](#output-files)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/NSGD.git
   cd NSGD
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

```
numpy>=1.18.4
matplotlib>=3.2.1
pandas>=1.0.3
scipy>=1.4.1
tqdm>=4.46.0
```

## Quick Start

### 1. Validate Installation (30 seconds)

```bash
python AutoscalerFaas/ServerlessSimulator.py --input input_test_tiny.json
```

### 2. Run a Full Experiment

```bash
python AutoscalerFaas/ServerlessSimulator.py --input input.json
```

### 3. Check Results

```bash
# View summary
cat logs/experiment_*/theta_*/all_runs_summary.csv

# Analyze with Python
python -c "import pandas as pd; df = pd.read_csv('logs/experiment_*/theta_*/all_runs_summary.csv'); print(df.describe())"
```

## Project Structure

```
NSGD/
├── AutoscalerFaas/              # Sequential simulator (recommended)
│   ├── ServerlessSimulator.py   # Main simulator
│   ├── Algorithm.py             # NSGD algorithms
│   ├── FunctionInstance.py      # Instance lifecycle
│   └── SimProcess.py            # Process generators
├── AutoscalerFaasParallel/      # Parallel simulator
├── plots_scripts/               # Analysis tools
├── input.json                   # Production config
├── input_example_quick.json     # Quick test
├── input_test_tiny.json         # Validation
├── QUICKSTART.md                # Quick guide
├── SIMULATOR_USAGE.md           # Detailed docs
└── requirements.txt             # Dependencies
```

## Usage

### Basic Usage

```bash
python AutoscalerFaas/ServerlessSimulator.py --input <config_file>

# Get help
python AutoscalerFaas/ServerlessSimulator.py --help
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
  "log_dir": "logs/"
}
```

For detailed configuration options, see [SIMULATOR_USAGE.md](SIMULATOR_USAGE.md).

## Output Files

```
logs/
└── experiment_arr5_Exponential_Exponential/
    └── theta_1_1_5_TIMESTAMP/
        ├── aggregated_results.json    # All runs combined
        ├── all_runs_summary.csv       # Excel-ready
        └── run_N_seed_S/              # Per-run results
            ├── results.json
            ├── theta.csv
            └── all_costs.csv
```

## Examples

### Quick Validation
```bash
python AutoscalerFaas/ServerlessSimulator.py --input input_test_tiny.json
```

### Multiple Seeds for Statistics
```bash
# Edit input.json: "seeds": [1, 42, 123, 456, 789]
python AutoscalerFaas/ServerlessSimulator.py --input input.json
```

### Analyze Results
```python
import pandas as pd

df = pd.read_csv('logs/.../all_runs_summary.csv')
print(f"Mean cold start: {df['prob_cold'].mean():.4f}")
print(f"Std deviation: {df['prob_cold'].std():.4f}")
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Citation

```bibtex
@software{nsgd_simulator,
  title={SimFaas-NSGD: Simulator with Non-Stationary Stochastic Gradient Descent for Serverless Autoscaling},
  author={Kambale, Abednego Wamuhindo and Anselmi, Jonatha and Ardagna, Danilo and Gaujal, Bruno},
  year={2025},
  url={https://github.com/Wamuhindo/NSGD}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/NSGD/issues)
- **Documentation**: See [SIMULATOR_USAGE.md](SIMULATOR_USAGE.md) and [QUICKSTART.md](QUICKSTART.md)

---

**⭐ If you find this project useful, please star it!**
