# SimFaaS-NSGD: A Serverless Simulator

SimFaaS-NSGD is a serverless computing platform simulator designed for performance analysis and autoscaling algorithm evaluation. It implements Non-Stationary Gradient Descent (NSGD) algorithms for dynamic resource allocation in serverless environments.

## Features

- **Serverless Function Simulation**: Simulates function instances with cold start, warm execution, and expiration behaviors
- **Autoscaling Algorithms**: Implements NSGD-based autoscaling algorithms with adaptive parameter optimization
- **Performance Analysis**: Comprehensive cost analysis and performance metrics collection
- **Multiple Execution Modes**: Sequential and parallel simulation capabilities
- **Configurable Workloads**: Support for different arrival patterns (exponential, constant, Pareto distributions)

## Project Structure

- `AutoscalerFaas/`: Sequential version of the serverless simulator
- `AutoscalerFaasParallel/`: Parallel version supporting multiprocessing simulation
- `costs/`: Cost analysis results from simulation runs
- `plots/`: Generated visualization plots from simulation data
- `plots_scripts/`: Python scripts for generating plots and analysis

## Requirements

- Python 3.7+
- NumPy >= 1.18.4
- Matplotlib >= 3.2.1
- Pandas >= 1.0.3
- SciPy >= 1.4.1
- tqdm >= 4.46.0

## Installation

1. Create a Python virtual environment:
```sh
python -m venv env_name
```

2. Activate the environment:
```sh
source env_name/bin/activate  # On Linux/macOS
# or
env_name\Scripts\activate     # On Windows
```

3. Install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage

### Sequential Simulation

Run the sequential version of the simulator:
```sh
python AutoscalerFaas/ServerlessSimulator.py
```

### Parallel Simulation

Run the parallel version with multiprocessing support:
```sh
python AutoscalerFaasParallel/ServerlessSimulator.py
```

### Generating Plots

Generate analysis plots using the plotting scripts:
```sh
cd plots_scripts
python plots_comparison.py  # Compare different algorithms
python cost_vs_arrival.py   # Cost vs arrival rate analysis
python plots_sensitivity.py # Sensitivity analysis
```

## Configuration

The simulator supports various configuration parameters:

- **Arrival Processes**: Exponential, Constant, or Pareto distributions
- **Service Processes**: Configurable warm and cold service times
- **Autoscaling Parameters**: NSGD learning rates, perturbation bounds, and optimization thresholds
- **System Limits**: Maximum concurrency, expiration thresholds, and simulation duration

## Output

The simulator generates:

- **Cost Files**: Detailed cost analysis results in the `costs/` directory
- **Performance Metrics**: Response times, rejection rates, and resource utilization
- **Visualization Plots**: Performance comparison charts in the `plots/` directory

