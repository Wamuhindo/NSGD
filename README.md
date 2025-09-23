# PPO-LSTM for comparison with Non-Stochastic Gradient Descent for Serverless Computing

This project implements an autoscaling framework based on reinforcement learning approach using Proximal Policy Optimization (PPO) with LSTM networks to compare with the proposed Non-Stochastic Gradient Descent (NSGD).

## Overview

The system combines:
- **PPO-LSTM**: Deep reinforcement learning for decision making in dynamic serverless environments
- **Serverless Simulation**: Comprehensive simulation framework for function-as-a-service platforms based on enhanced version of SimFaas
- **Auto-scaling**: Intelligent resource management and scaling decisions

## Project Structure

```
PPO_LSTM_NSGD/
├── main.py                     # Main entry point
├── Algorithm.py                # Auto-scaling algorithm implementation
├── ServerlessSimulator.py     # Serverless platform simulator
├── FunctionInstance.py        # Function instance management
├── SimProcess.py              # Process simulation utilities
├── Utility.py                 # Utility functions
├── utils.py                   # Common utilities and enums
├── classes/                   # Core classes
│   ├── Environment.py         # RL environment
│   ├── NetworkPPOLstm.py     # PPO-LSTM network architecture
│   ├── Logger.py             # Logging utilities
│   ├── Callbacks.py          # Training callbacks
│   └── serialization.py      # Model serialization
├── configs/                   # Configuration files
│   ├── exp_config.json       # Experiment configuration
│   ├── env_config.json       # Environment configuration
│   └── ray_config_*.json     # Ray RLlib configurations
├── costs/                     # Cost function data
├── plots/                     # Visualization and plotting scripts
└── *.csv                     # Experimental results
```

## Dependencies

### Core Requirements
- Python 3.8+
- Ray RLlib
- NumPy
- SciPy
- TensorFlow/PyTorch (for neural networks)
- Matplotlib (for visualization)

### Git Submodules
This project depends on the `RL4CC` (Reinforcement Learning for Cloud Computing) module:
- **RL4CC**: Provides the core reinforcement learning framework and training utilities

## Installation

1. **Clone the repository with submodules:**
   ```bash
   git clone --recurse-submodules <repository-url>
   git checkout ppo_lstm
   ```

   If you've already cloned without submodules:
   ```bash
   git checkout ppo_lstm
   git submodule update --init --recursive
   ```

2. **Install Python dependencies:**
   ```bash
   pip install ray[rllib] numpy scipy matplotlib tensorflow
   # or
   pip install -r requirements.txt  # if requirements.txt exists
   ```

3. **Set up Ray cluster (optional):**
   ```bash
   ray start --head --port=6379
   ```

## Usage

### Basic Training

Run the main training script:
```bash
python main.py
```

The system will:
1. Initialize Ray cluster connection
2. Load configuration from `configs/exp_config.json`
3. Start PPO-LSTM training with the serverless environment
4. Save results and checkpoints

### Configuration

Modify the configuration files in the `configs/` directory:

- **`exp_config.json`**: Experiment settings, algorithm choice, logging
- **`env_config.json`**: Environment parameters, observation/action spaces
- **`ray_config_*.json`**: Ray RLlib hyperparameters

### Simulation Parameters

Key simulation parameters in `env_config.json`:
```json
{
  "min_replicas": 1,
  "max_concurency": 50,
  "sampling_window": 200,
  "stats_window": 10000
}
```

## Algorithms

### NSGD Auto-scaling Algorithm

The Natural Stochastic Gradient Descent algorithm (`Algorithm.py`) implements:
- Adaptive threshold adjustment
- Cost-based optimization
- State-aware scaling decisions

Key parameters:
- `N`: Maximum number of instances
- `k_delta`, `k_gamma`: Learning rate parameters
- `theta_init`: Initial threshold
- `tau`: Time averaging parameter

### PPO-LSTM Network

The PPO-LSTM implementation features:
- Long Short-Term Memory for temporal dependencies
- Proximal Policy Optimization for stable learning
- Custom network architecture in `classes/NetworkPPOLstm.py`

## Results and Analysis

### Data Files
- `sensitivity_data.csv`: Sensitivity analysis results
- `resultsMixedAll.csv`: Mixed algorithm comparison results
- `resultsOurAlgoAll.csv`: Our algorithm performance results

### Visualization
Use scripts in the `plots/` directory:
```bash
python plots/plots_comparison.py      # Algorithm comparison
python plots/plots_sensitivity.py    # Sensitivity analysis
python plots/cost_vs_arrival.py      # Cost vs arrival rate analysis
```

## Cost Functions

The `costs/` directory contains various cost function configurations:
- Different minimum function execution times (5, 10, 15, 35 minutes)
- With/without deadline constraints (`dt` suffix)
- Pareto-optimal configurations (`pareto` suffix)

## Performance Metrics

The system tracks:
- **Resource utilization**: CPU, memory, GPU usage
- **Response times**: Cold start, warm start latencies
- **Cost efficiency**: Operational costs vs performance
- **Rejection rates**: Request dropping under high load

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## Troubleshooting

### Common Issues

1. **Ray connection errors:**
   ```bash
   ray stop  # Stop existing Ray processes
   ray start --head --port=6379
   ```

2. **Missing RL4CC module:**
   ```bash
   git submodule update --init --recursive
   ```

3. **Memory issues during training:**
   - Reduce `sampling_window` in environment config
   - Lower `max_concurrency` parameter
   - Use smaller neural network architectures

### Debugging

Enable verbose logging by modifying the Logger configuration in `main.py`:
```python
error = Logger(stream=sys.stderr, verbose=2, is_error=True)
```
