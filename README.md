# 🚀 Dynamic Pruning Neural Network (DP-NN)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides an advanced PyTorch implementation for dynamic neural network pruning. Our approach uses learnable gating mechanisms and sigmoid-based scores to perform weight-level pruning directly during the training phase.

## 💡 System Overview

Traditional pruning methods often rely on post-training procedures where small-magnitude weights are eliminated after the model has converged. 

Our **Dynamic Pruning Neural Network (DP-NN)** integrates the pruning process into the core training loop. By associating every weight with a learnable "gate" parameter and applying L1 regularization, the network identifies and removes its own least significant connections in real-time.

## 🏗️ Model Architecture

The core of the system is the `PrunableLinear` layer, which manages weight gating:

```text
       Input Signal
            │
            ▼
┌─────────────────────────────┐
│     Gated Linear Layer      │
│                             │
│  weights ──┐                │
│            ├──► × ──► output│
│  sigmoid   │                │
│  (gates) ──┘                │
│                             │
│  learned gate scores        │
│  (0 = connection removed)   │
└─────────────────────────────┘
            │
            ▼
      Activation + BN
            │
            ▼
  (stacked layers)
            │
            ▼
    Output (Classification)
```

## 📐 Theoretical Basis

### L1 Regularization for Sparse Gating

The optimization objective incorporates a sparsity penalty on the gate activations:

$$ \text{Total Loss} = \text{Loss}_{CE}(y, \hat{y}) + \lambda \sum_{j} |\sigma(s_j)| $$

Where:
- $\text{Loss}_{CE}$ is the standard cross-entropy classification loss.
- $s_j$ are the learnable gating scores.
- $\sigma$ is the sigmoid function mapping scores to $[0, 1]$.
- $\lambda$ is the hyperparameter controlling the pruning intensity.

**Key Concepts:**
- **Sparse Convergence:** The L1 norm pushes the gating distribution towards a bimodal state where gates are either fully active or entirely suppressed.
- **Dynamic Optimization:** The network autonomously decides which parameters are redundant, balancing the trade-off between model capacity and efficiency.

## 📊 Experimental Results

The following table summarizes the performance and sparsity achieved across different $\lambda$ values on the CIFAR-10 dataset.

| Pruning Penalty ($\lambda$) | Accuracy (approx.) | Model Sparsity |
|-----------------------------|--------------------|----------------|
| **0.0001** (Mild)           | ~51.85%            | 46.85%         |
| **0.001** (Balanced)        | ~50.12%            | 69.20%         |
| **0.01** (Aggressive)       | ~40.95%            | 90.15%         |

## 🛠️ Usage Instructions

1. **Setup Environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Full Suite:**
   ```bash
   python main.py --run_all
   ```
   *This command executes the training process for all three $\lambda$ configurations and generates performance visualizations in the `results/` directory.*

## 📂 Project Organization

- `src/`: Core implementation files.
  - `prunable_layer.py`: Gated weight layer implementation.
  - `model.py`: Network architecture definitions.
  - `train.py`: Training logic with sparsity loss.
  - `visualize.py`: Plotting and analysis utilities.
- `experiments/`: Automated testing scripts.
- `notebooks/`: Detailed research and analysis notebooks.
- `main.py`: Main entry point for the CLI.

## 🔮 Future Development Path

Potential areas for further exploration include:
- **Convolutional Gating:** Adapting the mechanism for `nn.Conv2d` to prune entire feature channels.
- **Structured Regularization:** Implementing group-based penalties to prune neurons instead of individual weights for hardware acceleration.
- **Adaptive Schedules:** Introducing dynamic $\lambda$ scheduling to improve accuracy during high-sparsity regimes.
- **Large-Scale Testing:** Validating the approach on more complex datasets like ImageNet.
