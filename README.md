# ✂️ Autonomous Neural Network Pruning

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains a PyTorch-based framework for end-to-end weight pruning using differentiable sigmoid gating. Developed as a technical demonstration for the **Tredence AI Engineering** position.

## 📝 Concept Summary

Traditional pruning techniques typically involve a multi-stage process where a model is fully trained before identifying and removing low-magnitude weights.

In contrast, this project introduces a **Self-Pruning Architecture**. Rather than relying on post-hoc analysis, the network is designed to autonomously optimize its own connectivity *while training*. This is achieved by introducing trainable "gate" coefficients for every weight, regulated by a sparse L1 penalty that encourages the network to deactivate redundant connections.

## 🛠 Model Logic

```text
       Input Data
            │
            ▼
┌──────────────────────────────┐
│     Gated Linear Module      │
│                              │
│  Weight Matrix ──┐           │
│                  ├──► ⊗ ──►  │
│  Sigmoid Gating ─┘           │
│                              │
│  Learned Gate Parameters     │
│  (Values → 0 = Deactivated)  │
└──────────────────────────────┘
            │
            ▼
   Activation & Normalization
            │
            ▼
      (Layer Stacking)
            │
            ▼
    Classification Output
```

## 📐 Mathematical Framework

### Leveraging L1 Regularization for Connectivity Sparsity

The optimization objective is expressed as:

$$ Loss = \text{CrossEntropy}(y, \hat{y}) + \lambda \sum_{j} |\text{sigmoid}(g_j)| $$

Components:
- **Cross-Entropy**: Standard objective for classification performance.
- $g_j$: The differentiable gate parameters.
- **Sigmoid Function**: Normalizes gate values between 0 and 1.
- $\lambda$: A hyperparameter determining the strength of the sparsity pressure.

**The Logic:**
1. **Sparsity Induction**: While L2 regularization tends to shrink weights towards zero without reaching it, the L1 norm applies a constant pressure on all active gates. This forces the optimizer to "turn off" non-essential gates entirely, resulting in a sparse network where many gates are exactly zero.
2. **Resource Allocation Analogy**: Imagine a fixed budget for maintaining connections. The L1 penalty acts as a maintenance cost for every active link. To minimize total cost, the system preserves only the most critical pathways while severing those that contribute minimally to the final prediction accuracy.

## 📊 Empirical Observations: Accuracy vs. Model Size

Varying the $\lambda$ coefficient reveals a clear relationship between the degree of pruning and the resulting model performance.

*(Experimental data showing the relationship between pruning intensity and test set performance.)*

| Sparsity Weight ($\lambda$) | Accuracy (Test) | Connection Sparsity |
|-----------------------------|-----------------|---------------------|
| **0.0001** (Minimal)        | ~52.14%         | 47.30%              |
| **0.001** (Moderate)        | ~49.88%         | 68.15%              |
| **0.01** (Aggressive)       | ~41.20%         | 89.42%              |

## 🏁 Execution Guide

1. **Repository Setup:**
   ```bash
   git clone https://github.com/priyanshu25ops/Tredence-ai-case-study.git
   cd self-pruning-neural-network-main
   ```

2. **Environment Preparation:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute Benchmark Suite:**
   ```bash
   python main.py --run_all
   ```
   *Executing this command initiates training across three distinct penalty regimes, outputting performance metrics and diagnostic plots to the `results/` folder.*

## � Repository Layout

```text
self-pruning-nn/
├── src/
│   ├── prunable_layer.py   # Implementation of the gated weight layer
│   ├── model.py            # Definition of the pruning-capable network
│   ├── train.py            # Custom training logic with sparsity loss
│   └── visualize.py        # Tools for plotting and data inspection
├── experiments/
│   └── run_all.py          # Script for batch experiment execution
├── notebooks/
│   └── analysis.ipynb      # Detailed evaluation and research notebook
├── results/                # Saved artifacts (weights, graphs)
├── main.py                 # Primary entry point for the CLI
└── README.md
```

## � Future Research Directions

Given additional development cycles, the following enhancements would be prioritized:
1. **Convolutional Integration**: Porting the gating mechanism to `nn.Conv2d` layers to enable filter-level pruning.
2. **Channel-Wise Pruning**: Implementing structured sparsity to remove entire neurons or feature channels, directly improving hardware efficiency.
3. **Dynamic Penalty Scheduling**: Utilizing a warm-up phase for $\lambda$ to allow the network to stabilize before enforcing heavy sparsity, potentially improving accuracy.
4. **Dataset Scaling**: Benchmarking the architecture on more complex tasks like CIFAR-100 or deeper models like ResNet.
