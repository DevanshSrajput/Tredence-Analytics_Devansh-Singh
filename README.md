# Self-Pruning Neural Network — CIFAR-10

Tredence AI Engineering Internship Case Study · April 2026

A feed-forward network that learns to prune its own weights **during training**
via per-weight learnable sigmoid gates and an L1 sparsity regulariser.

## How it works

Each weight `w_ij` has a companion `gate_score_ij` (learnable parameter).
Forward pass: `w_ij_effective = w_ij * sigmoid(gate_score_ij)`.
Loss: `CrossEntropy + λ · Σ sigmoid(gate_scores)`.
High λ → more sparsity. Network self-selects which weights to prune.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python prunable_net.py
```

Trains three times with `λ ∈ [1e-3, 1e-2, 1e-1]` (50 epochs each, Adam lr=1e-3,
batch size 128). CIFAR-10 is downloaded automatically to `./data/`.

## Results

| Lambda | Test Accuracy | Sparsity | Gate Distribution |
|--------|:-------------:|:--------:|:-----------------:|
| `1e-4` | 60.57% | 0.00% | ![lam=1e-4](plots/gate_dist_1e-04.png) |
| `1e-3` | 61.09% | 94.42% | ![lam=1e-3](plots/gate_dist_1e-03.png) |
| `1e-2` | 60.54% | 99.99% | ![lam=1e-2](plots/gate_dist_1e-02.png) |
| `1e-1` | 60.82% | 100.00% | ![lam=1e-1](plots/gate_dist_1e-01.png) |

See [report.md](report.md) for full method explanation and analysis.

## Repo structure

```
prunable_net.py   # PrunableLinear, SelfPruningNet, training loop, plot utility
report.md         # Method, results table, analysis
plots/            # Gate distribution histograms per lambda
requirements.txt  # torch, torchvision, numpy, matplotlib
```
