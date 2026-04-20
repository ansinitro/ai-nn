# Assignment 5 — Denoising Autoencoder & Classification Robustness

## Objective

Build and train a fully-connected autoencoder on **clean** windowed sensor data, then
use it to denoise **artificially corrupted** signals and measure how the denoising
step affects the CNN binary classifier from Assignment 4.

## Dataset

| Property | Value |
|---|---|
| Source | Jena Climate dataset (processed in Assignment 1) |
| Rows | 420,533 |
| Features | 26 numeric |
| Target | `T (degC)` → binary `HighTemperature` (median = 9.42) |
| Window size | 24 timesteps |

## Autoencoder Architecture

```
Input (624) → Dense(256, relu) → Dense(128, relu) → Dense(64, relu)
           → Dense(128, relu) → Dense(256, relu) → Dense(624, linear)
```

- **Loss**: MSE
- **Optimizer**: Adam
- **Training**: 30 epochs, batch size 1024, early stopping (patience 5)
- **Trained on**: clean data only (input == output)

## Pipeline

1. **Data preparation** – identical splits and scaling as Assignment 4 (70/15/15 chronological)
2. **Autoencoder training** – learns the manifold of clean windowed signals
3. **Noise injection** – Gaussian noise N(0, σ) added to test windows
4. **Denoising** – noisy windows passed through the trained autoencoder
5. **Classification evaluation** – CNN classifier applied to clean / noisy / denoised data
6. **Noise-level sweep** – σ ∈ {0.05, 0.10, 0.20, 0.30, 0.50}

## Key Results

### Classification Comparison (σ = 0.1)

| Variant | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Clean | 0.9937 | 0.9935 | 0.9930 | 0.9933 |
| Noisy | 0.9859 | 0.9842 | 0.9854 | 0.9848 |
| Denoised | 0.9819 | 0.9795 | 0.9817 | 0.9806 |

### Noise-Level Sweep (Accuracy)

| σ | Noisy Acc | Denoised Acc | Noisy F1 | Denoised F1 |
|---|---|---|---|---|
| 0.05 | 0.9912 | 0.9840 | 0.9905 | 0.9828 |
| 0.10 | 0.9853 | 0.9826 | 0.9843 | 0.9813 |
| **0.20** | **0.9744** | **0.9788** | **0.9725** | **0.9773** |
| **0.30** | **0.9619** | **0.9733** | **0.9592** | **0.9713** |
| **0.50** | **0.9351** | **0.9609** | **0.9308** | **0.9580** |

> **Key finding**: At σ ≥ 0.2, the autoencoder clearly outperforms raw noisy input.
> At low noise levels, the CNN is already robust enough that the autoencoder's own
> reconstruction error provides no benefit. The crossover point is around σ ≈ 0.15.

## Figures

| Figure | Description |
|---|---|
| `01_ae_learning_curves.png` | Autoencoder training & validation MSE loss |
| `02_signal_comparison.png` | Clean vs Noisy vs Denoised signals (4 features) |
| `03_classification_comparison.png` | Accuracy & F1 bar chart for 3 variants |
| `04_detailed_metrics.png` | 4-metric comparison (acc, prec, recall, F1) |
| `05_reconstruction_error.png` | Per-sample MSE reconstruction error histogram |
| `06_noise_levels.png` | Accuracy & F1 across noise levels |

## Outputs

- `autoencoder.h5` — saved autoencoder model
- `results_summary.json` — full metrics dump
- `figures/` — all visualisation plots

## How to Run

```bash
python3 assignment_5_solution.py
```

## Dependencies

- Python 3.12
- TensorFlow / Keras
- scikit-learn
- pandas, numpy, matplotlib
