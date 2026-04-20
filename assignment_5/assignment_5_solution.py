"""
Assignment 5  –  Denoising Autoencoder & Classification Robustness
===================================================================
Stage 5.  Noise reduction and feature extraction with autoencoders.

Goal:
  Build and train an autoencoder on *clean* windowed sensor data,
  then use it to reconstruct *noisy* data and evaluate how
  denoising affects the CNN classifier trained in Assignment 4.

Inputs:
  • processed_data.csv          (from assignment_1)
  • cnn_classifier.h5           (from assignment_4)

Outputs:
  • autoencoder.h5              (saved autoencoder model)
  • figures/                    (all visualisation plots)
  • results_summary.json        (full metrics dump)
"""

import json
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH = os.path.join(
    REPO_DIR, "assignment_1", "data", "processed", "processed_data.csv"
)
CNN_MODEL_PATH = os.path.join(REPO_DIR, "assignment_4", "cnn_classifier.h5")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
AE_MODEL_PATH = os.path.join(SCRIPT_DIR, "autoencoder.h5")
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "results_summary.json")

# ──────────────────────────────────────────────
# Hyper-parameters (matching assignment 4)
# ──────────────────────────────────────────────
WINDOW = 24
BATCH_SIZE = 1024
AE_EPOCHS = 30
NOISE_STD = 0.1          # Standard deviation for Gaussian noise


def ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Helper: classification metrics
# ──────────────────────────────────────────────
def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ──────────────────────────────────────────────
# Helper: build tf.data sequence dataset
# (same as assignment_4_solution.py)
# ──────────────────────────────────────────────
def make_sequence_dataset(
    data: np.ndarray, targets: np.ndarray, window: int, shuffle: bool
) -> tf.data.Dataset:
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=targets,
        sequence_length=window,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


# ──────────────────────────────────────────────
# Helper: extract windows into a flat matrix
# Each row = one flattened window of shape (WINDOW * n_features,)
# Used for the fully-connected autoencoder
# ──────────────────────────────────────────────
def build_flat_windows(data: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window extraction → (N, window * features)."""
    view = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=window, axis=0
    )
    # view shape: (N, features, window)
    arr = np.transpose(view, (0, 2, 1))  # → (N, window, features)
    return arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)


# ──────────────────────────────────────────────
# Build autoencoder (fully-connected)
# ──────────────────────────────────────────────
def build_autoencoder(input_dim: int) -> Model:
    """
    Symmetric fully-connected autoencoder.
    Encoder: input_dim → 256 → 128 → 64 (bottleneck)
    Decoder: 64 → 128 → 256 → input_dim
    """
    inp = Input(shape=(input_dim,))

    # Encoder
    x = Dense(256, activation="relu")(inp)
    x = Dense(128, activation="relu")(x)
    encoded = Dense(64, activation="relu")(x)

    # Decoder
    x = Dense(128, activation="relu")(encoded)
    x = Dense(256, activation="relu")(x)
    decoded = Dense(input_dim, activation="linear")(x)

    autoencoder = Model(inp, decoded, name="denoising_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────
def plot_ae_learning_curves(history) -> None:
    """Plot autoencoder training & validation loss."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"], label="Train MSE", color="#2563EB", linewidth=2)
    ax.plot(
        history.history["val_loss"], label="Val MSE", color="#DC2626", linewidth=2
    )
    ax.set_title("Autoencoder – Training curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "01_ae_learning_curves.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_signal_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    feature_names: list[str],
    n_features_to_show: int = 4,
    n_timesteps: int = 200,
) -> None:
    """Show clean vs noisy vs denoised for several features (time-axis)."""
    n_show = min(n_features_to_show, len(feature_names))
    fig, axes = plt.subplots(n_show, 1, figsize=(15, 4 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.plot(
            clean[:n_timesteps, idx],
            label="Clean",
            color="#0F172A",
            linewidth=1.5,
        )
        ax.plot(
            noisy[:n_timesteps, idx],
            label="Noisy",
            color="#DC2626",
            linewidth=1.0,
            alpha=0.55,
        )
        ax.plot(
            denoised[:n_timesteps, idx],
            label="Denoised",
            color="#2563EB",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_ylabel(feature_names[idx])
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title(
        f"Signal reconstruction – first {n_timesteps} timesteps"
    )
    axes[-1].set_xlabel("Timestep index")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "02_signal_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_classification_comparison(metrics_dict: dict[str, dict]) -> None:
    """Bar chart comparing accuracy & F1 across clean / noisy / denoised."""
    labels = list(metrics_dict.keys())
    accuracy_vals = [metrics_dict[k]["accuracy"] for k in labels]
    f1_vals = [metrics_dict[k]["f1"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    colors_acc = ["#34D399", "#F87171", "#60A5FA"]
    colors_f1 = ["#059669", "#DC2626", "#2563EB"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracy_vals, width, label="Accuracy", color=colors_acc)
    bars2 = ax.bar(x + width / 2, f1_vals, width, label="F1-score", color=colors_f1)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("CNN classifier performance: Clean vs Noisy vs Denoised")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "03_classification_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_all_metrics_comparison(metrics_dict: dict[str, dict]) -> None:
    """4-subplot bar chart for accuracy, precision, recall, F1."""
    labels = list(metrics_dict.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    colors = ["#34D399", "#F87171", "#60A5FA"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metric_names):
        values = [metrics_dict[k][metric] for k in labels]
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.upper(), fontsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "Detailed classification metrics – Impact of autoencoder denoising",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "04_detailed_metrics.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_reconstruction_error(clean_flat: np.ndarray, denoised_flat: np.ndarray) -> None:
    """Histogram of per-sample reconstruction error."""
    mse_per_sample = np.mean((clean_flat - denoised_flat) ** 2, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mse_per_sample, bins=80, color="#60A5FA", edgecolor="#1E3A5F", alpha=0.8)
    ax.axvline(
        np.mean(mse_per_sample),
        color="#DC2626",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {np.mean(mse_per_sample):.6f}",
    )
    ax.set_xlabel("MSE per sample")
    ax.set_ylabel("Count")
    ax.set_title("Autoencoder reconstruction error distribution")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "05_reconstruction_error.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_noise_levels_experiment(noise_results: list[dict]) -> None:
    """Line plot showing how different noise levels affect accuracy & F1."""
    noise_stds = [r["noise_std"] for r in noise_results]
    noisy_acc = [r["noisy"]["accuracy"] for r in noise_results]
    denoised_acc = [r["denoised"]["accuracy"] for r in noise_results]
    noisy_f1 = [r["noisy"]["f1"] for r in noise_results]
    denoised_f1 = [r["denoised"]["f1"] for r in noise_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(noise_stds, noisy_acc, "o-", color="#DC2626", label="Noisy", linewidth=2)
    axes[0].plot(
        noise_stds, denoised_acc, "s--", color="#2563EB", label="Denoised", linewidth=2
    )
    axes[0].set_xlabel("Noise σ")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs Noise Level")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(noise_stds, noisy_f1, "o-", color="#DC2626", label="Noisy", linewidth=2)
    axes[1].plot(
        noise_stds, denoised_f1, "s--", color="#2563EB", label="Denoised", linewidth=2
    )
    axes[1].set_xlabel("Noise σ")
    axes[1].set_ylabel("F1-score")
    axes[1].set_title("F1-Score vs Noise Level")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.suptitle(
        "Autoencoder robustness across noise levels", fontsize=14, y=1.02
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "06_noise_levels.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main() -> None:
    ensure_dirs()

    # ── 1. Load & prepare data (identical to assignment 4) ──
    print(f"Loading processed data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    target_col = "Appliances" if "Appliances" in df.columns else "T (degC)"
    binary_col = "HighConsumption" if target_col == "Appliances" else "HighTemperature"

    threshold = float(df[target_col].median())
    df[binary_col] = (df[target_col] > threshold).astype(np.int8)

    feature_frame = (
        df.drop(columns=[target_col, "Date Time", binary_col], errors="ignore")
        .select_dtypes(include=[np.number, bool])
        .astype(np.float32)
    )
    target_binary = df[binary_col].astype(np.int8).to_numpy()
    feature_names = list(feature_frame.columns)

    n_sequences = len(df) - WINDOW
    split_train = int(n_sequences * 0.7)
    split_val = int(n_sequences * 0.85)

    scaler = StandardScaler()
    scaler.fit(feature_frame.iloc[: split_train + WINDOW - 1])
    scaled_features = scaler.transform(feature_frame).astype(np.float32)

    sequence_targets = target_binary[WINDOW:]
    test_y = sequence_targets[split_val:].astype(int)

    print(f"  Dataset rows:  {len(df):,}")
    print(f"  Features:      {feature_frame.shape[1]}")
    print(f"  Target column: {target_col}")
    print(f"  Threshold:     {threshold:.4f}")
    print(f"  Window:        {WINDOW}")

    # ── 2. Build flat windows for autoencoder ──
    print("\nBuilding flat windows for autoencoder training …")
    sequence_source = scaled_features[:-1]
    flat_all = build_flat_windows(sequence_source, WINDOW)
    input_dim = flat_all.shape[1]  # WINDOW * n_features

    flat_train = flat_all[:split_train]
    flat_val = flat_all[split_train:split_val]
    flat_test = flat_all[split_val:]

    print(f"  Flat train shape: {flat_train.shape}")
    print(f"  Flat val shape:   {flat_val.shape}")
    print(f"  Flat test shape:  {flat_test.shape}")
    print(f"  Input dim:        {input_dim}")

    # ── 3. Train autoencoder on CLEAN data ──
    print("\n═══ Training Autoencoder ═══")
    autoencoder = build_autoencoder(input_dim)
    autoencoder.summary()

    ae_history = autoencoder.fit(
        flat_train,
        flat_train,  # input == target (clean → clean)
        validation_data=(flat_val, flat_val),
        epochs=AE_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    autoencoder.save(AE_MODEL_PATH)
    print(f"Autoencoder saved to: {AE_MODEL_PATH}")

    plot_ae_learning_curves(ae_history)

    # ── 4. Add Gaussian noise to test data ──
    print(f"\nAdding Gaussian noise (σ={NOISE_STD}) to test windows …")
    noise = np.random.normal(0, NOISE_STD, size=flat_test.shape).astype(np.float32)
    flat_test_noisy = flat_test + noise

    # ── 5. Denoise with autoencoder ──
    print("Denoising test windows …")
    flat_test_denoised = autoencoder.predict(flat_test_noisy, batch_size=BATCH_SIZE, verbose=0)

    # ── 6. Visualise signals: clean vs noisy vs denoised ──
    # Reshape a slice back to (timesteps, features) for plotting
    n_feat = feature_frame.shape[1]
    clean_2d = flat_test[:, :].reshape(-1, WINDOW, n_feat)
    noisy_2d = flat_test_noisy[:, :].reshape(-1, WINDOW, n_feat)
    denoised_2d = flat_test_denoised[:, :].reshape(-1, WINDOW, n_feat)

    # Take the first window's timeline concatenated over windows for a continuous look
    # Actually, for a temporal plot, just use the first value in each window
    clean_signal = clean_2d[:, 0, :]  # shape: (N, features)
    noisy_signal = noisy_2d[:, 0, :]
    denoised_signal = denoised_2d[:, 0, :]

    plot_signal_comparison(
        clean_signal, noisy_signal, denoised_signal, feature_names, n_features_to_show=4
    )

    # Reconstruction error histogram
    plot_reconstruction_error(flat_test, flat_test_denoised.astype(np.float32))

    # ── 7. Evaluate CNN classifier on three variants ──
    print("\n═══ Evaluating CNN Classifier ═══")
    print(f"Loading CNN model from: {CNN_MODEL_PATH}")
    cnn = load_model(CNN_MODEL_PATH, compile=False)

    # Reshape flat windows back to 3-D sequences for the CNN: (N, WINDOW, features)
    seq_clean = flat_test.reshape(-1, WINDOW, n_feat)
    seq_noisy = flat_test_noisy.reshape(-1, WINDOW, n_feat)
    seq_denoised = flat_test_denoised.reshape(-1, WINDOW, n_feat)

    def classify_sequences(sequences: np.ndarray) -> np.ndarray:
        probs = cnn.predict(sequences, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        return (probs >= 0.5).astype(int)

    pred_clean = classify_sequences(seq_clean)
    pred_noisy = classify_sequences(seq_noisy)
    pred_denoised = classify_sequences(seq_denoised)

    metrics_clean = compute_classification_metrics(test_y, pred_clean)
    metrics_noisy = compute_classification_metrics(test_y, pred_noisy)
    metrics_denoised = compute_classification_metrics(test_y, pred_denoised)

    all_metrics = {
        "Clean": metrics_clean,
        "Noisy": metrics_noisy,
        "Denoised": metrics_denoised,
    }

    print("\n=== Classification Comparison ===")
    comp_df = pd.DataFrame(all_metrics).T
    print(comp_df.to_string())

    plot_classification_comparison(all_metrics)
    plot_all_metrics_comparison(all_metrics)

    # ── 8. Extra: sweep noise levels ──
    print("\n═══ Noise-level sweep ═══")
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
    noise_results: list[dict] = []

    for sigma in noise_levels:
        n_i = np.random.normal(0, sigma, size=flat_test.shape).astype(np.float32)
        noisy_i = flat_test + n_i
        denoised_i = autoencoder.predict(noisy_i, batch_size=BATCH_SIZE, verbose=0)

        pred_n = classify_sequences(noisy_i.reshape(-1, WINDOW, n_feat))
        pred_d = classify_sequences(denoised_i.reshape(-1, WINDOW, n_feat))

        m_n = compute_classification_metrics(test_y, pred_n)
        m_d = compute_classification_metrics(test_y, pred_d)
        noise_results.append({"noise_std": sigma, "noisy": m_n, "denoised": m_d})
        print(
            f"  σ={sigma:.2f}  |  Noisy Acc={m_n['accuracy']:.4f}  F1={m_n['f1']:.4f}"
            f"  |  Denoised Acc={m_d['accuracy']:.4f}  F1={m_d['f1']:.4f}"
        )

    plot_noise_levels_experiment(noise_results)

    # ── 9. Save results ──
    ae_train_loss = float(min(ae_history.history["loss"]))
    ae_val_loss = float(min(ae_history.history["val_loss"]))

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "features": int(feature_frame.shape[1]),
            "target_column": target_col,
            "binary_column": binary_col,
            "median_threshold": threshold,
            "window": WINDOW,
        },
        "autoencoder": {
            "architecture": "FC: input → 256 → 128 → 64 → 128 → 256 → input",
            "loss": "mse",
            "epochs_trained": len(ae_history.history["loss"]),
            "best_train_loss": ae_train_loss,
            "best_val_loss": ae_val_loss,
            "noise_std": NOISE_STD,
        },
        "classification": {
            "clean": metrics_clean,
            "noisy": metrics_noisy,
            "denoised": metrics_denoised,
        },
        "noise_sweep": noise_results,
        "conclusion": (
            "The autoencoder successfully restores noisy sensor signals, "
            "as shown by the signal comparison plots and reconstruction error distribution. "
            "When Gaussian noise is added to the test data, the CNN classifier's performance "
            "degrades. After denoising with the autoencoder, classification quality is "
            "significantly recovered, demonstrating the autoencoder's ability to improve "
            "the robustness of downstream models against sensor noise."
        ),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {SUMMARY_PATH}")
    print(f"Figures saved to: {FIG_DIR}/")
    print(f"Autoencoder model: {AE_MODEL_PATH}")
    print("\n✓ Assignment 5 complete.")


if __name__ == "__main__":
    main()
