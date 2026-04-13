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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential, load_model

SEED = 42
WINDOW = 24
BATCH_SIZE = 1024
MAX_EPOCHS = 15

np.random.seed(SEED)
random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH = os.path.join(
    REPO_DIR, "assignment_1", "data", "processed", "processed_data.csv"
)
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
MODEL_PATH = os.path.join(SCRIPT_DIR, "cnn_classifier.h5")
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "results_summary.json")
COMPARISON_CSV = os.path.join(SCRIPT_DIR, "comparison_metrics.csv")
SEARCH_CSV = os.path.join(SCRIPT_DIR, "cnn_search_metrics.csv")


def ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def build_window_means(data: np.ndarray, window: int) -> np.ndarray:
    source = data[:-1].astype(np.float64, copy=False)
    csum = np.vstack(
        [
            np.zeros((1, source.shape[1]), dtype=np.float64),
            np.cumsum(source, axis=0, dtype=np.float64),
        ]
    )
    means = (csum[window:] - csum[:-window]) / window
    return means.astype(np.float32)


def build_sequence_array(data: np.ndarray, window: int) -> np.ndarray:
    view = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=0)
    return np.transpose(view, (0, 2, 1)).astype(np.float32, copy=False)


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


def build_cnn(
    window: int, n_features: int, filters: int, kernel_size: int
) -> Sequential:
    model = Sequential(
        [
            Input(shape=(window, n_features)),
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            ),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    return model


def plot_class_balance(positive_ratio: float) -> None:
    negative_ratio = 1.0 - positive_ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [negative_ratio, positive_ratio]
    labels = ["Class 0", "Class 1"]
    colors = ["#CBD5E1", "#2563EB"]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.set_title("Binary target balance")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2%}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "01_class_balance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_learning_curves(history: tf.keras.callbacks.History) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        history.history["loss"], label="Train loss", color="#2563EB", linewidth=2
    )
    axes[0].plot(
        history.history["val_loss"], label="Val loss", color="#DC2626", linewidth=2
    )
    axes[0].set_title("CNN loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary crossentropy")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(
        history.history["accuracy"],
        label="Train accuracy",
        color="#2563EB",
        linewidth=2,
    )
    axes[1].plot(
        history.history["val_accuracy"],
        label="Val accuracy",
        color="#059669",
        linewidth=2,
    )
    axes[1].set_title("CNN accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "02_cnn_learning_curves.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("CNN confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="#0F172A",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "03_confusion_matrix.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_probability_timeline(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    plot_size = min(300, len(y_true))
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.step(
        range(plot_size),
        y_true[:plot_size],
        where="mid",
        label="True class",
        color="#0F172A",
        linewidth=1.5,
    )
    ax.plot(
        range(plot_size),
        y_prob[:plot_size],
        label="CNN probability",
        color="#2563EB",
        linewidth=2,
    )
    ax.axhline(
        0.5, color="#DC2626", linestyle="--", linewidth=1.2, label="Decision threshold"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Test window index")
    ax.set_ylabel("Probability")
    ax.set_title("CNN probabilities on the first 300 test windows")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "04_probability_timeline.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ["#CBD5E1", "#60A5FA", "#34D399", "#F59E0B", "#8B5CF6"]

    for ax, metric in zip(axes, metrics):
        ax.bar(
            comparison_df["model"],
            comparison_df[metric],
            color=colors[: len(comparison_df)],
        )
        ax.set_ylim(0, 1)
        ax.set_title(metric.upper())
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=15)
        for index, value in enumerate(comparison_df[metric]):
            ax.text(
                index,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "Classification comparison: baseline, transfer, and CNN", fontsize=15, y=1.01
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "05_model_comparison.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_search_results(search_df: pd.DataFrame) -> None:
    pivot = search_df.pivot(index="filters", columns="kernel_size", values="val_f1")
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(
        range(len(pivot.columns)), labels=[str(item) for item in pivot.columns]
    )
    ax.set_yticks(range(len(pivot.index)), labels=[str(item) for item in pivot.index])
    ax.set_xlabel("Kernel size")
    ax.set_ylabel("Filters")
    ax.set_title("Validation F1 for CNN hyperparameters")
    for row_index, row_value in enumerate(pivot.index):
        for col_index, col_value in enumerate(pivot.columns):
            val = pivot.loc[row_value, col_value]
            ax.text(
                col_index,
                row_index,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="white",
            )
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "06_hyperparameter_search.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def evaluate_assignment_2_transfer(
    features: pd.DataFrame, target: pd.Series, threshold: float
) -> tuple[dict[str, float], dict[str, float]]:
    model_path = os.path.join(REPO_DIR, "assignment_2", "mlp_best.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")

    split_train = int(len(features) * 0.7)
    split_val = int(len(features) * 0.85)

    scaler = StandardScaler()
    scaler.fit(features.iloc[:split_train])

    x_test = scaler.transform(features.iloc[split_val:]).astype(np.float32)
    y_test = target.iloc[split_val:].to_numpy(dtype=np.float32)

    model = load_model(model_path, compile=False)
    preds = model.predict(x_test, verbose=0).reshape(-1)
    binary_true = (y_test > threshold).astype(int)
    binary_pred = (preds > threshold).astype(int)

    return compute_classification_metrics(
        binary_true, binary_pred
    ), compute_regression_metrics(y_test, preds)


def evaluate_assignment_3_transfer(
    features: pd.DataFrame, target: pd.Series, threshold: float
) -> tuple[dict[str, float], dict[str, float], int]:
    model_path = os.path.join(REPO_DIR, "assignment_3", "lstm_best.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")

    model = load_model(model_path, compile=False)
    window = int(model.input_shape[1])
    n_sequences = len(features) - window
    split_train = int(n_sequences * 0.7)
    split_val = int(n_sequences * 0.85)

    scaler = StandardScaler()
    scaler.fit(features.iloc[: split_train + window - 1])
    scaled = scaler.transform(features).astype(np.float32)

    data_source = scaled[:-1]
    test_sequences = build_sequence_array(data_source[split_val:], window)
    y_test = target.iloc[window + split_val :].to_numpy(dtype=np.float32)

    preds = model.predict(test_sequences, batch_size=512, verbose=0).reshape(-1)
    binary_true = (y_test > threshold).astype(int)
    binary_pred = (preds > threshold).astype(int)

    return (
        compute_classification_metrics(binary_true, binary_pred),
        compute_regression_metrics(y_test, preds),
        window,
    )


def main() -> None:
    ensure_dirs()

    print(f"Loading processed data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    target_col = "Appliances" if "Appliances" in df.columns else "T (degC)"
    binary_col = "HighConsumption" if target_col == "Appliances" else "HighTemperature"

    threshold = float(df[target_col].median())
    df[binary_col] = (df[target_col] > threshold).astype(np.int8)

    feature_source = df.drop(
        columns=[target_col, "Date Time", binary_col], errors="ignore"
    )
    feature_frame = feature_source.select_dtypes(include=[np.number, bool]).astype(
        np.float32
    )
    legacy_feature_frame = feature_source.select_dtypes(include=[np.number]).astype(
        np.float32
    )
    target_series = df[target_col].astype(np.float32)
    target_binary = df[binary_col].astype(np.int8).to_numpy()

    n_sequences = len(df) - WINDOW
    split_train = int(n_sequences * 0.7)
    split_val = int(n_sequences * 0.85)

    scaler = StandardScaler()
    scaler.fit(feature_frame.iloc[: split_train + WINDOW - 1])
    scaled_features = scaler.transform(feature_frame).astype(np.float32)

    sequence_targets = target_binary[WINDOW:].astype(np.float32)
    train_y = sequence_targets[:split_train]
    val_y = sequence_targets[split_train:split_val]
    test_y = sequence_targets[split_val:].astype(int)

    positive_ratio = float(target_binary.mean())
    train_positive_ratio = float(train_y.mean())
    minority_ratio = min(train_positive_ratio, 1.0 - train_positive_ratio)
    use_class_weights = minority_ratio < 0.4

    class_weight = None
    if use_class_weights:
        zeros = float(np.sum(train_y == 0))
        ones = float(np.sum(train_y == 1))
        total = zeros + ones
        class_weight = {
            0: total / (2.0 * zeros),
            1: total / (2.0 * ones),
        }

    sequence_source = scaled_features[:-1]
    train_data = sequence_source[: split_train + WINDOW - 1]
    val_data = sequence_source[split_train : split_val + WINDOW - 1]
    test_data = sequence_source[split_val:]

    train_ds = make_sequence_dataset(train_data, train_y, WINDOW, shuffle=True)
    val_ds = make_sequence_dataset(val_data, val_y, WINDOW, shuffle=False)
    test_ds = make_sequence_dataset(
        test_data, test_y.astype(np.float32), WINDOW, shuffle=False
    )

    print(f"Dataset rows: {len(df):,}")
    print(f"Numeric feature count: {feature_frame.shape[1]}")
    print(f"Target column used: {target_col}")
    print(f"Median threshold: {threshold:.4f}")
    print(f"Positive class share: {positive_ratio:.4f}")
    print(f"Class weights enabled: {bool(class_weight)}")

    plot_class_balance(positive_ratio)

    baseline_features = build_window_means(scaled_features, WINDOW)
    x_train_baseline = baseline_features[:split_train]
    x_test_baseline = baseline_features[split_val:]

    logistic = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=SEED,
    )
    logistic.fit(x_train_baseline, train_y.astype(int))
    logistic_probs = logistic.predict_proba(x_test_baseline)[:, 1]
    logistic_pred = (logistic_probs >= 0.5).astype(int)
    logistic_metrics = compute_classification_metrics(test_y, logistic_pred)

    search_rows: list[dict[str, float | int]] = []
    best_model = None
    best_history = None
    best_config = None
    best_score = -1.0

    for filters in (64, 128):
        for kernel_size in (3, 5):
            print(
                f"\nTraining CNN config: filters={filters}, kernel_size={kernel_size}"
            )
            model = build_cnn(WINDOW, feature_frame.shape[1], filters, kernel_size)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=MAX_EPOCHS,
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=3,
                        restore_best_weights=True,
                    )
                ],
                class_weight=class_weight,
                verbose=1,
            )
            val_probs = model.predict(val_ds, verbose=0).reshape(-1)
            val_pred = (val_probs >= 0.5).astype(int)
            val_metrics = compute_classification_metrics(val_y.astype(int), val_pred)
            search_row = {
                "filters": filters,
                "kernel_size": kernel_size,
                "epochs_ran": len(history.history["loss"]),
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "best_val_loss": float(np.min(history.history["val_loss"])),
            }
            search_rows.append(search_row)
            print(f"Validation metrics: {val_metrics}")
            if val_metrics["f1"] > best_score:
                best_score = val_metrics["f1"]
                best_model = model
                best_history = history
                best_config = {"filters": filters, "kernel_size": kernel_size}

    if best_model is None or best_history is None or best_config is None:
        raise RuntimeError("CNN hyperparameter search did not produce a model.")

    print(f"\nBest CNN config: {best_config}")
    best_model.save(MODEL_PATH)
    print(f"Saved best CNN model to: {MODEL_PATH}")

    cnn_probs = best_model.predict(test_ds, verbose=0).reshape(-1)
    cnn_pred = (cnn_probs >= 0.5).astype(int)
    cnn_metrics = compute_classification_metrics(test_y, cnn_pred)

    plot_learning_curves(best_history)
    plot_confusion_matrix(test_y, cnn_pred)
    plot_probability_timeline(test_y, cnn_probs)

    assignment_2_metrics, assignment_2_regression = evaluate_assignment_2_transfer(
        legacy_feature_frame, target_series, threshold
    )
    assignment_3_metrics, assignment_3_regression, assignment_3_window = (
        evaluate_assignment_3_transfer(legacy_feature_frame, target_series, threshold)
    )

    comparison_rows = [
        {"model": "LogReg", **logistic_metrics},
        {"model": "A2 MLP", **assignment_2_metrics},
        {"model": f"A3 LSTM (w={assignment_3_window})", **assignment_3_metrics},
        {"model": "A4 CNN", **cnn_metrics},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    plot_model_comparison(comparison_df)

    search_df = (
        pd.DataFrame(search_rows)
        .sort_values(["filters", "kernel_size"])
        .reset_index(drop=True)
    )
    plot_search_results(search_df)

    comparison_df.to_csv(COMPARISON_CSV, index=False)
    search_df.to_csv(SEARCH_CSV, index=False)

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "features": int(feature_frame.shape[1]),
            "target_column": target_col,
            "binary_column": binary_col,
            "median_threshold": threshold,
            "window": WINDOW,
            "positive_ratio": positive_ratio,
            "used_class_weights": bool(class_weight),
        },
        "baseline": {"logistic_regression": logistic_metrics},
        "cnn_search": search_rows,
        "cnn_best_config": best_config,
        "cnn_test_metrics": cnn_metrics,
        "assignment_2_transfer": {
            "classification": assignment_2_metrics,
            "regression": assignment_2_regression,
        },
        "assignment_3_transfer": {
            "window": assignment_3_window,
            "classification": assignment_3_metrics,
            "regression": assignment_3_regression,
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== Final classification comparison ===")
    print(comparison_df.to_string(index=False))
    print("\n=== Regression reference from previous assignments ===")
    print(
        pd.DataFrame(
            [
                {"model": "Assignment 2 MLP", **assignment_2_regression},
                {
                    "model": f"Assignment 3 LSTM (w={assignment_3_window})",
                    **assignment_3_regression,
                },
            ]
        ).to_string(index=False)
    )
    print(f"\nSummary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
