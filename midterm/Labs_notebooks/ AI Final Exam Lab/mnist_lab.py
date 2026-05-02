"""
Huawei AI Certification Lab: Handwritten Digit Recognition (MNIST)
Implements: data loading, visualization, DNN model, CNN model, prediction visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Dropout, Flatten, Dense,
)
from tensorflow.keras.models import Sequential, load_model


# ---------- 1.2.1 Dataset Acquisition ----------
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print("First 5 labels:", y_train[:5])
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_test shape :", x_test.shape, "y_test shape :", y_test.shape)

n_classes = 10
y_train_oh = keras.utils.to_categorical(y_train, n_classes)
y_test_oh = keras.utils.to_categorical(y_test, n_classes)
print("One-hot sample:\n", y_train_oh[:5])


# ---------- 1.2.2 Data Visualization ----------
plt.figure(figsize=(6, 6))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
plt.suptitle("First 25 MNIST training images")
plt.tight_layout()
plt.savefig("viz_first25.png", dpi=120)
plt.close()
print("Saved viz_first25.png")


# ---------- Preprocess for DNN (flatten + normalize) ----------
X_train_dnn = x_train.reshape(60000, 784).astype("float32") / 255.0
X_test_dnn = x_test.reshape(10000, 784).astype("float32") / 255.0


# ---------- 1.2.3 DNN Construction ----------
dnn = Sequential([
    Dense(512, activation="relu", input_dim=784),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(n_classes, activation="softmax"),
])
dnn.summary()

dnn.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizers.Adam(0.001),
    metrics=["accuracy"],
)

dnn_history = dnn.fit(
    X_train_dnn, y_train_oh,
    batch_size=128, epochs=10, verbose=2,
    validation_data=(X_test_dnn, y_test_oh),
)

dnn_score = dnn.evaluate(X_test_dnn, y_test_oh, verbose=0)
print("DNN Test loss:", dnn_score[0])
print("DNN Test accuracy:", dnn_score[1])

dnn.save("final_DNN_model.h5")
print("Saved final_DNN_model.h5")


# ---------- 1.2.4 CNN Construction ----------
X_train_cnn = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test_cnn = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3), padding="same",
               activation="relu", input_shape=(28, 28, 1)))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Conv2D(64, kernel_size=(3, 3), padding="valid", activation="relu"))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(n_classes, activation="softmax"))
cnn.summary()

cnn.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

cnn_history = cnn.fit(
    X_train_cnn, y_train_oh,
    epochs=5, batch_size=128, verbose=2,
    validation_data=(X_test_cnn, y_test_oh),
)

cnn_score = cnn.evaluate(X_test_cnn, y_test_oh, verbose=0)
print("CNN Test Loss:", cnn_score[0])
print("CNN Test Accuracy:", cnn_score[1])

cnn.save("final_CNN_model.h5")
print("Saved final_CNN_model.h5")


# ---------- Training-history curves ----------
def plot_history(history, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], "o-", label="train")
    axes[0].plot(history.history["val_loss"], "s-", label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], "o-", label="train")
    axes[1].plot(history.history["val_accuracy"], "s-", label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved {fname}")


plot_history(dnn_history, "DNN", "viz_dnn_history.png")
plot_history(cnn_history, "CNN", "viz_cnn_history.png")


# ---------- Model comparison bar chart ----------
fig, ax = plt.subplots(figsize=(6, 4))
names = ["DNN", "CNN"]
losses = [dnn_score[0], cnn_score[0]]
accs = [dnn_score[1], cnn_score[1]]
xs = np.arange(len(names))
w = 0.35
b1 = ax.bar(xs - w / 2, losses, w, label="test loss", color="#e76f51")
b2 = ax.bar(xs + w / 2, accs, w, label="test accuracy", color="#2a9d8f")
ax.set_xticks(xs); ax.set_xticklabels(names)
ax.set_title("DNN vs CNN — test metrics")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for b in (*b1, *b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
            f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("viz_model_compare.png", dpi=120)
plt.close()
print("Saved viz_model_compare.png")


# ---------- Confusion matrices ----------
def plot_confusion(model, X, y_true, title, fname):
    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion matrix — {title}")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved {fname}")
    return y_pred


dnn_pred = plot_confusion(dnn, X_test_dnn, y_test, "DNN", "viz_dnn_confusion.png")
cnn_pred = plot_confusion(cnn, X_test_cnn, y_test, "CNN", "viz_cnn_confusion.png")


# ---------- Per-class accuracy bar chart (CNN) ----------
per_class = np.zeros(n_classes)
for c in range(n_classes):
    mask = y_test == c
    per_class[c] = (cnn_pred[mask] == c).mean()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(n_classes), per_class, color="#264653")
ax.set_ylim(0.9, 1.0)
ax.set_title("CNN — per-class accuracy")
ax.set_xlabel("digit"); ax.set_ylabel("accuracy")
ax.set_xticks(range(n_classes))
for i, v in enumerate(per_class):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("viz_cnn_per_class.png", dpi=120)
plt.close()
print("Saved viz_cnn_per_class.png")


# ---------- Misclassified samples (CNN) ----------
wrong_idx = np.where(cnn_pred != y_test)[0]
n_show = min(25, len(wrong_idx))
if n_show > 0:
    rows = int(np.ceil(n_show / 5))
    fig, ax = plt.subplots(rows, 5, figsize=(8, 1.8 * rows))
    ax = np.atleast_2d(ax).flatten()
    for k in range(rows * 5):
        ax[k].axis("off")
    for k, i in enumerate(wrong_idx[:n_show]):
        ax[k].imshow(X_test_cnn[i].reshape(28, 28), cmap="Greys")
        ax[k].set_title(f"true {y_test[i]} / pred {cnn_pred[i]}",
                        fontsize=8, color="red")
    plt.suptitle(f"CNN misclassifications (showing {n_show} of {len(wrong_idx)})")
    plt.tight_layout()
    plt.savefig("viz_cnn_misclassified.png", dpi=120)
    plt.close()
    print(f"Saved viz_cnn_misclassified.png ({len(wrong_idx)} total errors)")


# ---------- 1.2.5 Prediction Result Visualization ----------
new_model = load_model("final_CNN_model.h5")
new_model.summary()


def res_visual(n):
    probs = new_model.predict(X_test_cnn[0:n], verbose=0)
    final_opt_a = np.argmax(probs, axis=1)

    fig, ax = plt.subplots(nrows=int(n / 5), ncols=5, figsize=(8, 1.8 * (n // 5)))
    ax = ax.flatten()
    print("Prediction results of the first {} images:".format(n))
    for i in range(n):
        print(final_opt_a[i], end=",")
        if (i + 1) % 5 == 0:
            print()
        img = X_test_cnn[i].reshape((28, 28))
        ax[i].imshow(img, cmap="Greys")
        ok = final_opt_a[i] == y_test[i]
        ax[i].set_title(
            f"pred {final_opt_a[i]} / true {y_test[i]}",
            fontsize=8, color="green" if ok else "red",
        )
        ax[i].axis("off")
    print("First {} images of the test set".format(n))
    plt.tight_layout()
    plt.savefig("viz_predictions.png", dpi=120)
    plt.close()
    print("Saved viz_predictions.png")


res_visual(20)
print("DONE")
