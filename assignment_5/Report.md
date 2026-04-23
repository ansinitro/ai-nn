---
title: "Assignment 5 -- Denoising Autoencoder and Classification Robustness"
author:
  - "Angsar Shaumen"
  - "Bekzat Sundetkhan"
date: "April 23, 2026"
geometry: margin=2.5cm
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{hyperref}
  - \hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
  - \usepackage{amsmath}
  - \setlength{\parskip}{6pt}
---

# Abstract

We implement a fully-connected denoising autoencoder trained on clean sensor windows from the Jena Climate dataset, and evaluate its effect on the robustness of the 1-D CNN classifier from Assignment 4. Gaussian noise with five standard deviation levels is injected into test windows to simulate sensor corruption. We compare the CNN classifier's performance (accuracy, precision, recall, F1-score) on **clean**, **noisy**, and **autoencoder-denoised** variants. Results show that the autoencoder consistently recovers classification quality under moderate to high noise, recovering up to **+2.6% accuracy** and **+2.7% F1** at the highest tested noise level.

---

# 1. Introduction

Neural networks deployed in production environments must tolerate imperfect inputs caused by sensor failures, communication noise, and environmental interference. A common mitigation strategy is to place a denoising preprocessing module before the main model. Autoencoders, trained solely on clean data, learn the manifold of valid inputs and can project noisy observations back onto that manifold.

In this assignment we:

1. Build a fully-connected autoencoder with a 64-dimensional bottleneck and train it to reconstruct clean time-series windows.
2. Inject Gaussian noise at five levels ($\sigma \in \{0.05,\ 0.10,\ 0.20,\ 0.30,\ 0.50\}$) into the test set and measure the degradation of the CNN classifier.
3. Pass the noisy test data through the autoencoder and re-evaluate the classifier, quantifying the recovery.
4. Conduct a noise-level sweep to visualise the crossover point at which denoising becomes beneficial.

---

# 2. Dataset and Pre-processing

**Dataset:** Jena Climate dataset (Max Planck Institute for Biogeochemistry). The same pre-processed version used in Assignments 1-4 is reused here.

\begin{center}
\begin{tabular}{ll}
\toprule
\textbf{Property} & \textbf{Value} \\
\midrule
Total rows & 420,533 \\
Raw features & 26 numeric \\
Target column & T (degC) \\
Binary threshold (median) & 9.42 degC \\
Window size & 24 timesteps (= 4 h) \\
Train / Val / Test split & 70\% / 15\% / 15\% (chronological) \\
\bottomrule
\end{tabular}
\end{center}

All features were standardised with StandardScaler fitted **only on the training set** to prevent data leakage.

**Noise injection:** Gaussian noise $\mathcal{N}(0,\,\sigma^2)$ was added independently to every feature of every test window **after** standardisation, keeping noise in the normalised feature space:
$$\tilde{x} = x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

---

# 3. Autoencoder Architecture

The autoencoder is a **symmetric fully-connected** (dense) network. Input and output dimensions equal the flattened window: $24\;\text{steps} \times 26\;\text{features} = 624$.

\begin{center}
\begin{tabular}{lll}
\toprule
\textbf{Layer} & \textbf{Units} & \textbf{Activation} \\
\midrule
Input & 624 & --- \\
Encoder Dense 1 & 256 & ReLU \\
Encoder Dense 2 & 128 & ReLU \\
\textbf{Bottleneck} & \textbf{64} & \textbf{ReLU} \\
Decoder Dense 1 & 128 & ReLU \\
Decoder Dense 2 & 256 & ReLU \\
Output & 624 & Linear \\
\bottomrule
\end{tabular}
\end{center}

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam ($lr = 10^{-3}$)
- **Batch size:** 1,024
- **Max epochs:** 30 with early stopping (patience = 5, monitoring val\_loss)
- **Best training MSE:** 0.013910
- **Best validation MSE:** 0.013918

The near-identical train and validation loss confirms that the autoencoder generalises well without overfitting.

---

# 4. Experimental Results

## 4.1 Autoencoder Training

The model converged within approximately 15 epochs. The learning curves (Figure 1) show closely tracking train and validation loss throughout, with no signs of overfitting. Early stopping triggered after the validation loss plateaued.

## 4.2 Signal Reconstruction

Visual inspection of randomly selected test windows (Figure 2) confirms that the denoised signal closely tracks the ground-truth clean signal while the raw noisy signal deviates noticeably. Reconstruction MSE on the test set: **0.013918**.

## 4.3 Classification Comparison at noise = 0.1

The Assignment 4 CNN classifier was applied to three test-set variants at $\sigma = 0.1$:

\begin{center}
\begin{tabular}{lllll}
\toprule
\textbf{Variant} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
\midrule
\textbf{Clean} & \textbf{0.9937} & \textbf{0.9935} & \textbf{0.9930} & \textbf{0.9933} \\
Noisy ($\sigma=0.1$) & 0.9859 & 0.9842 & 0.9854 & 0.9848 \\
Denoised ($\sigma=0.1$) & 0.9819 & 0.9795 & 0.9817 & 0.9806 \\
\bottomrule
\end{tabular}
\end{center}

At this low noise level the CNN is already highly robust (-0.78 pp accuracy from clean to noisy). The autoencoder's own reconstruction error slightly exceeds the noise damage, so denoised accuracy is marginally lower than noisy at this noise level. The crossover point is observed at approximately $\sigma \approx 0.15$.

## 4.4 Noise-Level Sweep

\begin{center}
\begin{tabular}{lllllll}
\toprule
$\sigma$ & Noisy Acc & Den. Acc & $\Delta$ Acc & Noisy F1 & Den. F1 & $\Delta$ F1 \\
\midrule
0.05 & 0.9912 & 0.9840 & -0.72 pp & 0.9905 & 0.9828 & -0.77 pp \\
0.10 & 0.9853 & 0.9826 & -0.27 pp & 0.9843 & 0.9813 & -0.30 pp \\
\textbf{0.20} & \textbf{0.9744} & \textbf{0.9788} & \textbf{+0.44 pp} & \textbf{0.9725} & \textbf{0.9773} & \textbf{+0.48 pp} \\
\textbf{0.30} & \textbf{0.9619} & \textbf{0.9733} & \textbf{+1.14 pp} & \textbf{0.9592} & \textbf{0.9713} & \textbf{+1.21 pp} \\
\textbf{0.50} & \textbf{0.9351} & \textbf{0.9609} & \textbf{+2.58 pp} & \textbf{0.9308} & \textbf{0.9580} & \textbf{+2.72 pp} \\
\bottomrule
\end{tabular}
\end{center}

Bold rows indicate regimes where denoising provides a **net positive gain**. The benefit grows monotonically with noise intensity (Figure 3).

---

# 5. Discussion

**Why is denoising harmful at low noise?**
The autoencoder compresses the 624-dimensional input to a 64-dimensional bottleneck, introducing its own reconstruction error (MSE $\approx$ 0.0139). At $\sigma = 0.1$, the SNR is high enough for the CNN to tolerate the noise directly; the autoencoder's compression artefacts are comparably damaging. Once the noise energy exceeds the reconstruction error (around $\sigma = 0.15$), denoising becomes strictly beneficial.

**Compression ratio and bottleneck choice.**
A 64-unit bottleneck achieves approximately $10\times$ compression ($624 \to 64$). Reducing the bottleneck would lower reconstruction quality; increasing it would reduce the denoising effect. 64 units was selected empirically.

**Practical implication.**
In a real-world IoT pipeline, if expected sensor noise is low ($\sigma < 0.15$ in normalised scale), the overhead of denoising may not be justified. At higher noise levels -- which correspond to realistic sensor faults, transmission errors, or environmental interference -- the autoencoder provides a clear and measurable benefit.

---

# 6. Conclusion

A symmetric fully-connected autoencoder with a 64-dimensional bottleneck successfully learns the manifold of clean Jena Climate sensor windows and serves as an effective denoising pre-processor for the downstream CNN binary classifier. Key findings:

- The autoencoder converges to a stable validation MSE of **0.013918** in approximately 15 epochs.
- At low noise ($\sigma \leq 0.1$), the CNN is inherently robust and denoising provides no benefit.
- At moderate and high noise ($\sigma \geq 0.2$), denoising **consistently improves** both accuracy and F1.
- At the highest tested noise ($\sigma = 0.5$), denoising recovers **+2.58 pp accuracy** and **+2.72 pp F1**.

These results demonstrate that autoencoders are a practical and lightweight defence against sensor noise, scaling in effectiveness with corruption intensity.

---

# 7. Deliverables

\begin{center}
\begin{tabular}{ll}
\toprule
\textbf{File} & \textbf{Description} \\
\midrule
assignment\_5\_solution.py & Full Python solution \\
autoencoder.h5 & Saved trained autoencoder (4.9 MB) \\
results\_summary.json & All numeric results \\
figures/01\_ae\_learning\_curves.png & Train/val loss curves \\
figures/02\_signal\_comparison.png & Clean vs. noisy vs. denoised signals \\
figures/03\_classification\_comparison.png & Accuracy and F1 bar chart \\
figures/04\_detailed\_metrics.png & 4-metric comparison \\
figures/05\_reconstruction\_error.png & Per-sample MSE histogram \\
figures/06\_noise\_levels.png & Accuracy and F1 across noise-level sweep \\
presentation.html & 8-slide HTML presentation \\
Report.pdf & This document \\
\bottomrule
\end{tabular}
\end{center}

---

# References

1. Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P.A. (2008). *Extracting and composing robust features with denoising autoencoders*. ICML 2008.
2. Klambauer, G., Unterthiner, T., Mayr, A., and Hochreiter, S. (2017). *Self-normalizing neural networks*. NeurIPS 2017.
3. Chollet, F. (2021). *Deep Learning with Python*, 2nd ed. Manning Publications.
4. Dataset: Jena Climate Dataset. Max Planck Institute for Biogeochemistry. https://www.bgc-jena.mpg.de
