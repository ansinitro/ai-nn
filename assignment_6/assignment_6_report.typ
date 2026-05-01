#set page(paper: "a4", margin: 2.5cm)
#set text(font: "Libertinus Serif", size: 11pt)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 22pt, weight: "bold")[Deep Learning & Neural Networks] \
  #v(0.5em)
  #text(size: 16pt)[Assignment 6: Siamese Networks for Anomaly Detection] \
  #v(1em)
  #text(size: 14pt)[*Authors:* Angsar Shaumen, Bekzat Sundetkhan] \
  #text(size: 12pt)[Astana IT University] \
  #v(1em)
  #text(size: 12pt)[May 2026]
]

#v(2em)

= Introduction
This report details the implementation and evaluation of a *Siamese Network* for anomaly detection on the Jena Climate dataset. Unlike standard classifiers that predict class labels directly, the Siamese network learns a metric space where normal samples are clustered together, and anomalies are pushed away. This approach is particularly effective for unsupervised or semi-supervised anomaly detection.

= Dataset and Preprocessing
The dataset used is the Jena Climate dataset, focusing on temperature forecasting and anomaly detection. 
- *Total Rows:* 420,533
- *Features:* 26
- *Window Size:* 24 hours
- *Anomaly Definition:* A temperature value above the median threshold of 9.42°C is treated as a "positive" event (high temperature). The Siamese network is trained to recognize the "normal" (below median) distribution and flag deviations.

= Methodology

== Siamese Network Architecture
The network uses a *1D-CNN backbone* to extract features from time-series windows:
- Input: `(24, 26)` (Window size, Features)
- Layers: Two Conv1D layers (64 and 128 filters) with MaxPooling.
- Embedding: A 128-dimensional dense layer with ReLU activation.

The Siamese architecture shares these weights between two inputs, calculating the *Euclidean Distance* between their embeddings.

== Contrastive Loss
The model is trained using *Contrastive Loss*:
$ L = y dot d^2 + (1-y) dot max(m - d, 0)^2 $
Where:
- $y=1$ for similar pairs (both normal).
- $y=0$ for dissimilar pairs (normal + anomaly).
- $d$ is the Euclidean distance.
- $m$ is the margin (set to 1.0).

= Results

== Training Performance
The model was trained for 20 epochs. The training curves show a steady convergence of the contrastive loss on both training and validation sets.

#figure(
  image("figures/01_siamese_training.png", width: 80%),
  caption: [Siamese Network Training Curves (Contrastive Loss)]
)

== Distance Distributions
After training, we compute the distance between test samples and a reference set of "normal" embeddings. The distribution shows a clear separation between normal windows and anomalous windows.

#figure(
  image("figures/02_distance_distributions.png", width: 80%),
  caption: [Distance Distribution: Normal vs. Anomaly Pairs]
)

== Detection Performance
The model achieves an *AUC of 0.9253*, indicating high discriminative power. At the optimal distance threshold of *0.2655*, the metrics are as follows:

#align(center)[
#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon,
  [*Metric*], [*Value*],
  [Accuracy], [0.8561],
  [Precision], [0.8518],
  [Recall], [0.8358],
  [F1-Score], [0.8437],
  [ROC AUC], [0.9253],
)
]

#figure(
  image("figures/03_roc_curve.png", width: 60%),
  caption: [ROC Curve for Siamese Anomaly Detector]
)

== Anomaly Timeline
The following visualization shows the anomaly scores (distances) over a segment of the test set compared to ground truth labels.

#figure(
  image("figures/06_anomaly_timeline.png", width: 90%),
  caption: [Anomaly Detection Timeline vs. Ground Truth]
)

#pagebreak()

= Comparative Analysis (Assignments 2-6)

A key objective of this stage was to integrate results from all previous assignments. 

== Performance Summary Table
#align(center)[
#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 7pt,
  [*Model*], [*Accuracy*], [*Precision*], [*Recall*], [*F1-Score*],
  [MLP (A2)], [0.9992], [0.9985], [0.9999], [0.9992],
  [LSTM (A3)], [0.9950], [0.9955], [0.9937], [0.9946],
  [CNN (A4)], [0.9937], [0.9935], [0.9930], [0.9933],
  [LogReg (A4)], [0.9796], [0.9787], [0.9774], [0.9781],
  [AE+CNN (A5)], [0.9819], [0.9795], [0.9817], [0.9806],
  [Siamese (A6)], [0.8561], [0.8518], [0.8358], [0.8437],
)
]

== Visual Comparison
The bar chart below compares the primary accuracy and F1 metrics across all models developed throughout the course.

#figure(
  image("figures/04_model_comparison.png", width: 90%),
  caption: [Model Comparison: Accuracy and F1 Scores]
)

The radar chart provides a holistic view of the classification metrics (Precision, Recall, F1, Accuracy) for the top-performing models.

#figure(
  image("figures/05_radar_chart.png", width: 70%),
  caption: [Metric Radar Chart across Assignments]
)

= Conclusion
The Siamese network successfully learned a metric space where anomalies can be detected based on distance to known normal patterns. While the pure classification models (MLP, LSTM, CNN) achieved higher raw scores by learning the binary labels directly, the Siamese network provides a more robust framework for anomaly detection in scenarios where labels might be scarce or unavailable. The integration of all stages demonstrates a comprehensive understanding of deep learning applications in time-series analysis.

= References
1. Hadsell, R., Chopra, S., and LeCun, Y. (2006). *Dimensionality reduction by learning an invariant mapping*. CVPR 2006.
2. Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P.A. (2008). *Extracting and composing robust features with denoising autoencoders*. ICML 2008.
3. Chollet, F. (2021). *Deep Learning with Python*, 2nd ed. Manning Publications.
4. Dataset: Jena Climate Dataset. Max Planck Institute for Biogeochemistry. https://www.bgc-jena.mpg.de
