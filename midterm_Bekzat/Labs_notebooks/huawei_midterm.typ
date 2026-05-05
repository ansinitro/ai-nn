#set page(paper: "a4", margin: 2.5cm)
#set text(font: "Libertinus Serif", size: 12pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 24pt, weight: "bold")[Deep Learning and AI Frameworks]
  #v(1em)
  #text(size: 18pt)[Midterm Implementation Report]
  #v(2em)
  #text(size: 14pt)[*Author:* Sundetkhan Bekzat]
  #v(1em)
  #text(size: 14pt)[*Instructor:* Akhmetova Zhanar]
  #v(1em)
  #text(size: 14pt)[Astana IT University]
  #v(2em)
]

#outline(indent: true)
#pagebreak()
= Huawei HCIA-AI V3.5 Midterm Labs

*Student:* Sundetkhan Bekzat  
*Instructor:* Akhmetova Zhanar  
*Subject:* Artificial Intelligence and Neural Networks

== Overview

This folder contains an independent implementation of the midterm lab set for Huawei HCIA-AI V3.5 sections 9.2, 9.3, and 9.4. The notebooks are written to run locally with synthetic or built-in datasets so the work can be checked without downloading large archives.

== Contents

- `9.2 Python`: four notebooks covering data types, control flow, file I/O, regex, decorators, and a small treasury ledger.
- `9.3 Machine Learning`: nine notebooks covering regression, feature engineering, recommendation, credit scoring, survival prediction, clustering, flower classification, user segmentation, and sentiment analysis.
- `9.4 Deep Learning`: five notebooks covering tensor basics, dense digit classification, transfer-learning workflow, residual connections, and TextCNN-style sentiment features.
- `assets`: generated figures used by the reports.
- `huawei_midterm.tex` and `huawei_midterm.pdf`: written report files for review.

== How To Run

From the repository root:

```bash
uv run --with jupyter --with numpy --with pandas --with scikit-learn --with matplotlib jupyter nbconvert --to notebook --execute --inplace "midterm_Bekzat/Labs_notebooks/9.3 Machine Learning/9.3.1 Machine Learning Basic Lab Guide/9.3.1 Machine Learning Basic Lab Guide.ipynb"
```

The notebooks avoid interactive input and use deterministic random seeds. MindSpore is optional in section 9.4; when it is unavailable, the notebooks use NumPy or scikit-learn fallbacks while preserving the same workflow idea.

#figure(image("assets/course_progress_bekzat.png", width: 85%), caption: [Course progress])

#pagebreak()
= Section 9.2: Python Programming Foundations

*Student:* Sundetkhan Bekzat

== Purpose

Section 9.2 builds the programming base required for later machine learning work. The notebooks cover Python syntax, core containers, loops, functions, object-oriented design, file processing, regular expressions, decorators, and exception handling.

== Completed Labs

- `9.2.1`: data types, deterministic branch checks, loops, Fibonacci generation, class-level state, and standard library inspection.
- `9.2.2`: text, CSV, and JSON file operations using temporary folders; a small database-like adapter replaces an external MySQL dependency.
- `9.2.3`: regex extraction and validation, iterable handling, and a timing decorator.
- `9.2.4`: treasury ledger with transactions, balances, category totals, formatted statements, and safe input parsing.

== Engineering Notes

The implementation avoids interactive `input()` calls so every notebook can be executed by `nbconvert`. External services are replaced with local adapters where the lab concept is more important than a real server connection.

== Result

The section produces reproducible terminal outputs and reusable utility patterns that prepare the later tabular and text-processing notebooks.

#pagebreak()
= Section 9.3: Machine Learning Practice

*Student:* Sundetkhan Bekzat

== Purpose

Section 9.3 moves from programming exercises into classical machine learning. The notebooks use NumPy, Pandas, Matplotlib, and scikit-learn to demonstrate preprocessing, model fitting, evaluation, and visualization.

== Main Work

- Regression and classification baselines are implemented with train/test splits and metrics.
- Missing credit features are repaired with median imputation before chi-square and wrapper-based selection.
- Recommendation is demonstrated through user-item similarity and SVD factors.
- Credit default prediction includes class weighting and threshold tuning.
- Titanic, flower, e-commerce, and customer-review tasks use compact local datasets or synthetic data.

== Visual Evidence

#figure(image("assets/ml_linear_regression_bekzat.png", width: 85%), caption: [Linear regression])

#figure(image("assets/feature_missing_values_bekzat.png", width: 85%), caption: [Missing values])

#figure(image("assets/kmeans_segments_bekzat.png", width: 85%), caption: [K-Means segmentation])

#figure(image("assets/retail_matrix_bekzat.png", width: 85%), caption: [Retail matrix])

== Result

All machine learning notebooks are deterministic and small enough for local execution. They keep the same lab objectives while using independent variable names, data construction, and explanations.

#pagebreak()
= Section 9.4: Deep Learning and AI Framework Concepts

*Student:* Sundetkhan Bekzat

== Purpose

Section 9.4 covers the deep learning part of the midterm. Because large MindSpore datasets and checkpoints are not guaranteed to be available locally, the notebooks are designed with executable fallbacks that preserve the architectural ideas.

== Completed Topics

- Tensor and batching concepts with optional MindSpore detection.
- Dense handwritten-digit classification using the built-in digits dataset.
- Transfer-learning workflow using synthetic image tensors and a frozen feature extractor.
- Residual block shape alignment and checkpoint-like parameter inspection.
- TextCNN-style sentiment analysis using n-gram pooling and a dense classification head.

== Visual Evidence

#figure(image("assets/digits_predictions_bekzat.png", width: 85%), caption: [Digits predictions])

#figure(image("assets/transfer_flow_bekzat.png", width: 85%), caption: [Transfer samples])

#figure(image("assets/text_sentiment_matrix_bekzat.png", width: 85%), caption: [Text sentiment matrix])

== Result

The notebooks are runnable without network downloads and still show the key engineering concerns of deep learning labs: tensor shapes, feature extraction, classification heads, residual paths, and text feature pooling.

#pagebreak()
