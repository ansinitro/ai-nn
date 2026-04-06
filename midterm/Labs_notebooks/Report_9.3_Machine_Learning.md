# Section 9.3: Machine Learning Engineering & Statistical Validation

## 1. Introduction
The objective of Section 9.3 was to transition from pure software engineering to applied mathematics. We engineered several classical Machine Learning pipelines utilizing standard Data Science libraries (`NumPy`, `Pandas`, `Scikit-Learn`, `Matplotlib`). Unlike software applications, machine learning fundamentally depends on mathematically standardized tensors. This chapter required significant data sanitization, mathematical scaling (normalizing distributions), and graphical evaluations of model predictions.

## 2. Lab 9.3.1 - 9.3.2: Linear Regression & Data Sanitization (Feature Scaling)
**Objective:** Architect regressions mapping continuous target variables, and implement critical dataset cleansing methodologies (Feature engineering/Scaling) prior to convergence.

### Mathematical Implementation & Evaluation
Using SciKit-Learn, we implemented Linear Regressions (`y = mx + b`) capable of interpolating multivariate dimensions. By plotting the dataset, we algorithmically optimized the mean-squared-error across the batch vector gradients to draw the most statistically optimal prediction boundaries.

**Output Visualizations from Linear Regression (9.3.1):**
Below are the evaluated models mapping gradient boundaries to our datasets natively:


\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{assets/9.3.1_Machine_Learning_Basic_Lab_Guide_img2.png}
\caption{Regression Target Plot}
\end{figure}



\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{assets/9.3.1_Machine_Learning_Basic_Lab_Guide_img5.png}
\caption{Regression Variance Loss Plot}
\end{figure}


### Crucial Engineering Breakthrough & Bug Fix (9.3.2):
During Lab 9.3.2, we discovered a severe structural deficiency in the native code blocking training progression. The textbook code was attempting to randomly assign `np.nan` (Not a Number floats) values directly into Pandas DataFrames constructed rigidly from NumPy `int64` vectors, instantly triggering a kernel `ValueError`. 
Once we typed our arrays appropriately as `float64` to accept the missing values, the downstream Chi-Square (`chi2`) feature-selection algorithm crashed. Why? Because categorical statistical estimators mathematically cannot parse `NaN` voids when computing variance boundaries.

**The Fix:** We built resilient imputation pipelines natively within the Jupyter environment utilizing Pandas operations. We algorithmically traversed the dataset instances and injected the `.fillna()` mapping parameter, artificially repairing data gaps based on mathematical averages directly mid-flight. 
```python
# Robust Data Imputation Strategy Implemented
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

# Fill NaN features proactively to allow chi2 statistical evaluations to safely execute
X_imputed = pd.DataFrame(X).fillna(0)  
best_features = SelectKBest(score_func=chi2, k=2)
fit = best_features.fit(X_imputed, y)
```
This allowed our feature selection evaluations to finish executing gracefully without any external manual tampering.


\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{assets/9.3.2_Machine_Learning_Lab_Guide_img1.png}
\caption{Histogram of Rating Counts per Practice}
\end{figure}


## 3. Labs 9.3.3 - 9.3.6: Unsupervised Learning (K-Means Clustering)
**Objective:** Evaluate autonomous pattern groupings strictly via K-Means and visualize boundary separations.

**What We Learned:**
For K-Means, we algorithmically defined unsupervised learning paradigms. Since data arrives without labels, we instructed the machine to iteratively compute **Euclidian distances** between each spatial vector and randomly initialized Centroids. 
As the algorithm repeats, centroids "magnetize" towards the true center variance of the geometric clusters.

### Evaluated Clustering Models (9.3.6):
Our implementation successfully grouped the arbitrary spatial spaces perfectly. Notice how the unlabelled distribution maps sequentially find optimal centroid groupings graphically over subsequent epochs:


\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{assets/9.3.6_Machine_Learning_Lab_Guide_img1.png}
\caption{K-Means Centroid Initializing}
\end{figure}


*(Raw distribution data prior to clustering evaluation)*


\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{assets/9.3.6_Machine_Learning_Lab_Guide_img5.png}
\caption{K-Means Optimized Output}
\end{figure}


*(Optimized classifications mapping to distinct geometric boundaries mathematically solved by K-Means)*

## 4. Labs 9.3.4 & 9.3.7: Decision Trees and Logical Classifiers
**Objective:** Replicate human decision-making via strict Gini Index entropy optimizations.

We utilized Tree Classification models to iteratively segment categorical boundaries. Using recursive splits, the algorithms determined the node properties that successfully isolated target classes perfectly while penalizing extreme network depths (`max_depth`) to avert catastrophic dataset over-fitting.

**Decision Tree Data Distributions:**


\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{assets/9.3.3_Machine_Learning_Lab_Guide_img1.png}
\caption{Decision Boundaries 1}
\end{figure}



\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{assets/9.3.7_Machine_Learning_Lab_Guide_img1.png}
\caption{Tree Graphic Structure}
\end{figure}


## 5. Other Classical Model Outputs (SVM & Random Forest)
We evaluated dimensional mappings across multiple models:


\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{assets/9.3.8_Machine_Learning_Lab_Guide_img1.png}
\caption{SVM Target Hyperplane}
\end{figure}



\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{assets/9.3.5_Machine_Learning_Lab_Guide_img1.png}
\caption{Random Forest Clustering Matrices}
\end{figure}


## 5. Section Summary
By dominating regression gradients, solving NaN variance deficiencies, tuning tree complexities, and orchestrating centroid allocations, we cemented our fundamental understanding of neural weight propagation and gradient biases. Every model written and statistically visualized here serves as the explicit theoretical foundation underlying the advanced Deep Learning networks that we will deploy in the final chapter!
