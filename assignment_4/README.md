# 🎓 Defense Script: Assignment 4 (1D CNN Time-Series Classification)

> **Instructor:** *“Okay, Angsar and Bekzat, present your Stage 4 assignment on Convolutional Neural Networks.”*

## 🗣️ SLIDE 1: Introduction (The Top Bun)
**You:** "Hello Zhanar Khanbekovna. Today we are presenting our implementation for Stage 4 of our Time-Series project. After exploring Recurrent Neural Networks (LSTM), we transitioned to **1D Convolutional Neural Networks (CNNs)**. While CNNs are predominantly famous for processing 2D images, applying a 1D convolution across a sliding time-series window allows the network to extract local temporal patterns extremely efficiently."

---

## 🗣️ SLIDE 2: Task Requirements & Context (The Lettuce)
**You:** "Unlike previous stages where we performed regression to predict exact temperatures, here we reframed the problem as a **Binary Classification task**: Will the temperature be High or Low? We calculated a global median threshold (9.42 °C), splitting our target class exactly 50/50. 
We continued using the same mathematical 3D data structure `(samples, 24 timesteps, 26 features)` that we introduced for the LSTM, passing 24-hour situational windows into the network."

---

## 🗣️ SLIDE 3: Architecture & Hyperparameter Search (The Meat)
> **Instructor:** *“Why use a CNN instead of an LSTM for time sequences? How did you choose the parameters?”*

**You:** "Because Convolutional filters can slide across the entire 24-step window simultaneously, calculating data patterns in parallel! LSTMs process sequences sequentially, creating a massive training bottleneck. 
To optimize the network, we automated a Grid Search, testing filter counts of 64 and 128, and Kernel Sizes (filter widths) of 3 and 5. The absolute **Best Configuration was 64 filters with a Kernel Size of 3**. Scaling up to 128 filters substantially increased the computational parameters with zero validation improvement, essentially just burning memory without producing higher accuracy."

---

## 🗣️ SLIDE 4: Experiment Results vs Linear Baseline
**You:** "To ensure our deep learning complex wasn't overkill, we established a strict baseline using classical Logistic Regression. It achieved a high 97.8% F1 Score, indicating our multi-variate features already contain strong linear correlations. 
However, our finalized CNN reliably dominated the baseline, achieving a remarkable **99.3% accuracy and F1 score** on unseen test sequences! Looking at our Confusion Matrix output, misclassifications (false positives and false negatives) were virtually eradicated."

---

## 🗣️ SLIDE 5: Final Comparison vs MLP and LSTM (The Bottom Bun)
**You:** "As a final conclusive experiment, we utilized Transfer Learning logic on our legacy networks from Stage 2 (MLP) and Stage 3 (LSTM), adapting them for this classification task to battle our new CNN head-to-head.
While all the Deep Neural architectures achieved over 99% accuracy on this balanced dataset, the **CNN establishes the structural 'Sweet Spot'**. It avoids the sequential processing lag of the LSTM—enabling drastically faster training times—while still natively grasping local contiguous relationships that the flat MLP struggles with. For fast, localized window classifications, 1D CNNs are highly optimal!"

**You:** "That concludes our final model presentation, thank you!"
