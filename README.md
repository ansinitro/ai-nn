# 🎓 Defense Script: Assignment 3 (LSTM Weather Forecasting)

> **Instructor:** *“Okay, Angsar and Bekzat, present your Stage 3 assignment on Recurrent Neural Networks.”*

## 🗣️ SLIDE 1: Introduction (The Top Bun)
**You:** "Hello Zhanar Khanbekovna. Today we are presenting our implementation for Stage 3 of our Time-Series project. For this assignment, we transitioned away from classical linear models and built a **Long Short-Term Memory (LSTM)** deep neural network to predict air temperatures accurately dynamically using the Jena Climate Dataset."

---

## 🗣️ SLIDE 2: Task Requirements & Context (The Lettuce)
**You:** "In Stage 2, we utilized a Multi-Layer Perceptron (MLP). The main limitation of the MLP was that it has no organic concept of 'time'. To make the MLP work, we had to manually force lag features into separate columns.
For this stage, we implemented the LSTM. Because LSTMs process sequences sequentially, we transformed our 2D tabular data into native **3D shapes: `(samples, timesteps, features)`**. We utilized NumPy's ultra-efficient pointer slicing to generate over 400,000 distinct rolling sequences instantly in memory."

---

## 🗣️ SLIDE 3: Architecture Choice (The Meat)
> **Instructor:** *“Why did you choose an LSTM with 64 units? Did you try other versions?”*

**You:** "Yes, we did. We chose `LSTM(64)` followed by a `Dense(32)` adapter logically. Weather data reflects a natural diurnal cycle (the heating and cooling rhythms of 24 hours). 
We discovered that 64 computational units are visually sufficient to capture all short-term momentum and daily patterns accurately. If we bloated the network with 128 or 256 units, the network unnecessarily burned memory processing power and immediately started drastically overfitting to sensor noise. We also implemented `EarlyStopping` on the validation loss which successfully mitigated any overfitting."

---

## 🗣️ SLIDE 4 & 5: Experiment Results 
**You:** "Looking at our Learning Curves outputted from TensorFlow, you'll see a massive decline in Error during the very first two Epochs. This indicates the LSTM immediately mathematically grasps the core day-night cycle constraints. 
In the **Fact vs. Prediction** graph, our red dotted line securely tracks the black line. Notice how precisely the model anticipates the *turnarounds* at the maximum and minimum temperature points! Typical standard MLPs lag behind the peak point due to average smoothing. The LSTM hits the curves directly."

---

## 🗣️ SLIDE 6: Hyperparameter Experiment - The Window Size
> **Instructor:** *“How does the window length affect accuracy on this dataset?”*

**You:** "Great question. We ran a strict parallel comparison mapping window trajectories of 12, 24, and 48 timesteps (equivalent to 2, 4, and 8 human hours).
- **12 steps (2 hours):** The model occasionally lacks macro positional context. 
- **24 steps (4 hours):** This proved to be our empirical 'Sweet Spot'. It provides enough historical context to evaluate whether the temperature is falling heavily or climbing, without overloading itself.
- **48 steps (8 hours):** Extending memory too far back actually slightly *harms* performance! The distant past (8 hours ago) contains a fundamentally different temperature profile which acts mathematically as 'Noise', exhausting the LSTM's *Forget Gate*."

---

## 🗣️ SLIDE 7: Final Conclusion vs MLP (The Bottom Bun)
**You:** "In conclusion, the LSTM fundamentally outclasses our previous MLP setup functionally and theoretically. It completely eradicates the need for manual feature engineering (like computing moving averages and lags mathematically beforehand). By simply converting data into basic 3D Sequence rolling-windows, the Recurrent architecture natively calculates sequence embedding weights internally, resulting in structurally superior time-series generalizations and predictions across the board!"

**You:** "That concludes our presentation, thank you!"
