Here is the explanation for **Data Science Fundamentals**. This section bridges the gap between raw algorithms and building a robust, working solution.

---

### 1. Model Evaluation Metrics
**Goal:** Quantify how good the model is. Accuracy is rarely enough.

**Basic Terms:**
*   **TP (True Positive):** Predicted Yes, Actual Yes.
*   **TN (True Negative):** Predicted No, Actual No.
*   **FP (False Positive):** Predicted Yes, Actual No (Type I Error / "False Alarm").
*   **FN (False Negative):** Predicted No, Actual Yes (Type II Error / "Missed").

**The Metrics:**
1.  **Accuracy:** $\frac{TP + TN}{Total}$.
    *   *Problem:* If 99% of data is Class 0, a model that predicts "All 0s" has 99% accuracy but is useless.
2.  **Precision:** $\frac{TP}{TP + FP}$.
    *   *Intuition:* "Of all the ones I labeled as Positive, how many are actually Positive?"
    *   *Use case:* Spam detection (You don't want to classify important email as spam).
3.  **Recall (Sensitivity):** $\frac{TP}{TP + FN}$.
    *   *Intuition:* "Of all the actual Positives, how many did I find?"
    *   *Use case:* Cancer detection (You must not miss a sick patient).
4.  **F1-Score:** $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$.
    *   *Intuition:* The harmonic mean. It forces both Precision and Recall to be decent.

**Python Example:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1]

print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"Precision: {precision_score(y_true, y_pred)}") # High precision = few false alarms
print(f"Recall: {recall_score(y_true, y_pred)}")       # High recall = few missed cases
print(f"F1: {f1_score(y_true, y_pred)}")
```

---

### 2. Confusion Matrix & ROC Curves
**Goal:** Visualize performance beyond a single number.

#### A. Confusion Matrix
A table showing where the model is confused.

**Visual:**
```text
                 Actual
               +       -
             _____________
Pred   +    | TP  |  FP   |
       -    | FN  |  TN   |
             -------------
```

#### B. ROC Curve & AUC
**ROC (Receiver Operating Characteristic):** A plot of TPR (Recall) vs. FPR (False Positive Rate) at various **thresholds**.
*   Instead of saying "If prob > 0.5, then 1", we check what happens if threshold is 0.1, 0.2... 0.9.
*   **AUC (Area Under Curve):** A single number from 0 to 1.
    *   **0.5:** Random guessing (Diagonal line).
    *   **1.0:** Perfect classifier.

**Visual:**
```text
   1.0 |    .-------  <-- Perfect Model (AUC=1)
   T   |   /
   P   |  /  .----    <-- Good Model (AUC=0.8)
   R   | /  /
       |/  /          <-- Random Guessing (AUC=0.5)
   0.0 |/_/__________
       0.0   FPR    1.0
```

**Python Example:**
```python
from sklearn.metrics import confusion_matrix, roc_auc_score

print(confusion_matrix(y_true, y_pred))
# For ROC/AUC, we need probabilities, not hard 0/1 predictions
probs = [0.1, 0.9, 0.4, 0.2, 0.8, 0.95] 
print(roc_auc_score(y_true, probs))
```

---

### 3. Underfitting vs. Overfitting Theory
**Goal:** Finding the "Goldilocks" zone (The Bias-Variance Tradeoff).

1.  **Underfitting (High Bias):**
    *   The model is too simple. It cannot capture the pattern.
    *   *Symptom:* High Training Error, High Test Error.
    *   *Analogy:* Studying for a math exam by only memorizing "1+1=2".
    *   *Fix:* Increase model complexity (add features, deeper trees).

2.  **Overfitting (High Variance):**
    *   The model is too complex. It memorizes noise and outliers.
    *   *Symptom:* **Low** Training Error, **High** Test Error.
    *   *Analogy:* Memorizing the exact answers to the practice test, but failing the real exam because the questions changed slightly.
    *   *Fix:* Regularization (L1/L2), simplify model, more data.

**Visual:**
```text
   Underfit         Good Fit         Overfit
    (Line)          (Curve)         (Squiggle)
    
      *                *               *--*
     /               _/              _/    \
    /               /               /       \
   *               *               *         *
```

---

### 4. Cross-Validation Practice
**Goal:** Reliable evaluation. One "Train/Test Split" might be lucky or unlucky.

**Principle (K-Fold):**
1.  Split data into $K$ parts (folds).
2.  Train on $K-1$ parts, Test on 1 part.
3.  Repeat $K$ times (rotating the test part).
4.  Average the score.

**Python Example:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# cv=5 means 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Scores: {scores}")
print(f"Mean Accuracy: {scores.mean()}") # Trust this number!
```

---

### 5. Hyperparameter Tuning Practice
**Goal:** Finding the best settings for the model (that the model cannot learn itself).
*   *Parameters:* Weights ($w, b$) learned during training.
*   *Hyperparameters:* Settings chosen *before* training (K in K-NN, Depth in Trees, Alpha in Lasso).

1.  **Grid Search:** Brute-force try every combination. (Thorough but slow).
2.  **Random Search:** Try random combinations. (Faster, usually finds a good result).

**Python Example:**
```python
from sklearn.model_selection import GridSearchCV

# We want to find the best 'n_neighbors' for KNN
param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X, y)

print(f"Best Params: {grid.best_params_}")
print(f"Best Score: {grid.best_score_}")
```

---

### 6. Feature Engineering
**Goal:** Creating better inputs to help the model learn. "Garbage in, Garbage out."

1.  **Encoding Categorical Data:**
    *   Models need numbers, not strings ("Red", "Blue").
    *   **One-Hot Encoding:** Creates new columns: `is_Red`, `is_Blue`. (Use for nominal data).
    *   **Label Encoding:** Assigns 0, 1, 2. (Use for ordinal data like "Low, Med, High").
2.  **Interaction Features:**
    *   Combining features. E.g., `Area = Width * Length`.
3.  **Text/Date Handling:**
    *   Extracting `DayOfWeek` from a Date.
    *   Counting word frequency (TF-IDF) from Text.

**Python Example:**
```python
import pandas as pd

df = pd.DataFrame({'Color': ['Red', 'Blue', 'Red'], 'Size': ['S', 'M', 'L']})

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Color'])
print(df_encoded)
# Output columns: Size, Color_Blue, Color_Red
```

---

### 7. Data Processing
**Goal:** Preparing raw data for the algorithm.

1.  **Handling Missing Values (Imputation):**
    *   Drop rows? (Only if you have lots of data).
    *   Fill with Mean/Median? (Standard approach).
    *   Fill with constant (-1)?
2.  **Feature Scaling (Normalization/Standardization):**
    *   **Crucial for:** Distance-based algorithms (K-NN, SVM, K-Means).
    *   If "Salary" is 100,000 and "Age" is 50, the distance calculation will only care about Salary. Scaling brings them to the same range.
    *   **StandardScaler:** Shifts data to Mean = 0, Std Dev = 1.
    *   **MinMaxScaler:** Squeezes data between 0 and 1.

**Python Example:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Data with missing value (NaN) and different scales
data = [[100000, 25], [50000, np.nan], [75000, 30]]

# 1. Fill missing age with mean
imputer = SimpleImputer(strategy='mean')
data_filled = imputer.fit_transform(data)

# 2. Scale features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_filled)

print(data_scaled) 
# Now all numbers are roughly around -1 to 1 range
```
