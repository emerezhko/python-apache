Here is a preparation plan for the Machine Learning syllabus. I have focused on the **intuition** behind the algorithms, how they work mathematically (simplified), and provided Python examples using the standard library `scikit-learn` (which is the industry standard for these tasks).

---

### 1. Linear Regression
**Goal:** Predict a continuous numerical value (e.g., house price, temperature).

**Principle:**
Imagine plotting data points on a 2D graph. Linear regression attempts to draw a **straight line** that passes as close as possible to all points.
*   **Formula:** $y = w \cdot x + b$ (where $w$ is weight/slope, $b$ is bias/intercept).
*   **Cost Function:** It tries to minimize the **Mean Squared Error (MSE)** — the sum of the squared vertical distances (residuals) between the data points and the line.

**Illustration:**
```text
      ^
      |      *
  y   |    *  /   <-- The Line (Model)
      |   * /
      |  / *      <-- The distance from point * to line is the Error
      | /
      +-------------->
             x
```

**Python Examples:**

*   **Example 1: Simple usage with Scikit-Learn**
    ```python
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # X: Feature (Size), y: Target (Price)
    X = np.array([[10], [20], [30], [40]])
    y = np.array([100, 200, 290, 410])

    model = LinearRegression()
    model.fit(X, y)

    print(f"Prediction for 50: {model.predict([[50]])[0]}") 
    # Output will be around 500
    print(f"Coefficient (slope): {model.coef_[0]}")
    ```

*   **Example 2: Multiple Linear Regression**
    ```python
    # Predicting based on 2 features (e.g., Size and Number of Rooms)
    X = [[10, 1], [20, 2], [30, 1]] 
    y = [100, 200, 280]
    
    model = LinearRegression().fit(X, y)
    print(model.predict([[20, 2]])) # Predict price for size 20, 2 rooms
    ```

---

### 2. Logistic Regression
**Goal:** Binary Classification (predicting 0 or 1, Yes or No).

**Principle:**
Despite the name "Regression," it is used for **classification**.
Instead of fitting a straight line that goes to infinity, it squashes the output between 0 and 1 using the **Sigmoid Function**.
*   **Sigmoid Formula:** $S(x) = \frac{1}{1 + e^{-x}}$
*   **Decision Boundary:** Usually, if the output probability $> 0.5$, we classify as 1 (True); otherwise 0 (False).

**Illustration:**
```text
  1.0 |           _______   <-- Class 1
      |          /
  P   |         /  <-- Sigmoid Curve
      |        /
  0.0 |_______/             <-- Class 0
      +----------------->
             x
```

**Python Examples:**

*   **Example 1: Basic Binary Classification**
    ```python
    from sklearn.linear_model import LogisticRegression

    # X: Exam Score, y: Pass(1)/Fail(0)
    X = [[10], [20], [80], [90]]
    y = [0, 0, 1, 1]

    clf = LogisticRegression()
    clf.fit(X, y)

    print(clf.predict([[50]]))       # Prediction (0 or 1)
    print(clf.predict_proba([[50]])) # Probability: [prob_0, prob_1]
    ```

---

### 3. L1 & L2 Regularization
**Goal:** Prevent **Overfitting**.
*Overfitting is when the model memorizes the training data (including noise) and fails to generalize to new data.*

**Principle:**
We add a "penalty" to the Loss Function based on the size of the weights ($w$).
*   **L1 (Lasso):** Penalty is the absolute value of weights ($|w|$).
    *   *Effect:* Drives some weights to exactly **zero**. Good for **feature selection** (removing useless features).
*   **L2 (Ridge):** Penalty is the square of weights ($w^2$).
    *   *Effect:* Makes weights very small (close to zero), but not exactly zero. Keeps all features but reduces their impact.

**Python Examples:**

*   **Example 1: Ridge (L2)**
    ```python
    from sklearn.linear_model import Ridge
    
    # Alpha controls the strength of regularization (higher = simpler model)
    ridge_model = Ridge(alpha=1.0) 
    ridge_model.fit(X, y)
    ```

*   **Example 2: Lasso (L1)**
    ```python
    from sklearn.linear_model import Lasso
    
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X, y)
    print(lasso_model.coef_) 
    # You might see some coefficients become exactly 0.0
    ```

---

### 4. K-Nearest Neighbors (K-NN)
**Goal:** Classification (mostly) or Regression.

**Principle:**
"Tell me who your friends are, and I'll tell you who you are."
1.  Store all training data (Lazy Learner).
2.  When a new point comes in, calculate the distance (Euclidean) to all points.
3.  Find the **K** closest points.
4.  **Vote:** The majority class among the neighbors becomes the prediction.

**Illustration:**
```text
   A   A
     A      ?  <-- New Point. If K=3, it looks at 3 closest neighbors.
   B   B       If neighbors are (A, A, B), prediction is A.
```

**Python Examples:**

*   **Example 1: Classification with K=3**
    ```python
    from sklearn.neighbors import KNeighborsClassifier

    # Features: [Height, Weight], Labels: 'Small'(0) or 'Large'(1)
    X = [[150, 50], [160, 55], [180, 80], [190, 85]]
    y = [0, 0, 1, 1]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    print(knn.predict([[170, 70]]))
    ```

---

### 5. Decision Trees
**Goal:** Classification and Regression.

**Principle:**
The model asks a series of "Yes/No" questions to split the data.
*   **Splitting:** It selects the feature that best separates the data (maximizes "purity").
*   **Metric:** Uses **Entropy** or **Gini Impurity** to measure how mixed a group is. We want groups to be pure (all 0s or all 1s).

**Illustration:**
```text
       [Is Weight > 70kg?]
           /       \
        Yes         No
       /             \
   [Class 1]    [Is Height > 160?]
                   /         \
                 Yes         No
                /             \
            [Class 1]      [Class 0]
```

**Python Examples:**

*   **Example 1: Decision Tree Classifier**
    ```python
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=3) # Limit depth to prevent overfitting
    clf.fit(X, y)
    
    # We can actually visualize the tree logic
    from sklearn.tree import export_text
    print(export_text(clf))
    ```

---

### 6. Model Ensembles
**Goal:** Combine multiple "weak" models to create one "strong" model.

#### A. Bagging (Bootstrap Aggregating) & Random Forest
**Principle:**
*   **Parallel Training:** Train many Decision Trees independently on random subsets of the data (with replacement).
*   **Voting:** For classification, the forest takes a majority vote of all trees.
*   *Analogy:* "Wisdom of the crowd." Even if one tree is wrong, the majority is likely right. It reduces **Variance**.

#### B. Boosting (Gradient Boosting)
**Principle:**
*   **Sequential Training:** Train trees one by one.
*   **Correction:** Each new tree tries to fix the **errors** made by the previous tree.
*   *Analogy:* Like a student learning. They take a test, see what they got wrong, and study *specifically those topics* for the next test. It reduces **Bias**.

**Python Examples:**

*   **Example 1: Random Forest (Bagging)**
    ```python
    from sklearn.ensemble import RandomForestClassifier

    # n_estimators = number of trees
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    print(rf.predict(X_test))
    ```

*   **Example 2: Gradient Boosting**
    ```python
    from sklearn.ensemble import GradientBoostingClassifier

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    gb.fit(X, y)
    ```

---

### 7. Support Vector Machines (SVM)
**Goal:** Classification (mostly).

**Principle:**
SVM tries to find the best boundary (hyperplane) that separates two classes with the **widest possible margin**.
*   **Support Vectors:** The specific data points that lie closest to the decision line. These are the "difficult" points that define the boundary.
*   **Kernel Trick:** If data isn't linearly separable (e.g., a circle inside a circle), SVM projects data into higher dimensions (3D, 4D, etc.) where a flat sheet *can* separate them.

**Illustration:**
```text
      Class A
      *   *
    *       *
  ----Margin----  <-- Wide Gap (Street)
  ==============  <-- The Hyperplane (Decision Boundary)
  ----Margin----
    o       o
      o   o
      Class B
```

**Python Examples:**

*   **Example 1: Linear SVM**
    ```python
    from sklearn.svm import SVC # Support Vector Classifier

    svm = SVC(kernel='linear')
    svm.fit(X, y)
    print(svm.predict(X_test))
    ```

*   **Example 2: Non-linear SVM (RBF Kernel)**
    ```python
    # Used when a straight line cannot separate the data
    svm_rbf = SVC(kernel='rbf', C=1.0) 
    svm_rbf.fit(X, y)
    ```

---

### Summary Table for the Olympiad

| Algorithm | Key Concept | Best Used For | Warning |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Best fit line, minimize MSE | Predicting numbers | Sensitive to outliers |
| **Logistic Regression**| Sigmoid curve, probability | Binary classification | Assumes linear boundary |
| **K-NN** | Distance to neighbors | Simple patterns | Slow on large data |
| **Decision Tree** | If-else splits, Entropy | Visualizing logic | Prone to overfitting |
| **Random Forest** | Many trees, Majority vote | Robust, general purpose | Hard to interpret |
| **Gradient Boosting** | Sequential correction of errors | High accuracy competitions | Can overfit if not tuned |
| **SVM** | Widest margin, Kernels | Complex/High-dim data | Slow on massive datasets |
