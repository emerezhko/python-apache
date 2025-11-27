Here is the explanation for the **Neural Networks (Basics)** section of the syllabus. This is the foundation of Deep Learning.

---

### 1. Perceptron Basics
**Goal:** The simplest building block of a neural network (a single artificial neuron).

**Principle:**
A Perceptron mimics a biological neuron. It takes multiple inputs, weighs their importance, sums them up, adds a bias, and decides whether to "fire" (output 1) or not (output 0).

*   **Formula:** $z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + \dots + b$
*   **Activation:** Output = 1 if $z > 0$, else 0 (Step Function).
*   **Limitation:** A single Perceptron can only solve **Linearly Separable** problems (it draws a straight line). It cannot solve XOR (Exclusive OR).

**Illustration:**
```text
Inputs    Weights
  x1 ----> w1 \
  x2 ----> w2  --> Sum (∑) + Bias (b) --> Activation --> Output (y)
  x3 ----> w3 /
```

**Python Example (sklearn):**
```python
from sklearn.linear_model import Perceptron
# X: [0,0], [0,1], [1,0], [1,1] (OR gate logic)
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 1]

net = Perceptron()
net.fit(X, y)
print(net.predict([[0, 1]])) # Output: 1
```

---

### 2. Gradient Descent
**Goal:** The optimization algorithm used to train networks (i.e., find the best weights).

**Principle:**
Imagine you are on a mountain (the Loss Function) blindfolded, and you want to reach the lowest valley (Minimum Error).
1.  Feel the slope of the ground under your feet (**Gradient**).
2.  Take a small step downhill (**Learning Rate**).
3.  Repeat until the slope is flat.

*   **The Update Rule:**
    $$w_{new} = w_{old} - \text{LearningRate} \cdot \frac{\partial Loss}{\partial w}$$
*   **Learning Rate ($\alpha$):**
    *   *Too small:* Takes forever to converge.
    *   *Too large:* Overshoots the minimum and diverges.

**Visual:**
```text
      \            /
       \          /
        \  O     /   <-- You are here
         \ |    /
          \|   /
           \__/      <-- Goal (Minimum Loss)
```

---

### 3. Backpropagation
**Goal:** Efficiently calculate gradients for every weight in a multi-layer network.

**Principle:**
"The Chain Rule of Calculus applied to networks."
When the network makes a mistake, we need to know **who to blame**.
1.  **Forward Pass:** Input data goes through the network $\to$ Prediction.
2.  **Calculate Loss:** Compare Prediction vs Actual.
3.  **Backward Pass (Backprop):** Calculate how much each weight contributed to the error, moving from the *Output* layer back to the *Input* layer.

**Python Example (PyTorch Concept):**
*In modern Deep Learning, we don't do the math manually. We use libraries like PyTorch or TensorFlow.*
```python
import torch

# Simple weight (parameter) that requires gradient
w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
y_true = torch.tensor(4.0)

# 1. Forward
y_pred = w * x 

# 2. Loss (MSE)
loss = (y_pred - y_true)**2

# 3. Backward (Backpropagation)
loss.backward()

# 4. Gradient shows how much 'w' needs to change
print(w.grad) 
```

---

### 4. Activation Functions
**Goal:** Introduce **Non-Linearity**. Without these, a Neural Network is just a giant Linear Regression model, no matter how many layers it has.

#### A. Sigmoid
*   **Formula:** $\frac{1}{1 + e^{-x}}$
*   **Range:** $(0, 1)$
*   **Use:** Output layer for Binary Classification (Probability).
*   **Problem:** **Vanishing Gradient**. For very high/low numbers, the curve is flat, so the gradient is $\approx 0$, and the network stops learning.

#### B. Tanh (Hyperbolic Tangent)
*   **Range:** $(-1, 1)$
*   **Use:** Often better than Sigmoid for hidden layers because it is **zero-centered**.
*   **Problem:** Still suffers from Vanishing Gradient.

#### C. ReLU (Rectified Linear Unit)
*   **Formula:** $\max(0, x)$ (If $x > 0$ return $x$, else 0).
*   **Range:** $[0, \infty)$
*   **Use:** The **Default** choice for hidden layers in modern networks.
*   **Benefit:** Fast to compute, solves Vanishing Gradient problem (for positive inputs).

**Visual:**
```text
Sigmoid (S-shape)      ReLU (Hockey Stick)
     1 |   /             |      /
       |  /              |     /
   0.5 + /               |    /
       |/              0 +---/
       +----------       +----------
```

---

### 5. Loss Functions
**Goal:** A mathematical formula to calculate "how wrong" the model is. The Gradient Descent minimizes this value.

#### A. Regression Losses (Predicting a number)
1.  **MSE (Mean Squared Error):** $\frac{1}{N} \sum (y - \hat{y})^2$
    *   Punishes large errors heavily (squaring them). Sensitive to outliers.
2.  **MAE (Mean Absolute Error):** $\frac{1}{N} \sum |y - \hat{y}|$
    *   Linear penalty. More robust to outliers.

#### B. Classification Losses (Predicting a class)
1.  **Binary Cross Entropy (Log Loss):** Used for Yes/No tasks (Sigmoid output).
    *   Formula: $- (y \log(\hat{y}) + (1-y) \log(1-\hat{y}))$
    *   Punishes confident wrong answers very harshly.
2.  **Categorical Cross Entropy:** Used for multi-class tasks (Softmax output).

**Python Example:**
```python
import numpy as np

# MSE Example
y_true = np.array([10, 20])
y_pred = np.array([12, 19])
mse = ((y_true - y_pred)**2).mean()
print(f"MSE: {mse}") # ((4) + (1)) / 2 = 2.5

# Cross Entropy Logic (Simplified)
# If Model predicts probability 0.9 for correct class (1): Low Loss
# If Model predicts probability 0.1 for correct class (1): High Loss
def binary_loss(y, p):
    return -np.log(p) if y == 1 else -np.log(1 - p)

print(binary_loss(1, 0.9)) # 0.10 (Small error)
print(binary_loss(1, 0.1)) # 2.30 (Big error)
```

---

### Summary Table for Part 3

| Concept | Key Role | Analogy |
| :--- | :--- | :--- |
| **Perceptron** | Decision Unit | A voter deciding Yes/No |
| **Gradient Descent** | Optimization | Walking down a hill blindfolded |
| **Backpropagation** | Gradient Calculation | Distributing blame for the error |
| **ReLU** | Activation | A switch (ON if positive, OFF if negative) |
| **Sigmoid** | Activation | Probability converter (0 to 1) |
| **MSE** | Regression Loss | Penalty for missing the target (Squared) |
| **Cross Entropy** | Classification Loss | Penalty for being confidently wrong |
