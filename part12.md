Here is the explanation for the **Deep Learning (Advanced)** section. This moves from basic neurons to the architectures and techniques that power modern AI like ChatGPT and Stable Diffusion.

---

### 1. Multi-Layer Perceptrons (MLP)
**Goal:** Solve non-linear problems (like XOR or image recognition) that a single neuron cannot.

**Principle:**
An MLP is a stack of layers:
1.  **Input Layer:** Receives raw data.
2.  **Hidden Layers:** Layers in the middle. They extract abstract features.
3.  **Output Layer:** The final prediction.
Crucially, every neuron is connected to every neuron in the next layer (**Fully Connected** or **Dense**).

**Visual:**
```text
   Input      Hidden      Output
    O ---------> O \
    O ---------> O ->-----> O
    O ---------> O /
```

**Python (PyTorch):**
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128), # Input -> Hidden (Linear transform)
    nn.ReLU(),           # Non-linearity
    nn.Linear(128, 10)   # Hidden -> Output
)
```

---

### 2. Data Embeddings (Text, Image, Audio)
**Goal:** Convert discrete data (words) or raw pixels into **vectors** (lists of numbers) that capture *meaning*.

**Principle:**
*   **Text:** Words with similar meanings should be close in vector space.
    *   *Example:* `Vector("King") - Vector("Man") + Vector("Woman") ≈ Vector("Queen")`.
*   **Image:** Instead of raw pixels, an image is embedded into a feature vector (e.g., by a CNN) representing "cat-ness" or "texture".
*   **Audio:** Raw waves are converted to Spectrograms, then embedded into vectors representing phonemes or tones.

**Visual:**
```text
   Word       Embedding (Vector)
  "Cat"  ->  [0.9, 0.1, 0.0, ...]
  "Dog"  ->  [0.8, 0.2, 0.0, ...]  <-- Close to Cat
  "Car"  ->  [0.0, 0.9, 0.5, ...]  <-- Far away
```

---

### 3. Pooling Techniques
**Goal:** Downsampling. Reduce the size of data to reduce computation and prevent overfitting, while keeping important features.

*   **Max Pooling:** Pick the **largest** number in a window. Keeps the strongest feature (e.g., the sharpest edge).
*   **Average Pooling:** Pick the **average** number. Smoothes the data.

**Visual:**
```text
   Matrix (2x2)       Max Pool       Avg Pool
   [1  3]
   [2  9]      --->      9             3.75
```

---

### 4. Attention Mechanism
**Goal:** Help the model focus on specific parts of the input when making a prediction.

**Principle:**
Imagine reading a sentence. When you read the word "bank", how do you know if it means a river bank or a money bank? You look at the context words ("money", "deposit", "water").
*   Attention assigns a **weight** (importance) to every other word relative to the current word.

**Visual:**
```text
   Input: "The animal didn't cross the street because it was tired."
   
   Question: What does "it" refer to?
   Attention Weights for "it":
      The (0.01) ... animal (0.85) ... street (0.1) ...
   Result: "it" focuses heavily on "animal".
```

---

### 5. Transformers
**Goal:** The architecture behind GPT, BERT, and ViT. It processes data in parallel (unlike RNNs) using **Self-Attention**.

**Principle:**
1.  **Self-Attention:** Every word looks at every other word to understand context.
2.  **Positional Encoding:** Since it processes words in parallel, it adds numbers to vectors to mark order (1st word, 2nd word...).
3.  **For Images (ViT):** The image is chopped into square "patches". Each patch is treated like a "word" in a sentence.

**Structure:**
```text
   [Input] -> [Positional Encoding] -> [Multi-Head Attention] -> [Feed Forward] -> [Output]
                                              ^
                                        (Repeat N times)
```

---

### 6. Autoencoders (Practice)
**Goal:** Unsupervised learning for compression or denoising.

**Principle:**
*   **Encoder:** Compresses input $x$ into a small "Bottleneck" vector $z$ (Latent Space).
*   **Decoder:** Tries to reconstruct the original input $x$ from $z$.
*   If the model succeeds, $z$ contains the "essence" of the data.

**Python (PyTorch):**
```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 28x28 image -> 64 features -> 3 features (Bottleneck)
        self.encoder = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 784))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

---

### 7. Optimization: SGD & Momentum
**Goal:** How to update weights efficiently.

1.  **SGD (Stochastic Gradient Descent):** Updates weights after *every single sample*.
    *   *Pros:* Fast updates. *Cons:* Very noisy/unstable path.
2.  **Mini-Batch GD:** Updates weights after a small batch (e.g., 32 images). **Standard.**
3.  **Momentum:**
    *   SGD can get stuck in flat areas or oscillate in ravines.
    *   Momentum adds a "velocity" term. If you were going downhill, keep going that direction even if the gradient changes slightly.
4.  **Adam / AdamW:**
    *   **Adam:** Combines Momentum (speed) + Adaptive Learning Rates (different steps for different parameters).
    *   **AdamW:** A fixed version of Adam that handles Weight Decay correctly. **Default choice for Transformers.**

---

### 8. Regularization Techniques (Practice)
**Goal:** Prevent Overfitting (memorizing training data).

1.  **Dropout:**
    *   During training, randomly set some neurons to **Zero**.
    *   Forces the network to not rely on one specific path.
    ```python
    nn.Dropout(p=0.5) # Drop 50% of neurons randomly
    ```

2.  **Early Stopping:**
    *   Monitor Validation Loss. If it starts going UP, stop training immediately.

3.  **Weight Decay:**
    *   Adds a penalty to the Loss function: $Loss + \lambda \sum w^2$.
    *   Forces weights to be small. Large weights usually imply overfitting.

---

### 9. Batch Normalization (Practice)
**Goal:** Stabilize training and allow higher learning rates.

**Principle:**
*   Deep networks suffer from "Internal Covariate Shift" (input distribution changes every layer).
*   **Batch Norm:** Normalizes the input of a layer to have Mean=0 and Variance=1 (using the current batch statistics), then scales/shifts it.
*   *Analogy:* It keeps the data "centered" so the next layer doesn't have to constantly chase moving targets.

**Python:**
```python
# Usually placed AFTER Linear/Conv and BEFORE Activation
layer = nn.Sequential(
    nn.Linear(64, 128),
    nn.BatchNorm1d(128),
    nn.ReLU()
)
```

---

### 10. Weight Initialization (Practice)
**Goal:** Where to start? If weights = 0, neurons won't learn distinct features. If weights are too large, gradients explode.

*   **Xavier (Glorot) Initialization:** Best for Sigmoid/Tanh activations.
*   **Kaiming (He) Initialization:** Best for **ReLU** activations.

**Python:**
```python
import torch.nn.init as init
# Applying Kaiming Init to a layer
init.kaiming_normal_(layer.weight)
```

---

### 11. Model Finetuning (Practice)
**Goal:** Don't reinvent the wheel. Use pre-trained models.

**Principle:**
1.  **Transfer Learning:** Take a model trained on ImageNet (ResNet) or Wikipedia (BERT).
2.  **Freeze:** Lock the early layers (feature extractors).
3.  **Replace Head:** Change the last layer to match your classes.
4.  **Train:** Train only the last layer (or the whole model with a very small learning rate).

**Parameter-Efficient Fine-Tuning (PEFT):**
*   **LoRA (Low-Rank Adaptation):** Instead of updating all 100B parameters of GPT-4, we inject small matrices (Adapters) and train only those. Saves massive memory.

**Python (Pseudo-code):**
```python
import torchvision.models as models

# 1. Load Pre-trained ResNet
resnet = models.resnet18(pretrained=True)

# 2. Freeze all params
for param in resnet.parameters():
    param.requires_grad = False

# 3. Replace output layer (originally 1000 classes) with our 2 classes
resnet.fc = nn.Linear(resnet.fc.in_features, 2)

# 4. Now only resnet.fc will be trained
```
