Here is the explanation for the **Computer Vision (CV)** section of the syllabus. This field deals with teaching computers to "see" and understand images.

---

### 1. Convolutional Layers (Fundamentals)
**Goal:** Feature extraction. Instead of looking at every pixel individually (which loses context), we look at local patterns (lines, curves, eyes).

**Principle:**
*   **Kernel (Filter):** A small matrix (e.g., $3 \times 3$) of numbers.
*   **Convolution:** You slide this kernel over the input image like a flashlight. At every step, you multiply the kernel numbers by the image pixels and sum them up.
*   **Result:** If the kernel looks for a "vertical line", the output will light up wherever there is a vertical line in the image.

**Visual:**
```text
  Image (5x5)       Kernel (3x3)      Output (Feature Map)
  [0 1 1 1 0]       [1  0  1]
  [0 0 1 1 1]   *   [0  1  0]   ->    Single number
  [0 0 0 1 1]       [1  0  1]         (Dot Product)
  ...
```

**Python (PyTorch):**
```python
import torch.nn as nn
# in_channels=3 (RGB), out_channels=16 (16 different filters/features)
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
```

---

### 2. Image Classification
**Goal:** Assign a single label to an entire image (e.g., "Cat").

**Pipeline:**
`Input Image -> Conv Layers (Features) -> Pooling (Downsample) -> Flatten -> Linear Layers -> Output Class`

**Practice Tip:**
Always use **Softmax** at the end to get probabilities (e.g., 90% Cat, 10% Dog).

---

### 3. Object Detection (YOLO, SSD, DETR)
**Goal:** Find *what* objects are present and *where* they are (Bounding Boxes).

**Architectures:**
1.  **YOLO (You Only Look Once):** fast. Splits image into a grid. Each grid cell predicts if there is an object center inside it and draws a box.
2.  **SSD (Single Shot Detector):** Similar to YOLO but looks at feature maps of different sizes to catch small and big objects.
3.  **DETR (DEtection TRansformer):** Modern. Uses Transformers to treat objects as a "set". No need for complex grid logic (Anchor-free).

**Metric:** **IoU (Intersection over Union)**.
*   Area of Overlap / Area of Union between predicted box and real box.

**Visual:**
```text
  Classification: "Cat"
  Detection:      "Cat" at [x=10, y=50, w=100, h=200]
```

---

### 4. Image Segmentation (U-Net)
**Goal:** Pixel-level classification. Draw the exact outline of the object.

**Architecture: U-Net**
Designed for biomedical imaging but used everywhere.
1.  **Encoder (Left side):** Normal CNN, downsamples image to understand "What" is in it (Context).
2.  **Decoder (Right side):** Upsamples image back to original size to locate "Where" it is.
3.  **Skip Connections:** Crucial! It copies high-resolution features from the Left side directly to the Right side so details (edges) aren't lost.

**Visual:**
```text
   Input                Output
  [High Res] --(Copy)--> [High Res]
     |                      ^
     v (Down)               | (Up)
  [Low Res]              [Low Res]
      \____________________/
```

---

### 5. Pre-trained Vision Encoders (e.g., ResNet)
**Goal:** Don't train from scratch. Use "knowledge" from millions of ImageNet photos.

**The Problem:** Deep networks used to be hard to train because gradients vanished (became 0).
**The Solution (ResNet):** **Residual Connections**.
*   Formula: $y = F(x) + x$
*   Instead of learning the whole transformation, the layer learns the "residual" (difference). The signal can "skip" layers if needed.

**Python:**
```python
import torchvision.models as models
# 'weights' downloads pre-trained parameters
resnet = models.resnet50(weights='IMAGENET1K_V1')
```

---

### 6. Image Augmentation
**Goal:** Create more training data and make the model robust by modifying existing images.

**Techniques:**
*   **Geometric:** Rotate, Flip, Crop, Zoom.
*   **Color:** Brightness, Contrast, Saturation jitter.
*   **Advanced:** MixUp (blending two images), CutOut (blacking out a square).

**Python:**
```python
from torchvision import transforms

aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2)
])
```

---

### 7. Generative Adversarial Networks (GANs)
**Goal:** Generate fake images that look real.

**Principle:** A "Game" between two networks.
1.  **Generator (The Forger):** Takes random noise and tries to paint a fake image.
2.  **Discriminator (The Detective):** Looks at real images and fake images and tries to tell which is which.
3.  **Loop:** The Generator gets better at fooling the Discriminator; the Discriminator gets better at spotting fakes.

---

### 8. Self-Supervised Learning (SSL)
**Goal:** Learn visual representations from huge datasets *without* human labels.

**Techniques:**
1.  **Contrastive Learning (e.g., SimCLR):**
    *   Take an image of a dog. Create 2 augmented versions (cropped, black-and-white).
    *   Tell the model: "These two vectors should be close. Vectors of other images should be far."
2.  **Masked Image Modeling (e.g., MAE):**
    *   Delete 75% of the image squares.
    *   Force the model to reconstruct the missing pixels. It learns context (e.g., "if I see a tail, there must be a dog body").

---

### 9. Vision-Text Encoders (e.g., CLIP)
**Goal:** Connect images and text in the same mathematical space. This enables **Zero-Shot Classification**.

**Principle:**
*   Train an Image Encoder and a Text Encoder on internet pairs (Image, Caption).
*   Maximize similarity between the vector of the image and the vector of its caption.
*   **Usage:** To classify an image, calculate distance between Image Vector and Text Vectors ["Dog", "Cat", "Car"]. The closest text is the label.

**Visual:**
```text
  Image Vector ("Picture of a dog")  ---->  [0.1, 0.9] \
                                                        High Similarity
  Text Vector ("A cute puppy")       ---->  [0.15, 0.8]/
```

---

### 10. Diffusion Models
**Goal:** The state-of-the-art for Image Generation (Stable Diffusion, DALL-E).

**Principle:**
Based on thermodynamics (diffusion of gas).
1.  **Forward Process (Destroy):** Slowly add Gaussian noise to an image until it becomes pure static (random noise).
2.  **Reverse Process (Create):** Train a neural network (usually a U-Net) to predict **"How much noise was added in this step?"** and subtract it.
3.  **Generation:** Start with pure random noise $\to$ Ask model to remove noise $\to$ Repeat $\to$ Clear Image.

**Difference from GANs:** Diffusion is more stable and generates more diverse images, but is slower (needs many steps to denoise).
