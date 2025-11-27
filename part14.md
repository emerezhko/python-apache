Here is the explanation for the **Natural Language Processing (NLP)** section. This covers how machines read, understand, and generate human language.

---

### 1. Text Classification (Practice)
**Goal:** Assign a label to a piece of text (e.g., Spam vs. Ham, Sentiment Analysis).

**Approaches:**
1.  **Classical (Baseline):** Convert text to numbers using **TF-IDF** (Term Frequency-Inverse Document Frequency) and feed it to Logistic Regression. Fast and effective for simple tasks.
2.  **Modern:** Fine-tune a Transformer (like BERT).

**Principle (TF-IDF):**
*   **TF:** How often does the word appear in *this* document?
*   **IDF:** How rare is the word across *all* documents? (Reduces the importance of words like "the", "is").

**Python Example (The Classic Baseline):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Training data
texts = ["I love this movie", "Worst film ever", "Great acting"]
labels = [1, 0, 1] # 1=Positive, 0=Negative

# Create a pipeline: Raw Text -> Numbers -> Model
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)

print(model.predict(["I did not like it"])) # Output: [0]
```

---

### 2. Pre-trained Text Encoders (e.g., BERT)
**Goal:** Understanding the **context** of words.

**Principle (BERT - Bidirectional Encoder Representations from Transformers):**
*   **Bidirectional:** Unlike previous models that read left-to-right, BERT reads the whole sentence at once. It knows that "Bank" in "Bank account" is different from "Bank" in "River bank".
*   **Training Method (Masked Language Modeling - MLM):** It hides 15% of words in a sentence and tries to guess them based on context.
    *   *Input:* "The cat sat on the [MASK]."
    *   *Target:* "mat".

**Visual:**
```text
   [CLS]  The   cat   sat   on   the  [MASK]
     |     |     |     |     |    |      |
   [E]   [E]   [E]   [E]   [E]  [E]    [E]  <-- Transformer Layers
     |     |     |     |     |    |      |
   Vector  ...   ...   ...   ...  ...   Prediction: "mat"
```

**Python (Hugging Face Transformers):**
```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# The vector representing the 'meaning' of the start of the sentence
cls_embedding = outputs.last_hidden_state[0][0] 
print(cls_embedding.shape) # torch.Size([768])
```

---

### 3. Encoder-Decoder Models
**Goal:** Sequence-to-Sequence (Seq2Seq) tasks where the input and output lengths differ.
*   *Examples:* Machine Translation (English to French), Summarization, Image Captioning.

**Principle:**
1.  **Encoder:** Reads the input (Text or Image) and compresses it into a high-dimensional **Context Vector** (Understanding).
2.  **Decoder:** Takes the context vector and generates the output sequence one token at a time (Auto-regressive).

**Vision-Language Modeling (Multimodal):**
*   **Encoder:** Vision Transformer (ViT) processes the image.
*   **Decoder:** GPT-style text decoder generates the description.

**Visual:**
```text
   Input: "Hello"
      |
   [Encoder] ---> [Context Vector] ---> [Decoder] ---> "Bonjour"
                                            ^
                                     (Generates "Bon", then "jour")
```

**Python Example (Translation with T5):**
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_text = "translate English to German: Good morning"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0])) # "Guten Morgen"
```

---

### 4. Pre-trained Language Models (LLMs)
**Goal:** General-purpose text generation and reasoning (e.g., GPT-4, Llama 3).

**Principle (Causal Language Modeling):**
*   Predict the **next word** given all previous words.
*   $P(w_t | w_{1}, w_{2}, ..., w_{t-1})$.

**Types:**
1.  **Open Source:** You download weights and run locally (e.g., Meta's Llama, Mistral).
    *   *Pros:* Privacy, free. *Cons:* Requires strong GPU.
2.  **API-Based:** You send text to a server (e.g., OpenAI, Anthropic, Google Gemini).
    *   *Pros:* Huge powerful models. *Cons:* Cost per token, data privacy.

**Python Example 1: Open Source (Local Inference):**
```python
# Requires 'transformers', 'accelerate', 'bitsandbytes' (for 4-bit loading)
from transformers import pipeline

# Using a small model for demo
pipe = pipeline("text-generation", model="gpt2") 
result = pipe("The theory of relativity is", max_length=20)
print(result[0]['generated_text'])
```

**Python Example 2: API (OpenAI style):**
```python
# import openai (pseudo-code)
# client = openai.OpenAI(api_key="...")

# response = client.chat.completions.create(
#     model="gpt-4",
#     messages=[
#         {"role": "system", "content": "You are a helpful math tutor."},
#         {"role": "user", "content": "Explain calculus in one sentence."}
#     ]
# )
# print(response.choices[0].message.content)
```

**Key Practice Note for Olympiads:**
*   Be ready to use **Quantization** (loading models in 4-bit or 8-bit) to fit LLMs into memory if you are running them locally.
*   Understand **Temperature** (Low = factual/deterministic, High = creative/random).
