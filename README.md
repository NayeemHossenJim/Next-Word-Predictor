<div align="center">

# 🔮 Next Word Predictor

### A character-aware, sequence-to-sequence LSTM language model built with TensorFlow/Keras  

### that learns from a personal FAQ corpus and generates contextually relevant next words

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Example Output](#-example-output)
- [Key Design Decisions](#-key-design-decisions)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

This project implements a **next-word prediction model** using a stacked **LSTM (Long Short-Term Memory)** neural network. The model is trained on a curated personal FAQ corpus that covers topics such as AI engineering, machine learning systems, and game development.

Given a seed phrase, the model iteratively predicts the most probable next word — generating coherent, contextually grounded continuations of the input text.

> **Use case:** Demonstrates how sequence modeling and NLP preprocessing can be combined to build a lightweight language model from scratch, without relying on pre-trained transformers.

---

## ⚙️ How It Works

```
Raw Text Corpus
      │
      ▼
 Keras Tokenizer          ← Builds a word-to-index vocabulary (282 unique words)
      │
      ▼
N-gram Sequence Gen.      ← Sliding window over every sentence to create prefix sequences
      │
      ▼
Pre-padding (maxlen)      ← All sequences zero-padded on the left to uniform length
      │
      ▼
  X  /  y split           ← X = all tokens except last │ y = last token (next word)
      │
      ▼
 to_categorical(y)        ← One-hot encode labels  →  shape: (samples, 283)
      │
      ▼
  LSTM Model              ← Embedding → LSTM × 3 → Dense (softmax)
      │
      ▼
   Prediction             ← seed text → tokenize → pad → argmax(predict) → decode word
```

---

## 🏗️ Model Architecture

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| 1 | `Embedding` | `(None, 37, 100)` | `28,300` |
| 2 | `LSTM` | `(None, 37, 150)` | `150,600` |
| 3 | `LSTM` | `(None, 37, 150)` | `180,600` |
| 4 | `LSTM` | `(None, 150)` | `180,600` |
| 5 | `Dense (softmax)` | `(None, 283)` | `42,733` |

**Compile settings:**

```python
optimizer  = 'adam'
loss       = 'categorical_crossentropy'   # y is one-hot encoded
metrics    = ['accuracy']
epochs     = 200
```

> **Why `categorical_crossentropy` and not `sparse_categorical_crossentropy`?**  
> Because `y` is converted to one-hot vectors via `to_categorical()`. Using `sparse_*` would require raw integer class IDs and would cause a shape mismatch error at training time.

---

## 📁 Project Structure

```
Next-Word-Predictor/
│
├── lstm_project.ipynb   # Main notebook — data prep, model training & inference
├── README.md            # Project documentation (you are here)
└── LICENSE              # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- pip or conda
- A GPU is optional but significantly speeds up the 200-epoch training

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/NayeemHossenJim/Next-Word-Predictor.git
   cd Next-Word-Predictor
   ```

2. **Create and activate a virtual environment** *(recommended)*

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install tensorflow numpy jupyter
   ```

### Running the Notebook

```bash
jupyter notebook lstm_project.ipynb
```

Or open it directly in **VS Code** with the Jupyter extension installed.

> Run all cells top-to-bottom. The final cell performs live next-word generation starting from a seed sentence.

---

## 🔬 Pipeline Walkthrough

### 1. Corpus Definition

A personal FAQ paragraph (≈ 300 words) covering AI engineering interests serves as the training corpus. It contains rich, domain-specific vocabulary relevant to ML systems, RAG pipelines, and game development.

### 2. Tokenization

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])
# Vocabulary size: 282 unique words → indices 1–282 (0 reserved for padding)
```

### 3. N-gram Sequence Generation

For every sentence in the corpus a sliding prefix window is built:

```python
for sentence in faqs.split('\n'):
    tokens = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokens)):
        input_sequences.append(tokens[:i+1])
# e.g. "I am an AI" → [I], [I,am], [I,am,an], [I,am,an,AI]
```

### 4. Padding

All sequences are left-padded to `max_len` (the length of the longest sequence):

```python
padded = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
```

### 5. Feature / Label Split

```python
X = padded[:, :-1]   # All tokens except the last  → inputs
y = padded[:, -1]    # The last token in each seq   → label (next word)
y = to_categorical(y, num_classes=283)   # One-hot encode
```

### 6. Model Training

```python
model.fit(X, y, epochs=200)
```

### 7. Inference

```python
text = "Yes. I am passionate about Game Development"
for i in range(10):
    tokens          = tokenizer.texts_to_sequences([text])[0]
    padded_tokens   = pad_sequences([tokens], maxlen=56, padding='pre')
    predicted_index = np.argmax(model.predict(padded_tokens))
    next_word       = [w for w, idx in tokenizer.word_index.items() if idx == predicted_index][0]
    text           += " " + next_word
```

---

## 💬 Example Output

**Seed:** `"Yes. I am passionate about Game Development"`

```
Yes. I am passionate about Game Development and
Yes. I am passionate about Game Development and enjoy
Yes. I am passionate about Game Development and enjoy exploring
Yes. I am passionate about Game Development and enjoy exploring how
Yes. I am passionate about Game Development and enjoy exploring how ai
...
```

---

## 🎯 Key Design Decisions

| Decision | Rationale |
|---|---|
| **3 stacked LSTM layers** | Deeper recurrence captures longer-range dependencies in the corpus |
| **Embedding dim = 100** | Balances expressiveness with the small vocabulary size (282 words) |
| **Pre-padding** | Ensures the most recent context is always aligned to the right of the input window |
| **`categorical_crossentropy`** | Required because `y` is one-hot via `to_categorical`; `sparse_*` would cause a rank mismatch |
| **`argmax` for prediction** | Greedy decoding — deterministic and fast for a small vocab model |

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/NayeemHossenJim">NayeemHossenJim</a>
</div>
