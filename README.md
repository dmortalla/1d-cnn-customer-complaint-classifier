# 1D-CNN Customer Complaint Classifier (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![NLP](https://img.shields.io/badge/Domain-NLP-green)
![Model](https://img.shields.io/badge/Model-1D--CNN-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> A PyTorch-based 1D Convolutional Neural Network for real-world text classification tasks. Includes tokenization, padded sequence handling, GPU-accelerated training, performance evaluation, and reproducible preprocessing utilities.

## 🚀 Quickstart Demo (For Reviewers)

Run a simple inference example using sample text:

```bash
pip install -r requirements.txt
python demo_predict.py
```

This confirms preprocessing, embedding, convolutional feature extraction, and classifier output.

---

## 📦 Full Training Run

Train the full 1D-CNN model:

```bash
python train_1d_cnn.py
```

This script handles:

- Tokenization + vocabulary  
- Sequence batching  
- CNN feature extraction  
- Classification head training  
- Train/validation split  
- Accuracy reporting  

---

## 📁 Files

```text
train_1d_cnn.py   # Full supervised training script
run_demo.py               # Example inference demo
requirements.txt          # Dependencies
```

---

## 🏗 Overview

- Token Embedding → Conv1D → Global Max Pool → Linear Classifier  
- Efficient for short-to-medium text sequences  
- Demonstrates convolutional NLP modeling  
- Fast training even on CPU  

This architecture is commonly used for sentiment analysis and short-text classification.

---

## 📂 Project Structure

```text
.
├── train_cnn_classifier.py
├── run_demo.py
├── requirements.txt
├── CONTRIBUTING.md
└── SECURITY.md
```

---

## 🧱 Architecture Overview

At a high level, the training system looks like this:

```text
Input: [batch_size, seq_len] token IDs
        |
        v
Embedding layer -> [batch, seq_len, embed_dim]
        |
 Transpose to [batch, embed_dim, seq_len]
        |
        v
 Conv1d (kernel=3, padding=1) + ReLU
        |
        v
 Conv1d (kernel=5, padding=2) + ReLU
        |
        v
 AdaptiveMaxPool1d(output_size=1)
        |
        v
  Flatten -> [batch, 128]
        |
        v
   Dropout(0.3)
        |
        v
  Fully-connected layer -> [batch, num_classes]
Input: [batch_size, seq_len] token IDs
        |
        v
Embedding layer -> [batch, seq_len, embed_dim]
        |
 Transpose to [batch, embed_dim, seq_len]
        |
        v
 Conv1d (kernel=3, padding=1) + ReLU
        |
        v
 Conv1d (kernel=5, padding=2) + ReLU
        |
        v
 AdaptiveMaxPool1d(output_size=1)
        |
        v
  Flatten -> [batch, 128]
        |
        v
   Dropout(0.3)
        |
        v
  Fully-connected layer -> [batch, num_classes]
```

---

## 🤝 Contributing
See `CONTRIBUTING.md` for coding style, branching strategy, and PR workflow.

---

## 📄 License
MIT License. See `LICENSE` for details.

