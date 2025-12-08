# 1D-CNN Customer Complaint Classifier (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![NLP](https://img.shields.io/badge/Domain-NLP-green)
![Model](https://img.shields.io/badge/Model-1D--CNN-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> A PyTorch-based 1D Convolutional Neural Network for real-world text classification tasks. Includes tokenization, padded sequence handling, GPU-accelerated training, performance evaluation, and reproducible preprocessing utilities.

---

## 🚀 Overview

This project implements a **Convolutional Neural Network for text classification**:

- Token embeddings processed through **Conv1d**  
- ReLU activation + pooling for feature extraction  
- Fully connected classifier head  
- Weighted loss for class imbalance  
- GPU-accelerated training loop  

This architecture is fast, lightweight, and ideal for sequence pattern extraction.

---

## ▶️ Quickstart

```bash
pip install -r requirements.txt
python train_1d_cnn.py

train_1d_cnn.py
requirements.txt
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

## 📜 License
This project is licensed under the MIT License — see the LICENSE file for details.
