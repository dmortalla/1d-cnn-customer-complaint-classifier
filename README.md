# 1D-CNN Customer Complaint Classifier (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![NLP](https://img.shields.io/badge/Domain-NLP-green)
![Model](https://img.shields.io/badge/Model-1D--CNN-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> A PyTorch-based 1D Convolutional Neural Network for real-world text classification tasks. Includes tokenization, padded sequence handling, GPU-accelerated training, performance evaluation, and reproducible preprocessing utilities.

---

## 🚀 Quickstart Demo

```bash
pip install -r requirements.txt
python run_demo.py
```

Runs an end-to-end example using a sample complaint for inference.

---

## 📁 Files

```text
train_cnn_classifier.py   # Full supervised training script
run_demo.py               # Sample inference demo
requirements.txt          # Dependencies
```

---

## 🏗 Overview

- Token embedding layer  
- 1D convolutional feature extractor  
- Global max pooling layer  
- Fully connected classification head  
- Cross-entropy loss + accuracy tracking  

This design balances speed, simplicity, and performance for text classification.

---

## 📂 Project Structure

```text
.
├── train_cnn_classifier.py
├── run_demo.py
├── requirements.txt
└── CONTRIBUTING.md
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
See CONTRIBUTING.md for guidelines on code style and submitting pull requests.

---

## 📄 License
MIT License. See `LICENSE` for details.
