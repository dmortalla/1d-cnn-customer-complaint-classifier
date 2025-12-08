# Contributing to Customer Complaint CNN Classifier

Thank you for contributing to this NLP classification project using a 1D-CNN architecture.  
This model showcases efficient sequence modeling using convolutional operations.

---

## 1. Fork the Repository

Click **Fork** on GitHub to create your own copy.

---

## 2. Clone Your Fork & Create a Branch

```bash
git clone https://github.com/<your-username>/cnn-customer-complaint-classifier.git
cd cnn-customer-complaint-classifier
git checkout -b feature/your-feature-name
```

---

## 3. Make Your Changes

- Keep changes modular and easy to understand.
- Maintain GPU compatibility for both training and inference.
- If altering tokenization or preprocessing, explain rationale.
- Keep the CNN pipeline (embedding → conv → pooling → classifier) readable.

---

## 4. Run Basic Checks

### Syntax Validation

```bash
python -m compileall .
```

### Quick Training Sanity Check

```bash
python train_cnn.py --epochs 1 --small-run
```

### Optional: Run Tests (if added)

```bash
pytest
```

---

## 5. Open a Pull Request

- Describe the purpose of the change.
- Include before/after accuracy comparisons for model changes.
- Provide sample outputs for preprocessing modifications.

---

## Code Style Guidelines

- Use intuitive variable names (`x_emb`, `conv_out`, etc.).
- Keep forward passes linear and easily traceable.
- Use Google-style docstrings when documenting functions.
- Add tensor shape notes where appropriate.

---

## Thank You

Your contributions make this project more robust and accessible.

