"""
Demo prediction script for the 1D CNN complaint classifier.

This uses a dummy tokenized sequence to show the forward pass.
Replace the dummy_ids with real token IDs from your tokenizer if desired.
"""

import torch
from train_1d_cnn import ComplaintCNN  # ensure this matches your model class name


def main():
    # These hyperparameters must match how the model is defined in train_1d_cnn.py
    vocab_size = 5000
    embed_dim = 64
    num_classes = 3
    max_len = 50

    model = ComplaintCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
    )

    # Dummy batch of token ids: [batch_size, seq_len]
    batch_size = 2
    dummy_ids = torch.randint(0, vocab_size, (batch_size, max_len))

    # Forward pass
    logits = model(dummy_ids)
    probs = torch.softmax(logits, dim=-1)

    print(f"Input shape:   {dummy_ids.shape}")
    print(f"Logits shape:  {logits.shape}")
    print("Probabilities:")
    print(probs)


if __name__ == "__main__":
    main()

