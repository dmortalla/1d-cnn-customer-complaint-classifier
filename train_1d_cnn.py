"""1D-CNN text classifier for customer complaints (dummy data example)."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


@dataclass
class Config:
    max_len: int = 128
    vocab_size: int = 20000
    embed_dim: int = 128
    num_classes: int = 5
    batch_size: int = 64
    num_epochs: int = 3
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DummyComplaintDataset(Dataset):
    """Placeholder dataset using random token sequences."""

    def __init__(self, n_samples: int, cfg: Config) -> None:
        super().__init__()
        self.inputs = torch.randint(
            low=1, high=cfg.vocab_size, size=(n_samples, cfg.max_len)
        )
        self.labels = torch.randint(low=0, high=cfg.num_classes, size=(n_samples,))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


class TextCNN(nn.Module):
    """Simple 1D-CNN model for text classification."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(cfg.embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)           # [batch, seq_len, embed_dim]
        x = x.transpose(1, 2)          # [batch, embed_dim, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)   # [batch, 128]
        x = self.dropout(x)
        return self.fc(x)


def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    ds = DummyComplaintDataset(n_samples=2000, cfg=cfg)
    idx_train, idx_val = train_test_split(
        list(range(len(ds))), test_size=0.2, random_state=42
    )

    train_subset = torch.utils.data.Subset(ds, idx_train)
    val_subset = torch.utils.data.Subset(ds, idx_val)

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader


def train(cfg: Config) -> None:
    device = torch.device(cfg.device)
    model = TextCNN(cfg).to(device)

    train_loader, val_loader = get_dataloaders(cfg)
    class_weights = torch.ones(cfg.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Validation accuracy: {acc:.4f}")


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
