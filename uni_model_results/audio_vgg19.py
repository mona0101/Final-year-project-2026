import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    base_dir: str = "/content/drive/MyDrive/Colab Notebooks/dataset/Audio"
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-4
    epochs: int = 5
    duration: float = 10.0
    segment_length: float = 0.25
    sr: int = 44100
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    threshold: float = 0.5


# ============================================================
# Helpers
# ============================================================
def numerical_sort(value: str):
    parts = re.split(r"(\d+)", value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def validate_split_dir(split_dir: str, split_name: str) -> None:
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"{split_name} split not found: {split_dir}")

    for category in ["Drone", "Background"]:
        category_dir = os.path.join(split_dir, category)
        if not os.path.isdir(category_dir):
            raise FileNotFoundError(
                f"Missing category folder '{category}' in {split_name}: {category_dir}"
            )


# ============================================================
# Dataset
# ============================================================
class DroneAudioSegmentDataset(Dataset):
    """
    Audio-only baseline dataset.
    Each 10-second wav file is split into 0.25-second segments.
    For each segment, a normalized MFCC tensor with shape (1, 40, 40) is returned.
    Labels: Drone=1, Background=0
    """

    def __init__(
        self,
        audio_root_split: str,
        duration: float = 10.0,
        segment_length: float = 0.25,
        sr: int = 44100,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.audio_root_split = audio_root_split
        self.duration = duration
        self.segment_length = segment_length
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_segments = int(self.duration / self.segment_length)
        self.samples = self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No audio samples were found in: {audio_root_split}")

    def _load_samples(self) -> List[Dict]:
        samples: List[Dict] = []

        for category in ["Drone", "Background"]:
            label = 1.0 if category == "Drone" else 0.0
            category_path = os.path.join(self.audio_root_split, category)
            if not os.path.isdir(category_path):
                continue

            scenario_dirs = sorted(
                [
                    d
                    for d in os.listdir(category_path)
                    if os.path.isdir(os.path.join(category_path, d))
                ],
                key=numerical_sort,
            )

            for scenario in scenario_dirs:
                scenario_path = os.path.join(category_path, scenario)
                audio_files = sorted(
                    [f for f in os.listdir(scenario_path) if f.lower().endswith(".wav")],
                    key=numerical_sort,
                )

                for audio_file in audio_files:
                    audio_path = os.path.join(scenario_path, audio_file)
                    for seg_idx in range(self.num_segments):
                        samples.append(
                            {
                                "audio_path": audio_path,
                                "start": seg_idx * self.segment_length,
                                "label": label,
                            }
                        )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        y, _ = librosa.load(
            sample["audio_path"],
            sr=self.sr,
            mono=True,
            offset=sample["start"],
            duration=self.segment_length,
        )

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        # Force the time dimension to 40 so the output becomes (40, 40)
        mfcc = mfcc[:, : self.n_mfcc]
        pad_width = max(0, self.n_mfcc - mfcc.shape[1])
        if pad_width > 0:
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")

        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return x, label


# ============================================================
# Model
# ============================================================
class GlobalPooling2D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), x.size(1), -1)
        return torch.mean(x, dim=2)


class AudioVGG19TRIDENT(nn.Module):
    """
    VGG19 features + global average pooling + batch norm + linear head.
    Returns logits for binary classification.
    """

    def __init__(self):
        super().__init__()
        vgg_features = list(tv_models.vgg19(weights=None).features)
        vgg_features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.features = nn.Sequential(*vgg_features)
        self.global_pool = GlobalPooling2D()
        self.bn = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.bn(x)
        x = self.fc(x)
        return x.squeeze(1)


# ============================================================
# Training / Evaluation
# ============================================================
def evaluate(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            running_loss += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            total += y.numel()

    avg_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total += y.numel()

    return running_loss / total if total else 0.0



def plot_history(history: Dict[str, List[float]], test_acc: float) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss vs Epoch")
    plt.xticks(list(epochs))
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["val_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch")
    plt.xticks(list(epochs))
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["val_acc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.xticks(list(epochs))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(["Best Validation", "Test"], [max(history["val_acc"]), test_acc])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Best Validation vs Test Accuracy")
    plt.grid(True, axis="y")
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dir = os.path.join(cfg.base_dir, "Train")
    val_dir = os.path.join(cfg.base_dir, "Validation")
    test_dir = os.path.join(cfg.base_dir, "Test")

    validate_split_dir(train_dir, "Train")
    validate_split_dir(val_dir, "Validation")
    validate_split_dir(test_dir, "Test")

    train_ds = DroneAudioSegmentDataset(
        train_dir,
        duration=cfg.duration,
        segment_length=cfg.segment_length,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    val_ds = DroneAudioSegmentDataset(
        val_dir,
        duration=cfg.duration,
        segment_length=cfg.segment_length,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    test_ds = DroneAudioSegmentDataset(
        test_dir,
        duration=cfg.duration,
        segment_length=cfg.segment_length,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = AudioVGG19TRIDENT().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, cfg.threshold)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, cfg.threshold)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_history(history, test_acc)


if __name__ == "__main__":
    main()
