"""
Deliverable 2: CNN to predict object position.
Input:  raw before-image (3x128x128) + one-hot action (4,)
Output: (x, y) position after the action
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PositionCNN(nn.Module):
    def __init__(self, n_actions=4):
        super().__init__()
        # Convolutional encoder: 3x128x128 → 256x4x4
        self.cnn = nn.Sequential(
            nn.Conv2d(3,   32,  5, stride=2, padding=2),   # → 32 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(32,  64,  3, stride=2, padding=1),   # → 64 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1),   # → 128 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),   # → 256 x 8 x 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                        # → 256 x 4 x 4
        )
        cnn_out_dim = 256 * 4 * 4  # 4096

        # Fully connected head: CNN features + action → position
        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, imgs, action_onehot):
        # imgs: (B, 3, 128, 128)
        feat = self.cnn(imgs).view(imgs.size(0), -1)  # (B, 4096)
        x = torch.cat([feat, action_onehot], dim=1)               # (B, 4100)
        return self.head(x)                                        # (B, 2)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(data_dir="."):
    """
    Uses teacher's collected data directly.
    Loads: imgs_{idx}.pt, actions_{idx}.pt, positions_{idx}.pt
    """
    imgs_list, actions_list, positions_list = [], [], []
    for idx in range(4):
        imgs_list.append(torch.load(os.path.join(data_dir, f"imgs_{idx}.pt")))
        actions_list.append(torch.load(os.path.join(data_dir, f"actions_{idx}.pt")))
        positions_list.append(torch.load(os.path.join(data_dir, f"positions_{idx}.pt")))

    imgs      = torch.cat(imgs_list).float() / 255.0  # (N, 3, 128, 128)
    actions   = torch.cat(actions_list).long()         # (N,)
    positions = torch.cat(positions_list).float()      # (N, 2)

    actions_onehot = torch.zeros(len(actions), 4)
    actions_onehot.scatter_(1, actions.unsqueeze(1), 1)  # (N, 4)

    return imgs, actions_onehot, positions


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(data_dir=".", model_path="deliverable2.pth",
          epochs=30, batch_size=32, lr=1e-3):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[D2] Using device: {device}")

    imgs, actions_onehot, positions = load_dataset(data_dir)

    dataset    = TensorDataset(imgs, actions_onehot, positions)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = PositionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        running = 0.0
        for img_b, act_b, pos_b in train_loader:
            img_b, act_b, pos_b = img_b.to(device), act_b.to(device), pos_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img_b, act_b), pos_b)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(img_b)
        train_loss = running / train_size
        train_losses.append(train_loss)

        # --- validate ---
        model.eval()
        running = 0.0
        with torch.no_grad():
            for img_b, act_b, pos_b in val_loader:
                img_b, act_b, pos_b = img_b.to(device), act_b.to(device), pos_b.to(device)
                running += criterion(model(img_b, act_b), pos_b).item() * len(img_b)
        val_loss = running / val_size
        val_losses.append(val_loss)

        print(f"[D2 CNN] Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Save loss curve after every epoch in case of crash
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses,   label="Val")
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
        plt.title("Deliverable 2 – CNN Loss Curve")
        plt.legend(); plt.tight_layout()
        plt.savefig("d2_loss_curve.png"); plt.close()

    torch.save(model.state_dict(), model_path)
    print(f"[D2] Model saved → {model_path}")
    print("[D2] Loss curve saved → d2_loss_curve.png")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test(data_dir=".", model_path="deliverable2.pth"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    imgs, actions_onehot, positions = load_dataset(data_dir)

    model = PositionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0
    loader = DataLoader(TensorDataset(imgs, actions_onehot, positions), batch_size=32)
    with torch.no_grad():
        for img_b, act_b, pos_b in loader:
            img_b, act_b, pos_b = img_b.to(device), act_b.to(device), pos_b.to(device)
            total_loss += criterion(model(img_b, act_b), pos_b).item() * len(img_b)
    test_loss = total_loss / len(imgs)

    print(f"[D2 CNN] Test MSE Loss: {test_loss:.6f}")
    return test_loss

    print(f"[D2 CNN] Test MSE Loss: {test_loss:.6f}")
    return test_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["train", "test", "both"], default="both")
    parser.add_argument("--data_dir",   default=".")
    parser.add_argument("--model_path", default="deliverable2.pth")
    parser.add_argument("--epochs",     type=int, default=30)
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        train(args.data_dir, args.model_path, args.epochs)
    if args.mode in ("test", "both"):
        test(args.data_dir, args.model_path)
