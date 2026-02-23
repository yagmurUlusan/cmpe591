"""
Deliverable 1: MLP to predict object position.
Input:  raw before-image (3x128x128, flattened) + one-hot action (4,)
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

class PositionMLP(nn.Module):
    def __init__(self, img_channels=3, img_size=128, n_actions=4):
        super().__init__()
        img_flat = img_channels * img_size * img_size   # 3*128*128 = 49152
        self.net = nn.Sequential(
            nn.Linear(img_flat + n_actions, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, imgs, action_onehot):
        # imgs: (B, 3, 128, 128) → flatten to (B, 49152)
        x = imgs.view(imgs.size(0), -1)
        x = torch.cat([x, action_onehot], dim=1)   # (B, 49156)
        return self.net(x)                          # (B, 2)


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

def train(data_dir=".", model_path="deliverable1.pth",
          epochs=30, batch_size=32, lr=1e-3):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[D1] Using device: {device}")

    imgs, actions_onehot, positions = load_dataset(data_dir)

    dataset    = TensorDataset(imgs, actions_onehot, positions)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = PositionMLP().to(device)
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

        print(f"[D1 MLP] Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Save loss curve after every epoch in case of crash
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses,   label="Val")
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
        plt.title("Deliverable 1 – MLP Loss Curve")
        plt.legend(); plt.tight_layout()
        plt.savefig("d1_loss_curve.png"); plt.close()

    torch.save(model.state_dict(), model_path)
    print(f"[D1] Model saved → {model_path}")
    print("[D1] Loss curve saved → d1_loss_curve.png")
    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test(data_dir=".", model_path="deliverable1.pth"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    imgs, actions_onehot, positions = load_dataset(data_dir)

    model = PositionMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.MSELoss()
    # Test in batches to avoid memory issues
    total_loss = 0.0
    loader = DataLoader(TensorDataset(imgs, actions_onehot, positions), batch_size=32)
    with torch.no_grad():
        for img_b, act_b, pos_b in loader:
            img_b, act_b, pos_b = img_b.to(device), act_b.to(device), pos_b.to(device)
            total_loss += criterion(model(img_b, act_b), pos_b).item() * len(img_b)
    test_loss = total_loss / len(imgs)

    print(f"[D1 MLP] Test MSE Loss: {test_loss:.6f}")
    return test_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["train", "test", "both"], default="both")
    parser.add_argument("--data_dir",   default=".")
    parser.add_argument("--model_path", default="deliverable1.pth")
    parser.add_argument("--epochs",     type=int, default=30)
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        train(args.data_dir, args.model_path, args.epochs)
    if args.mode in ("test", "both"):
        test(args.data_dir, args.model_path)
