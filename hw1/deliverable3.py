"""
Deliverable 3: CNN image prediction.
Input:  raw before-image (3x128x128) + one-hot action (4,)
Output: predicted after-image — guaranteed 3x128x128

Simple and fast model designed for 8GB Apple Silicon Macs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Very simple model: encode image → inject action → decode to image
# No skip connections, tiny channel sizes → very fast
# ---------------------------------------------------------------------------

class ImagePredictionCNN(nn.Module):
    def __init__(self, n_actions=4):
        super().__init__()

        # Encoder: 128x128 → 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  16, 4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
        )

        # Action injection
        self.action_fc = nn.Linear(64 * 8 * 8 + n_actions, 64 * 8 * 8)

        # Decoder: 8x8 → 128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16,  3, 4, stride=2, padding=1),  # 128x128
            nn.Sigmoid(),
        )

    def forward(self, img, action_onehot):
        b = img.size(0)
        z = self.encoder(img).view(b, -1)               # (B, 64*8*8)
        z = torch.cat([z, action_onehot], dim=1)        # (B, 64*8*8 + 4)
        z = F.relu(self.action_fc(z)).view(b, 64, 8, 8) # (B, 64, 8, 8)
        out = self.decoder(z)                            # (B, 3, 128, 128)
        # Guarantee 128x128
        if out.shape[-2:] != (128, 128):
            out = F.interpolate(out, size=(128, 128), mode='bilinear', align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(data_dir=".", max_samples=400):
    imgs_before_list, actions_list, imgs_after_list = [], [], []
    for idx in range(4):
        imgs_before_list.append(torch.load(os.path.join(data_dir, f"imgs_before_{idx}.pt")))
        actions_list.append(torch.load(os.path.join(data_dir, f"actions_{idx}.pt")))
        imgs_after_list.append(torch.load(os.path.join(data_dir, f"imgs_{idx}.pt")))

    imgs_before = torch.cat(imgs_before_list).float() / 255.0
    actions     = torch.cat(actions_list).long()
    imgs_after  = torch.cat(imgs_after_list).float()  / 255.0

    imgs_before = imgs_before[:max_samples]
    actions     = actions[:max_samples]
    imgs_after  = imgs_after[:max_samples]

    actions_onehot = torch.zeros(len(actions), 4)
    actions_onehot.scatter_(1, actions.unsqueeze(1), 1)

    return imgs_before, actions_onehot, imgs_after


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(data_dir=".", model_path="deliverable3.pth", epochs=10, batch_size=8, lr=1e-3):
    device = get_device()
    print(f"[D3] Using device: {device}")

    imgs_before, actions_onehot, imgs_after = load_dataset(data_dir)
    print(f"[D3] Loaded {len(imgs_before)} samples")

    dataset    = TensorDataset(imgs_before, actions_onehot, imgs_after)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=0)

    model     = ImagePredictionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for img_b, act_b, tgt_b in train_loader:
            img_b = img_b.to(device)
            act_b = act_b.to(device)
            tgt_b = tgt_b.to(device)
            optimizer.zero_grad()
            pred = model(img_b, act_b)
            loss = criterion(pred, tgt_b)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(img_b)
        train_loss = running / train_size
        train_losses.append(train_loss)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for img_b, act_b, tgt_b in val_loader:
                img_b = img_b.to(device)
                act_b = act_b.to(device)
                tgt_b = tgt_b.to(device)
                running += criterion(model(img_b, act_b), tgt_b).item() * len(img_b)
        val_loss = running / val_size
        val_losses.append(val_loss)

        print(f"[D3] Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        # Save after every epoch
        torch.save(model.state_dict(), model_path)
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses,   label="Val")
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
        plt.title("Deliverable 3 – Image Prediction Loss Curve")
        plt.legend(); plt.tight_layout()
        plt.savefig("d3_loss_curve.png"); plt.close()

    print(f"[D3] Done! Model → {model_path}")
    print("[D3] Loss curve → d3_loss_curve.png")
    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test(data_dir=".", model_path="deliverable3.pth", n_vis=5):
    device = get_device()

    imgs_before, actions_onehot, imgs_after = load_dataset(data_dir)

    model = ImagePredictionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion  = nn.MSELoss()
    total_loss = 0.0
    all_preds  = []

    loader = DataLoader(TensorDataset(imgs_before, actions_onehot, imgs_after),
                        batch_size=8, num_workers=0)
    with torch.no_grad():
        for img_b, act_b, tgt_b in loader:
            pred = model(img_b.to(device), act_b.to(device))
            total_loss += criterion(pred, tgt_b.to(device)).item() * len(img_b)
            all_preds.append(pred.cpu())

    test_loss = total_loss / len(imgs_before)
    preds     = torch.cat(all_preds)

    print(f"[D3] Test MSE Loss: {test_loss:.6f}")
    print(f"[D3] Output shape:  {preds.shape}")  # should be (N, 3, 128, 128)

    # Visualise
    indices = np.random.choice(len(imgs_after), n_vis, replace=False)
    fig, axes = plt.subplots(n_vis, 3, figsize=(10, n_vis * 3))
    for col, title in enumerate(["Before (Input)", "After (Ground Truth)", "After (Predicted)"]):
        axes[0, col].set_title(title, fontsize=12)
    for row, idx in enumerate(indices):
        for col, img_t in enumerate([imgs_before[idx], imgs_after[idx], preds[idx]]):
            axes[row, col].imshow(img_t.permute(1, 2, 0).numpy())
            axes[row, col].axis("off")

    plt.suptitle("Deliverable 3 – Reconstruction Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig("d3_reconstructions.png")
    plt.close()
    print("[D3] Reconstructions → d3_reconstructions.png")
    return test_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["train", "test", "both"], default="both")
    parser.add_argument("--data_dir",   default=".")
    parser.add_argument("--model_path", default="deliverable3.pth")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--n_vis",      type=int, default=5)
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        train(args.data_dir, args.model_path, args.epochs)
    if args.mode in ("test", "both"):
        test(args.data_dir, args.model_path, args.n_vis)
