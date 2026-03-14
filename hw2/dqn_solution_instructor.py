"""
Deep Q-Network (DQN) for Homework 2
=====================================
Three hyperparameter runs are compared:
  Run 1 (baseline) : instructor-provided default hyperparameters
  Run 2 (improved) : larger buffer, soft update, LR scheduler, Huber loss
  Run 3 (best)     : further tuned eps decay, batch size, LayerNorm

State:  high_level_state() → [ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]  (6-dim)
Images: env.state()        → torch.Tensor (3, 128, 128) float32 in [0,1]
Device: MPS (Apple Silicon) > CUDA > CPU
"""

import random
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from homework2 import Hw2Env

# ─────────────────────────────────────────────
# 1.  DEVICE  —  prefer MPS (Apple Silicon) > CUDA > CPU
# ─────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2.  HYPER-PARAMETER CONFIGURATIONS
#     Run 1 = instructor baseline
#     Run 2 = improved (our tuning)
#     Run 3 = best
# ─────────────────────────────────────────────
N_ACTIONS = 8

CONFIGS = {
    "run1_baseline": {
        # Instructor-provided default hyperparameters
        "n_episodes":    2500,
        "buffer_length": 10_000,
        "batch_size":    128,
        "eps_start":     0.9,
        "eps_end":       0.05,
        "eps_decay":     10_000,   # step-based linear decay
        "gamma":         0.99,
        "lr":            1e-4,
        "tau":           0.005,    # soft target update
        "warmup_steps":  1000,
        "update_freq":   4,
        "lr_step":       None,     # no LR scheduling
    },
    "run2_improved": {
        # Larger buffer, soft update, LR scheduler, Huber loss
        "n_episodes":    2000,
        "buffer_length": 50_000,
        "batch_size":    64,
        "eps_start":     1.0,
        "eps_end":       0.05,
        "eps_decay":     0.997,    # multiplicative decay
        "gamma":         0.99,
        "lr":            3e-4,
        "tau":           0.005,    # soft target update
        "warmup_steps":  1000,
        "update_freq":   4,
        "lr_step":       500,
    },
    "run3_best": {
        # Further tuned: faster eps decay, LayerNorm, bigger buffer
        "n_episodes":    2000,
        "buffer_length": 100_000,
        "batch_size":    128,
        "eps_start":     1.0,
        "eps_end":       0.05,
        "eps_decay":     0.995,    # even faster decay
        "gamma":         0.99,
        "lr":            3e-4,
        "tau":           0.01,     # faster soft update
        "warmup_steps":  500,
        "update_freq":   4,
        "lr_step":       400,
    },
}

# Active config — change this to run different experiments
CFG = CONFIGS["run1_baseline"]

# Unpack active config into module-level variables (used by existing code)
GAMMA              = CFG["gamma"]
EPSILON_START      = CFG["eps_start"]
MIN_EPSILON        = CFG["eps_end"]
EPSILON_DECAY      = CFG["eps_decay"]
EPSILON_DECAY_ITER = 10
LEARNING_RATE      = CFG["lr"]
BATCH_SIZE         = CFG["batch_size"]
UPDATE_FREQ        = CFG["update_freq"]
BUFFER_LENGTH      = CFG["buffer_length"]
N_EPISODES         = CFG["n_episodes"]
WARMUP_STEPS       = CFG["warmup_steps"]
TAU                = CFG["tau"]
LR_STEP            = CFG["lr_step"]
LR_GAMMA           = 0.5


# ─────────────────────────────────────────────
# 3.  REPLAY BUFFER
# ─────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# 4.  Q-NETWORK  (MLP — high-level state)
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Input:  [ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]  (6-dim)
    Output: Q-value for each of the 8 actions
    Deeper network with BatchNorm for more stable training.
    """
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 5.  CNN Q-NETWORK  (optional, raw pixels)
# ─────────────────────────────────────────────
class QNetworkCNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(32,  64,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(64,  128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),
        )
        self.head = nn.Linear(512, n_actions)
    def forward(self, x):
        return self.head(self.conv(x).mean(dim=[2, 3]))


# ─────────────────────────────────────────────
# 6.  DQN AGENT  (Double DQN + Huber loss + LR scheduler)
# ─────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim, n_actions):
        self.n_actions    = n_actions
        self.epsilon      = EPSILON_START
        self.update_count = 0
        self.step_count   = 0   # global env steps for linear eps decay

        self.q_net      = QNetwork(state_dim, n_actions).to(DEVICE)
        self.target_net = QNetwork(state_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        if LR_STEP is not None:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=LR_STEP * 50 // UPDATE_FREQ,
                gamma=LR_GAMMA
            )
        else:
            self.scheduler = None
        self.buffer = ReplayBuffer(BUFFER_LENGTH)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            return int(self.q_net(state_t).argmax(dim=1).item())

    def update_epsilon_linear(self):
        """Linear epsilon decay used when eps_decay is an int (instructor style)."""
        progress = min(1.0, self.step_count / EPSILON_DECAY)
        self.epsilon = EPSILON_START - progress * (EPSILON_START - MIN_EPSILON)

    def soft_update_target(self):
        """Polyak averaging: target = tau*online + (1-tau)*target"""
        for t_param, o_param in zip(self.target_net.parameters(),
                                    self.q_net.parameters()):
            t_param.data.copy_(TAU * o_param.data + (1.0 - TAU) * t_param.data)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return None

        batch       = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(np.array([t.state      for t in batch])).to(DEVICE)
        actions     = torch.LongTensor (np.array([t.action     for t in batch])).to(DEVICE)
        rewards     = torch.FloatTensor(np.array([t.reward     for t in batch])).to(DEVICE)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(DEVICE)
        dones       = torch.FloatTensor(np.array([t.done       for t in batch])).to(DEVICE)

        # Q(s, a)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions  = self.q_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + GAMMA * next_q_values * (1.0 - dones)

        # Huber loss
        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.update_count += 1

        # Epsilon decay — linear (int) or multiplicative (float)
        if isinstance(EPSILON_DECAY, int):
            self.update_epsilon_linear()
        else:
            if self.update_count % EPSILON_DECAY_ITER == 0:
                self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

        # Soft target update every step
        self.soft_update_target()

        return loss.item()

    def save(self, path="dqn_checkpoint.pt"):
        torch.save({
            "q_net":        self.q_net.state_dict(),
            "epsilon":      self.epsilon,
            "update_count": self.update_count,
        }, path)
        print(f"Saved → {path}")

    def load(self, path="dqn_checkpoint.pt"):
        ckpt = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.epsilon      = ckpt["epsilon"]
        self.update_count = ckpt["update_count"]
        print(f"Loaded ← {path}")


# ─────────────────────────────────────────────
# 7.  TRAINING LOOP
# ─────────────────────────────────────────────
def train_dqn(n_episodes=N_EPISODES, render=False):
    render_mode = "gui" if render else "offscreen"
    env = Hw2Env(n_actions=N_ACTIONS, render_mode=render_mode)

    env.reset()
    state_dim = len(env.high_level_state())
    print(f"State dim: {state_dim}  |  Device: {DEVICE}")

    agent      = DQNAgent(state_dim, N_ACTIONS)
    ep_rewards = []
    ep_rps     = []
    step_count = 0

    for episode in range(n_episodes):
        env.reset()
        state      = env.high_level_state()
        done       = False
        cum_reward = 0.0
        ep_steps   = 0

        while not done:
            if step_count < WARMUP_STEPS:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                action = agent.select_action(state)

            _, reward, is_terminal, is_truncated = env.step(action)
            done       = is_terminal or is_truncated
            next_state = env.high_level_state()

            agent.buffer.push(state, action, reward, next_state, float(done))
            state       = next_state
            cum_reward += reward
            ep_steps   += 1
            step_count += 1
            agent.step_count = step_count  # for linear eps decay

            if step_count >= WARMUP_STEPS and step_count % UPDATE_FREQ == 0:
                agent.update()

        ep_rewards.append(cum_reward)
        ep_rps.append(cum_reward / max(ep_steps, 1))

        if (episode + 1) % 50 == 0:
            avg_r   = np.mean(ep_rewards[-50:])
            avg_rps = np.mean(ep_rps[-50:])
            lr_now  = agent.optimizer.param_groups[0]["lr"]
            print(f"Ep {episode+1:5d} | Reward {avg_r:7.3f} | "
                  f"RPS {avg_rps:.4f} | ε={agent.epsilon:.3f} | "
                  f"lr={lr_now:.2e} | steps={step_count}")

    agent.save("dqn_checkpoint.pt")
    return agent, ep_rewards, ep_rps


# ─────────────────────────────────────────────
# 8.  EVALUATION
# ─────────────────────────────────────────────
def evaluate_dqn(agent, n_episodes=10, render=False):
    render_mode = "gui" if render else "offscreen"
    env = Hw2Env(n_actions=N_ACTIONS, render_mode=render_mode)
    rewards, rpss = [], []

    for ep in range(n_episodes):
        env.reset()
        state = env.high_level_state()
        done, cum_reward, steps = False, 0.0, 0
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action  = int(agent.q_net(state_t).argmax(1).item())
            _, reward, is_terminal, is_truncated = env.step(action)
            done        = is_terminal or is_truncated
            state       = env.high_level_state()
            cum_reward += reward
            steps      += 1
        rewards.append(cum_reward)
        rpss.append(cum_reward / max(steps, 1))
        print(f"Eval ep {ep+1:2d}: reward={cum_reward:.3f}  rps={rpss[-1]:.4f}")

    print(f"\nMean reward: {np.mean(rewards):.3f} | Mean RPS: {np.mean(rpss):.4f}")


# ─────────────────────────────────────────────
# 9.  PLOTTING
# ─────────────────────────────────────────────
def plot_training(rewards, rpss, window=50, save_path="training_curves.png"):
    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(rewards, alpha=0.2, color="steelblue", label="Raw")
    if len(rewards) >= window:
        axes[0].plot(range(window-1, len(rewards)), smooth(rewards, window),
                     color="steelblue", linewidth=2, label=f"{window}-ep avg")
    axes[0].set_title("Cumulative Reward per Episode")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(rpss, alpha=0.2, color="coral", label="Raw")
    if len(rpss) >= window:
        axes[1].plot(range(window-1, len(rpss)), smooth(rpss, window),
                     color="coral", linewidth=2, label=f"{window}-ep avg")
    axes[1].set_title("Reward Per Step (RPS) per Episode")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("RPS")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERABLES
# ─────────────────────────────────────────────────────────────────────────────

def collect_forward_model_data(n_episodes=200):
    """
    Collect (image_before, action_onehot, obj_pos_after).
    high_level_state() → [ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]
    obj_pos = hl[2:4]
    env.state() → torch.Tensor (3,128,128) in [0,1]
    """
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    images, actions_oh, obj_positions = [], [], []

    for ep in range(n_episodes):
        env.reset()
        done = False
        while not done:
            img    = env.state().numpy().astype(np.float32)
            action = random.randint(0, N_ACTIONS - 1)
            act_oh = np.zeros(N_ACTIONS, dtype=np.float32)
            act_oh[action] = 1.0

            _, _, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated

            hl      = env.high_level_state()
            obj_pos = hl[2:4].astype(np.float32)

            images.append(img)
            actions_oh.append(act_oh)
            obj_positions.append(obj_pos)

        if (ep + 1) % 20 == 0:
            print(f"  D1/D2 data: ep {ep+1}/{n_episodes} ({len(images)} samples)")

    return np.array(images), np.array(actions_oh), np.array(obj_positions)


# ── D1: Object position MLP ──────────────────────────────────────────────────
class ObjPosMLP(nn.Module):
    """Flatten image + one-hot action → MLP → (obj_x, obj_y)"""
    def __init__(self, img_size=128, n_actions=N_ACTIONS):
        super().__init__()
        in_dim = 3 * img_size * img_size + n_actions
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512,    256), nn.ReLU(),
            nn.Linear(256,    128), nn.ReLU(),
            nn.Linear(128,    2),
        )
    def forward(self, img_flat, action_oh):
        return self.net(torch.cat([img_flat, action_oh], dim=1))


def train_obj_pos_mlp(images, actions_oh, obj_positions, epochs=50, lr=1e-3):
    print("\n=== Deliverable 1: Object Position MLP ===")
    imgs_flat = images.reshape(len(images), -1)
    X_img = torch.FloatTensor(imgs_flat)
    X_act = torch.FloatTensor(actions_oh)
    Y     = torch.FloatTensor(obj_positions)

    split = int(0.9 * len(Y))
    # num_workers=0 required on macOS
    train_loader = DataLoader(TensorDataset(X_img[:split], X_act[:split], Y[:split]),
                              batch_size=128, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_img[split:], X_act[split:], Y[split:]),
                              batch_size=128, num_workers=0)

    model = ObjPosMLP().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    tl, vl = [], []

    for epoch in range(epochs):
        model.train(); t = 0.0
        for ib, ab, yb in train_loader:
            ib, ab, yb = ib.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
            loss = F.mse_loss(model(ib, ab), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            t += loss.item()

        model.eval(); v = 0.0
        with torch.no_grad():
            for ib, ab, yb in val_loader:
                ib, ab, yb = ib.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
                v += F.mse_loss(model(ib, ab), yb).item()

        tl.append(t / len(train_loader))
        vl.append(v / len(val_loader))
        sched.step(vl[-1])

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train={tl[-1]:.6f} | val={vl[-1]:.6f} | "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    torch.save(model.state_dict(), "obj_pos_mlp.pt")
    _plot_losses(tl, vl, "D1: Object Position MLP", "d1_loss.png")
    return model


# ── D2: Object position CNN ───────────────────────────────────────────────────
class ObjPosCNN(nn.Module):
    """CNN backbone + one-hot action → (obj_x, obj_y)"""
    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(32,  64,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(64,  128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + n_actions, 256), nn.ReLU(),
            nn.Linear(256, 2),
        )
    def forward(self, img, action_oh):
        feat = self.conv(img).mean(dim=[2, 3])
        return self.head(torch.cat([feat, action_oh], dim=1))


def train_obj_pos_cnn(images, actions_oh, obj_positions, epochs=50, lr=1e-4):
    print("\n=== Deliverable 2: Object Position CNN ===")
    X_img = torch.FloatTensor(images)
    X_act = torch.FloatTensor(actions_oh)
    Y     = torch.FloatTensor(obj_positions)

    split = int(0.9 * len(Y))
    train_loader = DataLoader(TensorDataset(X_img[:split], X_act[:split], Y[:split]),
                              batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_img[split:], X_act[split:], Y[split:]),
                              batch_size=64, num_workers=0)

    model = ObjPosCNN().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    tl, vl = [], []

    for epoch in range(epochs):
        model.train(); t = 0.0
        for ib, ab, yb in train_loader:
            ib, ab, yb = ib.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
            loss = F.mse_loss(model(ib, ab), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            t += loss.item()

        model.eval(); v = 0.0
        with torch.no_grad():
            for ib, ab, yb in val_loader:
                ib, ab, yb = ib.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
                v += F.mse_loss(model(ib, ab), yb).item()

        tl.append(t / len(train_loader))
        vl.append(v / len(val_loader))
        sched.step(vl[-1])

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train={tl[-1]:.6f} | val={vl[-1]:.6f} | "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    torch.save(model.state_dict(), "obj_pos_cnn.pt")
    _plot_losses(tl, vl, "D2: Object Position CNN", "d2_loss.png")
    return model


# ── D3: Image reconstruction ─────────────────────────────────────────────────

def ssim_loss(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Structural Similarity (SSIM) loss.
    Compares luminance, contrast and structure — much better than MSE alone
    at preserving small colourful objects (red cube, green circle).
    Returns 1 - SSIM  (so minimising this maximises similarity).
    """
    # Gaussian kernel for local statistics
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)                 # (W, W)
    kernel = kernel.unsqueeze(0).unsqueeze(0)                # (1,1,W,W)
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size).contiguous()

    pad = window_size // 2

    def _conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])

    mu_x    = _conv(pred)
    mu_y    = _conv(target)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    sig_x  = _conv(pred   * pred)   - mu_x_sq
    sig_y  = _conv(target * target) - mu_y_sq
    sig_xy = _conv(pred   * target) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sig_x + sig_y + C2))

    return 1.0 - ssim_map.mean()


def combined_loss(pred, target, alpha=0.5):
    """
    alpha  * MSE  +  (1-alpha) * SSIM_loss
    alpha=0.5 balances pixel accuracy with structural similarity.
    """
    return alpha * F.mse_loss(pred, target) + (1 - alpha) * ssim_loss(pred, target)


class ImageReconNet(nn.Module):
    """
    UNet-style encoder-decoder with skip connections.
    Skip connections pass encoder feature maps directly to the decoder,
    helping the network reconstruct fine details (small objects, sharp edges).

    Architecture:
        Encoder: 5 conv blocks, each halves spatial resolution
        Bottleneck: fuse 512-d latent with action one-hot
        Decoder: 5 deconv blocks with skip connections from encoder
    """
    def __init__(self, n_actions=N_ACTIONS, latent_dim=512):
        super().__init__()

        # Encoder blocks (each stored separately for skip connections)
        self.enc1 = nn.Sequential(nn.Conv2d(3,   32,  4, 2, 1), nn.ReLU())  # 128→64
        self.enc2 = nn.Sequential(nn.Conv2d(32,  64,  4, 2, 1), nn.ReLU())  # 64→32
        self.enc3 = nn.Sequential(nn.Conv2d(64,  128, 4, 2, 1), nn.ReLU())  # 32→16
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU())  # 16→8
        self.enc5 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU())  # 8→4

        # Bottleneck: fuse spatial average of enc5 with action
        self.fuse          = nn.Linear(latent_dim + n_actions, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)

        # Decoder blocks — input channels doubled because of skip connections
        self.dec5 = nn.Sequential(nn.ConvTranspose2d(512,       256, 4, 2, 1), nn.ReLU())  # 4→8
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(256 + 256, 128, 4, 2, 1), nn.ReLU())  # 8→16  (skip from enc4)
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64,  4, 2, 1), nn.ReLU())  # 16→32 (skip from enc3)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64  + 64,  32,  4, 2, 1), nn.ReLU())  # 32→64 (skip from enc2)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(32  + 32,  3,   4, 2, 1), nn.Sigmoid()) # 64→128 (skip from enc1)

    def forward(self, img, action_oh):
        # Encode — save intermediate feature maps for skip connections
        e1 = self.enc1(img)    # (B, 32,  64, 64)
        e2 = self.enc2(e1)     # (B, 64,  32, 32)
        e3 = self.enc3(e2)     # (B, 128, 16, 16)
        e4 = self.enc4(e3)     # (B, 256, 8,  8)
        e5 = self.enc5(e4)     # (B, 512, 4,  4)

        # Bottleneck — fuse with action
        z = e5.mean(dim=[2, 3])                                    # (B, 512)
        z = F.relu(self.fuse(torch.cat([z, action_oh], dim=1)))    # (B, 512)
        z = self.decoder_input(z).view(-1, 512, 4, 4)             # (B, 512, 4, 4)

        # Decode — concatenate skip connections at each level
        d5 = self.dec5(z)                              # (B, 256, 8,  8)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))    # (B, 128, 16, 16)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))    # (B, 64,  32, 32)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))    # (B, 32,  64, 64)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))    # (B, 3,   128, 128)
        return d1


def collect_image_pairs(n_episodes=200):
    """Collect (image_before, action_onehot, image_after)"""
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    images, actions_oh, next_images = [], [], []

    for ep in range(n_episodes):
        env.reset()
        done = False
        while not done:
            img    = env.state().numpy().astype(np.float32)
            action = random.randint(0, N_ACTIONS - 1)
            act_oh = np.zeros(N_ACTIONS, dtype=np.float32)
            act_oh[action] = 1.0

            _, _, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated

            next_img = env.state().numpy().astype(np.float32)
            images.append(img)
            actions_oh.append(act_oh)
            next_images.append(next_img)

        if (ep + 1) % 20 == 0:
            print(f"  D3 data: ep {ep+1}/{n_episodes} ({len(images)} samples)")

    return np.array(images), np.array(actions_oh), np.array(next_images)


def train_image_recon(images, actions_oh, next_images, epochs=50, lr=1e-4):
    """
    Trains ImageReconNet with combined MSE + SSIM loss.
    MSE  → pixel-level accuracy
    SSIM → structural similarity (preserves edges, small objects, colours)
    """
    print("\n=== Deliverable 3: Image Reconstruction (MSE + SSIM loss) ===")
    X_img = torch.FloatTensor(images)
    X_act = torch.FloatTensor(actions_oh)
    Y_img = torch.FloatTensor(next_images)

    split = int(0.9 * len(Y_img))
    train_loader = DataLoader(TensorDataset(X_img[:split], X_act[:split], Y_img[:split]),
                              batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_img[split:], X_act[split:], Y_img[split:]),
                              batch_size=32, num_workers=0)
    val_ds = TensorDataset(X_img[split:], X_act[split:], Y_img[split:])

    model = ImageReconNet().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    tl, vl = [], []

    for epoch in range(epochs):
        model.train(); t = 0.0
        for ib, ab, yb in train_loader:
            ib, ab, yb = ib.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
            loss = combined_loss(model(ib, ab), yb)   # ← MSE + SSIM
            opt.zero_grad(); loss.backward(); opt.step()
            t += loss.item()

        model.eval(); v = 0.0
        with torch.no_grad():
            for ib, ab, yb in val_loader:
                ib, ab, yb = ib.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
                v += combined_loss(model(ib, ab), yb).item()

        tl.append(t / len(train_loader))
        vl.append(v / len(val_loader))
        sched.step(vl[-1])

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train={tl[-1]:.6f} | val={vl[-1]:.6f} | "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    torch.save(model.state_dict(), "img_recon.pt")
    _plot_losses(tl, vl, "D3: Image Reconstruction (MSE + SSIM)", "d3_loss.png")
    _visualise_reconstruction(model, val_ds)
    return model


def _visualise_reconstruction(model, val_ds, n=4, save_path="d3_reconstructions.png"):
    model.eval()
    idxs = random.sample(range(len(val_ds)), min(n, len(val_ds)))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3*n))
    for i, idx in enumerate(idxs):
        img, act, gt = val_ds[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(DEVICE),
                         act.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        for j, frame in enumerate([img, pred, gt]):
            axes[i, j].imshow(frame.permute(1, 2, 0).numpy())
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(["Input", "Predicted", "Ground Truth"][j])
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Reconstruction examples → {save_path}")


def _plot_losses(tl, vl, title, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(tl, label="Train")
    plt.plot(vl, label="Val")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print(f"STEP 1: Training DQN — Active config: {[k for k,v in CONFIGS.items() if v is CFG][0]}")
    print(f"        Device={DEVICE} | Episodes={N_EPISODES} | Buffer={BUFFER_LENGTH} | Batch={BATCH_SIZE} | tau={TAU}")
    print("=" * 60)
    agent, rewards, rpss = train_dqn(n_episodes=N_EPISODES, render=False)
    plot_training(rewards, rpss, save_path="training_curves.png")

    print("\n" + "=" * 60)
    print("STEP 2: Evaluating DQN agent")
    print("=" * 60)
    evaluate_dqn(agent, n_episodes=10, render=False)

    print("\n" + "=" * 60)
    print("STEP 3: Collecting data for D1 & D2 (300 episodes)")
    print("=" * 60)
    images, actions_oh, obj_positions = collect_forward_model_data(n_episodes=300)
    print(f"Samples: {len(images)} | images: {images.shape} | obj_pos: {obj_positions.shape}")

    train_obj_pos_mlp(images, actions_oh, obj_positions, epochs=50)
    train_obj_pos_cnn(images, actions_oh, obj_positions, epochs=50)

    print("\n" + "=" * 60)
    print("STEP 4: Collecting image pairs for D3 (300 episodes)")
    print("=" * 60)
    imgs, acts, next_imgs = collect_image_pairs(n_episodes=300)
    print(f"Image pairs: {len(imgs)}")
    train_image_recon(imgs, acts, next_imgs, epochs=50)

    print("\n✓ All done!")
    print("  training_curves.png  d1_loss.png  d2_loss.png")
    print("  d3_loss.png  d3_reconstructions.png")
    print("  dqn_checkpoint.pt  obj_pos_mlp.pt  obj_pos_cnn.pt  img_recon.pt")
