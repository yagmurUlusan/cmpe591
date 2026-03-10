# Homework 2 — Deep Q-Network (DQN)

**Course:** CMPE 591  
**Environment:** MuJoCo tabletop pushing task  
**Goal:** Train a robot arm to push an object to a target position using reinforcement learning, and build forward models that predict object positions and future images.

---

## Setup

```bash
# Clone and update the repo
git pull

# Activate environment
conda activate envforcmpe591

# Mac (required)
export KMP_DUPLICATE_LIB_OK=TRUE

# Linux headless server only
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Run everything
python dqn_solution.py
```

---

## Environment

The robot end-effector can move in **8 discrete directions** (evenly spaced on a circle). Each episode places the object and goal at random positions. The reward encourages the end-effector to reach the object and push it toward the goal:

```
reward = 1 / distance(ee, obj) + 1 / distance(obj, goal)
```

Episodes end after **50 steps** or when the object reaches the goal (within 0.01m).

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| State representation | High-level (6-dim) | `[ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]` |
| Number of actions | 8 | Discrete directions |
| Gamma | 0.99 | Discount factor |
| Epsilon start | 1.0 | Full exploration |
| Epsilon decay | 0.997 per 10 updates | Faster than default |
| Min epsilon | 0.05 | More exploitation |
| Learning rate | 3e-4 | Adam optimizer |
| LR schedule | StepLR × 0.5 | Every ~500 episodes |
| Batch size | 64 | Larger for MPS |
| Update frequency | Every 4 steps | |
| Target network sync | Every 200 updates | More stable |
| Replay buffer | 50,000 | Diverse experience |
| Training episodes | 2,000 | |
| Warmup steps | 1,000 | Pure random exploration |
| Device | MPS (Apple Silicon) | ~5-10x faster than CPU |

---

## Network Architecture

### DQN — MLP (high-level state)

```
Input: [ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]  →  6-dim

Linear(6, 256) → LayerNorm → ReLU
Linear(256, 256) → LayerNorm → ReLU
Linear(256, 128) → ReLU
Linear(128, 8)   →  Q-value for each action
```

**Double DQN:** Online network selects the best next action; frozen target network evaluates it. Prevents Q-value overestimation.  
**Huber Loss:** More robust than MSE for large errors, prevents gradient explosion.  
**LayerNorm:** Stabilises training across varying reward scales.

---

## Training Results

### Reward & RPS Curves

![Training Curves](training_curves.png)

| Metric | Ep 50 | Ep 500 | Ep 1000 | Ep 2000 |
|---|---|---|---|---|
| Avg Reward | ~5 | ~11 | ~16 | ~22 |
| Avg RPS | ~0.10 | ~0.24 | ~0.33 | ~0.45 |

- Reward grows steadily from ~5 to ~22 over 2000 episodes **(4.4x improvement)**
- LR decay at episodes 500, 1000, 1500 produces visible performance jumps
- High variance is expected — object and goal spawn randomly each episode
- Epsilon reaches minimum (0.05) around episode 850

### Evaluation (greedy policy, 10 episodes)

| Episode | Reward | RPS |
|---|---|---|
| 1 | 34.42 | 0.688 |
| 2 | 5.33 | 0.107 |
| 3 | 27.93 | 0.559 |
| 4 | 21.28 | 0.426 |
| 5 | 27.78 | 0.556 |
| 6 | 27.28 | 0.546 |
| 7 | 35.42 | 0.708 |
| 8 | 31.56 | 0.631 |
| 9 | 27.99 | 0.560 |
| 10 | 19.28 | 0.386 |
| **Mean** | **25.83** | **0.517** |

---

## Deliverables

### Deliverable 1 — Object Position Prediction (MLP)

Predicts the next object `(x, y)` position from a flattened image and one-hot action vector.

**Architecture:**
```
Flatten image: 3×128×128 = 49,152 values
Concat one-hot action: + 8 values  =  49,160 total input

Linear(49160, 512) → ReLU
Linear(512, 256)   → ReLU
Linear(256, 128)   → ReLU
Linear(128, 2)     →  (obj_x, obj_y)
```

**Results:**

![D1 Loss](d1_loss.png)

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 10 | 0.026022 | 0.150365 |
| 20 | 0.025086 | 0.118763 |
| 30 | 0.023993 | 0.061939 |
| 40 | 0.016260 | 0.098016 |
| 50 | 0.010550 | 0.031129 |

Converges but shows higher validation variance. Flattening the image loses all spatial structure, making position prediction harder.

---

### Deliverable 2 — Object Position Prediction (CNN)

Same prediction task using a CNN backbone to extract spatial features.

**Architecture:**
```
Conv2d(3→32,   4,2,1) → ReLU   # 128×128 → 64×64
Conv2d(32→64,  4,2,1) → ReLU   # 64×64   → 32×32
Conv2d(64→128, 4,2,1) → ReLU   # 32×32   → 16×16
Conv2d(128→256,4,2,1) → ReLU   # 16×16   → 8×8
Conv2d(256→512,4,2,1) → ReLU   # 8×8     → 4×4
AvgPool(spatial)      →  512-dim feature

Concat one-hot action (8)  →  520-dim
Linear(520, 256) → ReLU
Linear(256, 2)   →  (obj_x, obj_y)
```

**Results:**

![D2 Loss](d2_loss.png)

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 10 | 0.001335 | 0.001177 |
| 20 | 0.001102 | 0.001031 |
| 30 | 0.000719 | 0.001064 |
| 40 | 0.000442 | 0.000740 |
| 50 | 0.000357 | 0.000654 |

**~47x lower loss than MLP** (0.00065 vs 0.031). The CNN backbone preserves spatial structure and accurately localises the object from the image.

---

### Deliverable 3 — Post-Action Image Reconstruction

Reconstructs the next frame given the current image and the action taken.

**Architecture — UNet encoder-decoder with skip connections:**
```
Encoder:
  e1: Conv(3→32)   128→64
  e2: Conv(32→64)   64→32
  e3: Conv(64→128)  32→16
  e4: Conv(128→256) 16→8
  e5: Conv(256→512)  8→4  → AvgPool → 512-dim latent

Bottleneck:
  Linear(512 + 8_action, 512) → fused latent z

Decoder (skip connections from encoder at each level):
  DeConv(512→256) → concat(e4) → DeConv(512→128)
  DeConv(128→64)  → concat(e3) → DeConv(256→64)
  DeConv(64→32)   → concat(e2) → DeConv(128→32)
  DeConv(32→3)    → concat(e1) → DeConv(64→3) → Sigmoid
  Output: (3, 128, 128)
```

**Loss function:**
```python
loss = 0.5 × MSE(pred, target)   # pixel-level accuracy
     + 0.5 × SSIM(pred, target)  # structural similarity (edges, contrast, colour)
```

SSIM penalises differences in structure and colour — forcing the model to preserve small objects like the red cube and green circle that MSE alone ignores.

**Results:**

![D3 Loss](d3_loss.png)

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 10 | 0.051776 | 0.051925 |
| 20 | 0.049750 | 0.050597 |
| 30 | 0.048761 | 0.049714 |
| 40 | 0.048150 | 0.049339 |
| 50 | 0.047664 | 0.048996 |

**Sample Reconstructions:**

![D3 Reconstructions](d3_reconstructions.png)

The model successfully reconstructs the robot arm pose and the positions of the red cube and green goal circle. Skip connections preserve fine spatial details compared to a plain encoder-decoder.

---

## Key Takeaways

| Finding | Detail |
|---|---|
| MPS gives ~5-10x speedup | 2000 episodes feasible on MacBook Air |
| CNN >> MLP for position prediction | 47x lower MSE (0.00065 vs 0.031) |
| Double DQN + Huber loss | Stable training, no Q-value explosion |
| LR scheduling | Clear performance jumps at decay steps |
| SSIM + UNet skip connections | Small objects visible in reconstructions |

---

## Files in This Repository

| File | Description |
|---|---|
| `dqn_solution.py` | Main script — DQN agent + all 3 deliverables |
| `homework2.py` | Environment definition (provided by instructor) |
| `environment.py` | Base environment (provided by instructor) |
| `README.md` | This file |
| `training_curves.png` | DQN cumulative reward and RPS over 2000 episodes |
| `d1_loss.png` | Deliverable 1 — MLP train/val loss curve |
| `d2_loss.png` | Deliverable 2 — CNN train/val loss curve |
| `d3_loss.png` | Deliverable 3 — Reconstruction train/val loss curve |
| `d3_reconstructions.png` | Sample input / predicted / ground truth image comparisons |
| `dqn_checkpoint.pt` | Trained DQN weights — **[Google Drive](https://drive.google.com/drive/folders/1oRG1V8URITNCBWcqE6fBedMgUicpcTrp?usp=sharing)** |
| `obj_pos_mlp.pt` | D1 MLP weights — **[Google Drive](https://drive.google.com/drive/folders/1oRG1V8URITNCBWcqE6fBedMgUicpcTrp?usp=sharing)** |
| `obj_pos_cnn.pt` | D2 CNN weights — **[Google Drive](https://drive.google.com/drive/folders/1oRG1V8URITNCBWcqE6fBedMgUicpcTrp?usp=sharing)** |
| `img_recon.pt` | D3 reconstruction weights — **[Google Drive](https://drive.google.com/drive/folders/1oRG1V8URITNCBWcqE6fBedMgUicpcTrp?usp=sharing)** |

> **Model Weights (Google Drive):** [`dqn_checkpoint.pt`, `obj_pos_mlp.pt`, `obj_pos_cnn.pt`, `img_recon.pt`](https://drive.google.com/drive/folders/1oRG1V8URITNCBWcqE6fBedMgUicpcTrp?usp=sharing)  
> Download and place all `.pt` files into the `src/` folder before running evaluation.

---

## How to Reproduce

```bash
git clone https://github.com/cmpe591/your-repo
cd your-repo/src
conda activate envforcmpe591
export KMP_DUPLICATE_LIB_OK=TRUE
python dqn_solution.py
# Automatically trains DQN for 2000 episodes, then runs D1, D2, D3
```
