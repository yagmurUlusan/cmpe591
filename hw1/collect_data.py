"""
collect_data.py  –  Data collection script for HW1.
Imports Hw1Env from homework1.py (teacher's file, not modified).
Collects 1000 samples using 4 parallel processes (250 each) and saves:
  - imgs_before_{idx}.pt  : image BEFORE the action  (3 x 128 x 128)
  - imgs_{idx}.pt         : image AFTER  the action  (3 x 128 x 128)
  - positions_{idx}.pt    : object (x, y) position after the action
  - actions_{idx}.pt      : action id (0-3)

Run:
    python collect_data.py
"""

from multiprocessing import Process

import numpy as np
import torch

from homework1 import Hw1Env   # teacher's file — not modified


def collect(idx, N):
    env = Hw1Env(render_mode="offscreen")

    positions   = torch.zeros(N, 2,       dtype=torch.float)
    actions     = torch.zeros(N,          dtype=torch.uint8)
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after  = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)

    for i in range(N):
        env.reset()
        action_id = np.random.randint(4)

        # Capture state BEFORE the action
        _, pixels_before = env.state()

        # Execute action
        env.step(action_id)

        # Capture state AFTER the action
        obj_pos, pixels_after = env.state()

        positions[i]   = torch.tensor(obj_pos)
        actions[i]     = action_id
        imgs_before[i] = pixels_before
        imgs_after[i]  = pixels_after

    torch.save(positions,   f"positions_{idx}.pt")
    torch.save(actions,     f"actions_{idx}.pt")
    torch.save(imgs_before, f"imgs_before_{idx}.pt")
    torch.save(imgs_after,  f"imgs_{idx}.pt")
    print(f"[Process {idx}] Saved {N} samples.")


if __name__ == "__main__":
    N_PER_PROCESS = 250   # 4 x 250 = 1000 total samples
    N_PROCESSES   = 4

    processes = []
    for i in range(N_PROCESSES):
        p = Process(target=collect, args=(i, N_PER_PROCESS))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("Done! All data collected.")
