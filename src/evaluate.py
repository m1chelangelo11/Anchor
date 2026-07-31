from model import Predictor
from toystateworld import rollout, compute_drifts, is_trajectory_stable
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os



np.random.seed(seed=42)



def model_rollout(model, z0, K):
    traj = [z0]
    z0 = torch.tensor(z0, dtype=torch.float32)
    z0 = z0.unsqueeze(0)
    z = z0
    for _ in range(K):
        z = model(z)
        z_new = tuple(z[0].tolist())
        traj.append(z_new)

    return traj



def average_drift(model, n_samples, K, burn_in):
    drifts = []
    unstable_traj = 0
    unstable_true_traj = 0
    unstable_model_traj = 0

    for i in range(0, n_samples):
        z0_raw = np.random.uniform(-1, 1, 2)
        warm_traj = rollout(z0_raw, burn_in)
        if not is_trajectory_stable(warm_traj):
            unstable_traj += 1
            continue
        
        z0_warm = warm_traj[-1]
        true_traj = rollout(z0_warm, K)
        if not is_trajectory_stable(true_traj):
            unstable_true_traj += 1
            continue

        model_traj = model_rollout(model, z0_warm, K)
        if not is_trajectory_stable(model_traj):
            unstable_model_traj += 1
            continue
        
        drift = compute_drifts(true_traj, model_traj)
        drifts.append(drift)

    return np.mean(drifts, axis=0), unstable_traj, unstable_true_traj, unstable_model_traj
    


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "predictor.pt"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

weights = torch.load(MODEL_PATH, weights_only=True)
model = Predictor(hidden_size=64)
model.load_state_dict(weights)
model.eval()

n_samples = 100
K = 50
burn_in = 5

with torch.inference_mode():
    means, n_unstable, n_unstable_true, n_unstable_model = average_drift(model, n_samples, K, burn_in)

print(f"Unstable warmup trajectories: {n_unstable}/100")
print(f"Unstable true trajectories: {n_unstable_true}/100")
print(f"Unstable model rollouts: {n_unstable_model}/100")

np.save(RESULTS_DIR / "baseline_drift_means.npy", means)

plt.semilogy(list(range(1, 51)), means[1:])
plt.xlabel("Rollout step k")
plt.ylabel("Drift ||true - predicted||")
plt.title(f"Average drift, baseline (n={n_samples}, K={K})")
plt.savefig(RESULTS_DIR / "baseline_drift_plot.png", dpi=150)
plt.grid()
plt.show()


