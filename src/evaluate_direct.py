import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DIRECT_MODEL_PATH, DP_HIDDEN_SIZE, K_MAX, RESULTS_DIR, SEED
from direct_predictor import DirectPredictor
from toystateworld import build_horizon_input, dist, is_trajectory_stable, rollout

np.random.seed(seed=SEED)


def average_diff(
    model: DirectPredictor,
    n_samples: int,
    burn_in: int,
    k_max: int = K_MAX,
) -> tuple[list[np.ndarray], int, int, int]:
    """
    Compute average difference between direct prediction and true
    state from the trajectory rollout from a given number of
    trajectories.

    Args:
        model (DirectPredictor): direct predictor object
        n_samples (int): number of trajectories
        burn_in (int): number of discarded point from trajectory
        k_max (int): maximum number of steps to predict to

    Returns:
        tuple[list[np.ndarray], int, int]: list of average differences
        for each k, number of unstable trajectories, number of
        unstable true trajectories.
    """

    diffs = []
    unstable_traj = 0
    unstable_true_traj = 0

    for k in range(1, k_max + 1):
        diffs_k = []
        for i in range(n_samples):
            z0_raw = np.random.uniform(-1, 1, 2)
            warm_traj = rollout(z0_raw, burn_in)
            if not is_trajectory_stable(warm_traj):
                unstable_traj += 1
                continue

            z0_warm = warm_traj[-1]
            true_traj = rollout(z0_warm, k)
            if not is_trajectory_stable(true_traj):
                unstable_true_traj += 1
                continue

            true_state = true_traj[-1]
            model_input = torch.tensor(
                build_horizon_input(z0_warm, k, k_max), dtype=torch.float32
            )
            model_input = model_input.unsqueeze(0)
            predicted_state = tuple(model(model_input)[0].tolist())
            diff = dist(true_state, predicted_state)
            diffs_k.append(diff)

        diffs.append(np.mean(diffs_k, axis=0))

    return (diffs, unstable_traj, unstable_true_traj)


if __name__ == "__main__":
    weights = torch.load(DIRECT_MODEL_PATH, weights_only=True)
    model = DirectPredictor(hidden_size=DP_HIDDEN_SIZE)
    model.load_state_dict(weights)
    model.eval()

    n_samples = 100
    k_max = K_MAX
    burn_in = 5

    with torch.inference_mode():
        means, n_unstable, n_unstable_true = average_diff(
            model, n_samples, burn_in, k_max
        )

    print(f"Unstable warmup trajectories: {n_unstable}/{k_max*n_samples}")
    print(f"Unstable true trajectories: {n_unstable_true}/{k_max*n_samples}")

    np.save(RESULTS_DIR / "direct_diff_means.npy", means)

    plt.semilogy(list(range(1, k_max+1)), means)
    plt.xlabel("Prediction step k")
    plt.ylabel("Difference ||true - predicted||")
    plt.title(f"Average differences, direct prediction (n={n_samples}, k_max={k_max})")
    plt.savefig(RESULTS_DIR / "direct_diff_plot.png", dpi=150)
    plt.grid()
    plt.show()
