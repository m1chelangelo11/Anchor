import matplotlib.pyplot as plt
import torch
import numpy as np



def step(z, a=1.4, b=0.3):
    x, y = z
    x_next = 1 - a * x ** 2 + y
    y_next = b * x

    return (x_next, y_next)



def rollout(z0, K):
    traj = [z0]
    z = z0
    for _ in range(K):
        z = step(z)
        traj.append(z)

    return traj



def dist(point_a, point_b):
    x_a, y_a = point_a
    x_b, y_b = point_b
    euc_dist = ((x_a - x_b) ** 2 + (y_a - y_b) ** 2) ** 0.5

    return euc_dist



def compute_drifts(traj_a, traj_b):
    drifts = []
    for point_a, point_b in zip(traj_a, traj_b):
        distance = dist(point_a, point_b)
        drifts.append(distance)

    return drifts



def generate_dataset(n_traj, len_traj, burn):
    inputs = []
    targets = []

    for i in range(0, n_traj):
        z0 = np.random.uniform(-1, 1, 2)
        traj = rollout(z0, len_traj)
        if any(abs(point[0]) > 10 or abs(point[1]) > 10 for point in traj):
            continue
        traj = traj[burn:]

        for input, target in zip(traj, traj[1:]):
            inputs.append(input)
            targets.append(target)

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)



if __name__ == "__main__":
    X, y = generate_dataset(n_traj=1000, len_traj=20, burn=5)
    print(X.shape, y.shape)
