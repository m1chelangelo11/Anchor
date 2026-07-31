from train import training
import numpy as np


if __name__ == "__main__":
    loss_history = []
    configs = [(1000, 64, 3000), (1000, 64, 5000), (1000, 64, 10000), (1000, 64, 15000)]
    n_repeats = 10
    learning_rate = 1e-3

    for config in configs:
        loss_pair = []
        n_traj, hidden_size, n_epochs = config

        for rep in range(n_repeats):
            loss = training(n_traj, hidden_size, n_epochs, learning_rate)
            loss_pair.append(loss)

        loss_history.append(loss_pair)

    for config, losses in zip(configs, loss_history):
        mean = np.mean(losses)
        std = np.std(losses)

        print(f"Number of trajectories: {config[0]} | "
            f"Hidden layer size: {config[1]} | "
            f"Number of repeats: {n_repeats}| "
            f"Number of epochs: {config[2]} | "
            f"Mean loss: {mean} | "
            f"Loss std: {std} | ")