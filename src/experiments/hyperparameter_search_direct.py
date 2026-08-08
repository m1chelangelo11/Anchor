import numpy as np

from train_direct import training_direct

if __name__ == "__main__":
    loss_history = []
    configs = [
        (1000, 128, 20000),
        (1000, 256, 20000),
        #(1000, 64, 20000),
        #(1000, 64, 30000),
    ]
    n_repeats = 10
    learning_rate = 1e-3

    for config in configs:
        loss_pair = []
        n_traj, hidden_size, n_epochs = config

        for rep in range(n_repeats):
            loss = training_direct(
                n_traj=n_traj,
                hidden_size=hidden_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
            )
            loss_pair.append(loss)

        loss_history.append(loss_pair)

    for config, losses in zip(configs, loss_history):
        mean = np.mean(losses)
        std = np.std(losses)

        print(
            f"Number of trajectories: {config[0]} | "
            f"Hidden layer size: {config[1]} | "
            f"Number of repeats: {n_repeats}| "
            f"Number of epochs: {config[2]} | "
            f"Mean loss: {mean} | "
            f"Loss std: {std} | "
        )
