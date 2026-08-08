import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from config import DIRECT_MODEL_PATH, DP_HIDDEN_SIZE, K_MAX, SEED
from direct_predictor import DirectPredictor
from toystateworld import generate_horizon_dataset

np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)


def training_direct(
    n_traj: int = 1000,
    len_traj: int = 20,
    burn: int = 5,
    k_max: int = K_MAX,
    hidden_size: int = DP_HIDDEN_SIZE,
    n_epochs: int = 10000,
    learning_rate: float = 1e-3,
) -> float:
    """
    Trains the direct predictor model on Henon map dataset.

    Args:
        n_traj (int): number of trajectories in a dataset
        len_traj (int): length of each trajectory
        burn (int): number of points discarded from the beggining
            of each trajectory. See generate_dataset for more info
        k_max (int): maximum number of steps in prediction

        hidden_size (int): number of neurons in a hidden layer
        n_epochs (int): number of epochs in training
        learning_rate (float): model's speed of learning

    Returns:
        float: predictor's final loss
    """

    X, y = generate_horizon_dataset(n_traj, len_traj, burn, k_max)

    model = DirectPredictor(hidden_size=hidden_size)
    loss_fn = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        model.train()
        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch: {epoch + 1:04d} | Loss: {loss.item():.4g}")

    torch.save(model.state_dict(), DIRECT_MODEL_PATH)

    return loss.item()


if __name__ == "__main__":
    training_direct(
        n_traj=1000,
        len_traj=20,
        burn=5,
        k_max=25,
        hidden_size=DP_HIDDEN_SIZE,
        n_epochs=20000,
        learning_rate=1e-3,
    )
