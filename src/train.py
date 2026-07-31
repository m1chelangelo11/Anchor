from toystateworld import generate_dataset
from model import Predictor
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import torch
from config import MODEL_PATH, SEED, HIDDEN_SIZE



np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)



def training(n_traj: int = 1000, len_traj: int = 20, burn: int = 5, 
             hidden_size: int = HIDDEN_SIZE, n_epochs: int = 10000, learning_rate: float = 1e-3) -> float:
    """
    Trains the one-step predictor model on Henon map dataset.

    Args:
        n_traj (int): number of trajectories in a dataset
        len_traj (int): length of each trajectory
        burn (int): number of points discarded from the beggining
            of each trajectory. See generate_dataset for more info

        hidden_size (int): number of neurons in a hidden layer
        n_epochs (int): number of epochs in training
        learning_rate (float): model's speed of learning
    
    Returns:
        float: predictor's final loss
    """

    X, y = generate_dataset(n_traj, len_traj, burn)

    model = Predictor(hidden_size=hidden_size)
    loss_fn = nn.MSELoss()
    optimizer = Adam(params=model.parameters(),lr=learning_rate)

    for epoch in range(n_epochs):
        model.train()
        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch: {epoch+1:04d} | Loss: {loss.item():.4g}")

    torch.save(model.state_dict(), MODEL_PATH)

    return loss.item()



if __name__ == "__main__":
    training(n_traj=1000, len_traj=20, burn=5, 
             hidden_size=64, n_epochs=10000, learning_rate=1e-3)