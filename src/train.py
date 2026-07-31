from toystateworld import generate_dataset
from model import Predictor
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch



MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "predictor.pt"
np.random.seed(seed=42)
torch.manual_seed(seed=42)



def training(n_traj, hidden_size, n_epochs, learning_rate):
    X, y = generate_dataset(n_traj, 20, 5)

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

    torch.save(model.state_dict(), MODEL_PATH)

    return loss.item()



if __name__ == "__main__":



    # loss_history = []
    # configs = [(1000, 64, 3000), (1000, 64, 5000), (1000, 64, 10000), (1000, 64, 15000)]
    # n_repeats = 10
    # learning_rate = 1e-3
    # for config in configs:
    #     loss_pair = []
    #     n_traj, hidden_size, n_epochs = config
    #     for rep in range(n_repeats):
    #         loss = training(n_traj, hidden_size, n_epochs, learning_rate)
    #         loss_pair.append(loss)
    #     loss_history.append(loss_pair)

    # for config, losses in zip(configs, loss_history):
    #     mean = np.mean(losses)
    #     std = np.std(losses)
    #     print(f"Number of trajectories: {config[0]} | "
    #           f"Hidden layer size: {config[1]} | "
    #           f"Number of repeats: {n_repeats}| "
    #           f"Number of epochs: {config[2]} | "
    #           f"Mean loss: {mean} | "
    #           f"Loss std: {std} | ")



    training(n_traj=1000, hidden_size=64, n_epochs=10000, learning_rate=1e-3)