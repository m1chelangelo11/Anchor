import torch
from torch import nn

from config import DP_HIDDEN_SIZE, K_MAX


class DirectPredictor(nn.Module):
    """
    Class representing a direct head predictor model.

    Attributes:
        hidden_layer1 (torch.nn.Linear): Linear hidden layer
        activation1 (torch.nn.Tanh): Tanh activation layer
        hidden_layer2 (torch.nn.Linear): Linear hidden layer
        activation2 (torch.nn.Tanh): Tanh activation layer
        output_layer (torch.nn.Linear): Linear output layer
    """

    def __init__(self, hidden_size: int = DP_HIDDEN_SIZE):
        """
        Initialize DirectPredictor object.

        Args:
            hidden_size (int): size of the hidden layers, default
            value is 64 because of experiments
        """

        super().__init__()

        self.hidden_layer1 = nn.Linear(K_MAX + 2, hidden_size)
        self.activation1 = nn.Tanh()

        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a forward pass through a neural network.

        Args:
            x (torch.Tensor): input tensor with data package

        Returns:
            torch.Tensor: output tensor with model predictions
        """

        x = self.hidden_layer1(x)
        x = self.activation1(x)

        x = self.hidden_layer2(x)
        x = self.activation2(x)

        x = self.output_layer(x)

        return x
