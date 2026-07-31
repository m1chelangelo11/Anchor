import torch
import torch.nn as nn



class Predictor(nn.Module):
    def __init__(self, hidden_size=2):
        super().__init__()

        self.hidden_layer = nn.Linear(2, hidden_size)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x

    

if __name__ == "__main__":
    model = Predictor()
    x = torch.randn(5, 2)
    out = model(x)
    print(out.shape)