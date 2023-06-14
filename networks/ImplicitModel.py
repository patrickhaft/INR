import torch 
import torch.nn as nn


class ImplicitModel(nn.Module):
    def __init__(self):
        super(ImplicitModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2))

    def forward(self, x):
        x = x.to(self.device)
        x = self.model(x)
        return x