import torch
from torch import nn
import numpy as np
from . import trainable_activation 

class TrainableActLayer(nn.Module):

    def __init__(self, in_features, out_features, vmin, vmax, num_weights, init="sine", init_scale=1.0, num_channels=1, bias=True):
        super().__init__()
        self.omega_0 = 30
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # self.activation = nn.ReLU()
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6 / self.in_features), 
                                             np.sqrt(6 / self.in_features))


        
        self.trainable_activation = trainable_activation.TrainableActivation(num_channels=num_channels, vmin=vmin, vmax=vmax, num_weights=num_weights, init=init, init_scale=init_scale)

    def forward(self, input):
        input = self.linear(input)
        input = self.trainable_activation(input)

        return input

    def draw(self):
        return self.trainable_activation.draw()