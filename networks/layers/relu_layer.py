import torch
from torch import nn
import numpy as np


class ReLULayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False):
        super().__init__()
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features), 
                                             np.sqrt(6 / self.in_features))
        
    def forward(self, input):
        return self.activation(self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.linear(input)
        return self.activation(intermediate), intermediate