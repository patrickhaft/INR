import torch
from torch import nn
import numpy as np


class WireLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=20, s_0=10):
        super().__init__()
        self.omega_0 = omega_0
        self.s_0 = s_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        linear_intermditate = self.linear(input)
        omega_intermediate = self.omega_0 * linear_intermditate
        s_intermediate = - torch.abs(torch.square(self.s_0 * linear_intermditate))
        return torch.exp(omega_intermediate) * torch.exp(s_intermediate)
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        linear_intermditate = self.linear(input)
        omega_intermediate = torch.imag(self.omega_0 * linear_intermditate)
        s_intermediate = - torch.abs(torch.square(self.s_0 * linear_intermditate))
        return torch.exp(omega_intermediate) * torch.exp(s_intermediate), linear_intermditate