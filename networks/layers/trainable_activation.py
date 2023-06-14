import torch
from torch import nn
import numpy as np

mps_device = torch.device("mps")


class ActivationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights, vmin, vmax):
        ctx.save_for_backward(input, weights)  # Save tensors for backward pass
        ctx.vmin = vmin
        ctx.vmax = vmax
        
        distance = vmax - vmin

        num_weights = weights.shape[0]

        distance_per_weight = distance / (num_weights - 1)

        distance_vmin = input - vmin

        num_distance = distance_vmin / distance_per_weight

        lower_weight_index = torch.clamp(num_distance, 0, num_weights - 1).to(torch.int32)
        upper_weight_index = torch.clamp(num_distance + 1, 0, num_weights - 1).to(torch.int32)

        output = weights[lower_weight_index] + (num_distance - lower_weight_index) * (weights[upper_weight_index] - weights[lower_weight_index])

        
        return output
    
    
    @staticmethod
    def draw( weights, vmin, vmax, scale=2):
        x = torch.linspace(scale*vmin, scale*vmax, 1001, dtype=weights.dtype).unsqueeze_(0)
        x = x.repeat(weights.shape[0], 1)
        mps_device = torch.device("mps")
        x = x.to(mps_device)
        f_x = ActivationFunction.apply(x, weights, vmin, vmax)
        return x, f_x
    


class TrainableActivation(nn.Module):
    def __init__(self, num_channels, vmin, vmax, num_weights, init="linear", init_scale=1.0) -> None:
        super(TrainableActivation, self).__init__()

        self.num_channels = num_channels
        self.vmin = vmin
        self.vmax = vmax
        self.num_weights = num_weights
        self.init = init
        self.init_scale = init_scale
        self.group = 1

        self.weight = nn.Parameter(torch.Tensor(self.num_channels, self.num_weights))
        # self.weight = torch.Tensor(self.num_channels, self.num_weights)
        self.reset_parameters()

    
    def reset_parameters(self):
        np_x = np.linspace(self.vmin, self.vmax, self.num_weights, dtype=np.float32)

        if self.init == "constant":
            np_w = np.ones_like(np_x) * self.init_scale
        elif self.init == "linear":
            np_w = np_x * self.init_scale
        elif self.init == "quadratic":
            np_w = np_x**2 * self.init_scale
        elif self.init == "abs":
            np_w = np.abs(np_x) * self.init_scale
        elif self.init == "sine":
            np_w = np.sin(np_x) * self.init_scale
        else:
            raise RuntimeError(f"Unsupported init type {self.init}!")

        self.weight.data = torch.from_numpy(np_w)
        # self.weight = self.weight.to(mps_device)

    def forward(self, x):

        distance = self.vmax - self.vmin

        distance_per_weight = distance / (self.num_weights - 1)

        distance_vmin = x - self.vmin

        num_distance = distance_vmin / distance_per_weight

        lower_weight_index = torch.clamp(num_distance, 0, self.num_weights - 1).to(torch.int32).to(mps_device)
        upper_weight_index = torch.clamp(num_distance + 1, 0, self.num_weights - 1).to(torch.int32).to(mps_device)

        x = self.weight[lower_weight_index] 
        x += (num_distance - lower_weight_index) * (self.weight[upper_weight_index] - self.weight[lower_weight_index])

        return x

    def draw(self, scale=2):
        x = torch.linspace(scale*self.vmin, scale*self.vmax, 1001, dtype=self.weight.dtype).unsqueeze_(0)
        x = x.repeat(self.weight.shape[0], 1)
        mps_device = torch.device("mps")
        x = x.to(mps_device)
        return x, self(x)