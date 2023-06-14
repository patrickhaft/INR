import torch 
import numpy as np

from torch import nn

from .layers.trainable_act_layer import TrainableActLayer

class Trainable(nn.Module):

    def __init__(self,in_features, hidden_features, hidden_layers, out_features, vmin, vmax, num_weights, init="sine", init_scale=1.0, num_channels=1):
        super().__init__()
        
        self.net = []
        self.net.append(TrainableActLayer(in_features, hidden_features, 
                                          vmin, vmax, num_weights, init, init_scale, 
                                          num_channels, True))
        
        for i in range(hidden_layers):
            self.net.append(TrainableActLayer(hidden_features, hidden_features, 
                                          vmin, vmax, num_weights, init, init_scale, 
                                          num_channels, True))
        
        # self.net.append(TrainableActLayer(hidden_features, out_features, 
        #                                   vmin, vmax, num_weights, init, init_scale, 
        #                                   num_channels, True))
        
        final_linear = nn.Linear(hidden_features, out_features)
            
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / 30., 
                                            np.sqrt(6 / hidden_features) / 30.)
                
            self.net.append(final_linear)
        

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords
    
    def draw(self):
        layer_draw = []
        for i in range(self.net.__len__() - 1):
            layer_draw.append(self.net[i].draw())
        return layer_draw