import copy
import os,sys

import numpy as np
import torch
import torch.nn as nn
from torch import log
from torch.jit import Final
import torch.nn.functional as F

class RatioHead(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim, 
                 num_layers,
                 new_loss_args,
                 device = 'cuda',
                 ):
        super(RatioHead, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)],
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        ).to(device)
        
        self.speed_range = new_loss_args.get('speed_range')
        self.a_max, self.a_min = self.speed_range[1], self.speed_range[0]
    
    def forward(self,
                action_features,
            ):
        raw_pred = self.decoder(action_features.squeeze(1))
        
        pred_a = self.a_min + (self.a_max - self.a_min) * torch.sigmoid(raw_pred[:, 0])
     
        return pred_a

    def loss(self, action_features, targets):
        pred_a = self.forward(action_features)  # [B]

        loss_a = F.mse_loss(pred_a, targets)

        loss = loss_a

        return loss