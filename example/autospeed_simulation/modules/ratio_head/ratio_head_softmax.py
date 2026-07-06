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

        self.speed_range = new_loss_args.get('speed_range')
        self.speed_step = new_loss_args.get('speed_step')
        self.a_max, self.a_min = self.speed_range[1], self.speed_range[0]

        self.num_classes = int(round((self.a_max - self.a_min) / self.speed_step)) + 1
        print()
        print(f"ratio_head num_classes: {self.num_classes}")

        self.speed_values = torch.linspace(self.a_min, self.a_max, self.num_classes).to(device)
        print(f"ratio_head speed_values: {self.speed_values}")
        print()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)],
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, self.num_classes),
        ).to(device)

    def forward(self,
                action_features,
            ):
        logits = self.decoder(action_features.squeeze(1))                    

        probs = F.softmax(logits, dim=-1)                    

        pred_class = torch.argmax(probs, dim=-1)       

        pred_a = self.speed_values[pred_class]       

        return pred_a

    def loss(self, action_features, targets):
        logits = self.decoder(action_features.squeeze(1))                    

                       
        targets = targets.unsqueeze(-1)          
        distances = torch.abs(self.speed_values.unsqueeze(0) - targets)                    
        target_classes = torch.argmin(distances, dim=-1)       

        loss_a = F.cross_entropy(logits, target_classes)

        loss = loss_a

        return loss
