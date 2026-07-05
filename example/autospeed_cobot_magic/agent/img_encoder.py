import os
import sys

import torch
import torch.nn as nn


sys.path.append(os.getcwd())


class static_dino_encoder_offline(nn.Module):
    def __init__(self, dinov2_type="vitb", img_size=518):
        super().__init__()
        self.dinov2_type = dinov2_type

        if "vitb" in dinov2_type:
            model_name = "dinov2_vitb14_reg"
            self.dino_feature_dim = 768
        elif "vits" in dinov2_type:
            model_name = "dinov2_vits14_reg"
            self.dino_feature_dim = 384
        elif "vitl" in dinov2_type:
            model_name = "dinov2_vitl14_reg"
            self.dino_feature_dim = 1024
        elif "vitg" in dinov2_type:
            model_name = "dinov2_vitg14_reg"
            self.dino_feature_dim = 1536
        else:
            raise ValueError(f"Unsupported DINOv2 type: {dinov2_type}")

        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", model_name)
        self.dino_encoder.eval()
        for param in self.dino_encoder.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.dino_encoder.forward_features(x)
