import torch
import torch.nn as nn
import os,sys
sys.path.append(os.getcwd())
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SKPCA
import einops
from torchvision import transforms as T
import torchvision.transforms as transforms

class static_dino_encoder_offline(nn.Module):
    def __init__(self, dinov2_type="vitb",img_size=518):
        super().__init__()
        self.dinov2_type = dinov2_type
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weight_path = os.path.join(project_root, "weights", "dinov2_weight", f"dinov2_{dinov2_type}14_reg4_pretrain.pth")
        ckpt_path = weight_path

        dinov2_path = os.path.join(project_root, "weights", "dinov2")
        if dinov2_path not in sys.path:
            sys.path.insert(0, dinov2_path)

        from dinov2.models import vision_transformer as vits

        if 'vitb' in weight_path:
            self.dino_encoder = vits.vit_base(
                patch_size=14,
                num_register_tokens=4,
                interpolate_antialias=True,
                interpolate_offset=0.0,
                img_size=img_size,
                init_values=1.0,
                ffn_layer="mlp",
                block_chunks=0,
            )
            self.dino_feature_dim = 768
        elif 'vits' in weight_path:
            self.dino_encoder = vits.vit_small(
                patch_size=14,
                num_register_tokens=4,
                interpolate_antialias=True,
                interpolate_offset=0.0,
                img_size=img_size,
                init_values=1.0,
                ffn_layer="mlp",
                block_chunks=0,
            )
        elif 'vitl' in weight_path:
            self.dino_encoder = vits.vit_large(
                patch_size=14,
                num_register_tokens=4,
                interpolate_antialias=True,
                interpolate_offset=0.0,
                img_size=img_size,
                init_values=1.0,
                ffn_layer="mlp",
                block_chunks=0,
            )
            self.dino_feature_dim = 384

        if os.path.exists(ckpt_path):
            print(f"Loading local DINOv2 weights: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.dino_encoder.load_state_dict(state_dict, strict=True)
        else:
            raise FileNotFoundError(f"Weight file does not exist: {ckpt_path}")

    def forward(self, x):
        return self.dino_encoder.forward_features(x)

class Dinov2ObservationEncoder(nn.Module):
    def __init__(self,
                dinov2_type,
                frozen,
                norm,
                device,
                output_dim,
                img_size,
                ):
        super().__init__()
        self.device = device

        self.dinov2 = static_dino_encoder_offline(dinov2_type=dinov2_type).to(device)

        self.dinov2_type = dinov2_type
        self.norm = norm
        self.separate_encoders = separate_encoders
        self.pixel_keys = pixel_keys
        self.frozen = frozen
        self.output_dim = output_dim
        self.img_size = img_size
        if self.norm:
            MEAN = torch.tensor([0.485, 0.456, 0.406])
            STD = torch.tensor([0.229, 0.224, 0.225])
            self.normalize = T.Normalize(mean=MEAN, std=STD)

        if frozen:
            if not separate_encoders:
                for param in self.dinov2.parameters():
                    param.requires_grad = False
            else:
                for key,value in self.dinov2.items():
                    for param in value.parameters():
                        param.requires_grad = False

        if self.img_size == [480,640]:
            self.spatial_adapter = nn.Sequential(
                nn.Conv2d(768, 512, 3, padding=1),                  
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),                  
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, stride = 2, padding=1),
                nn.Flatten(1),
            )
            self.adapter_2 = nn.Sequential(
                nn.Linear(64*12*9, self.output_dim),                  
                nn.LayerNorm(self.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.10)
            )
        elif self.img_size == [128,128]:
            self.spatial_adapter = nn.Sequential(
                nn.Conv2d(768, 512, 3, padding=1),                  
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),                  
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                nn.Flatten(1),
            )
            self.adapter_2 = nn.Sequential(
                nn.Linear(128*5*5, self.output_dim),                  
                nn.LayerNorm(self.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.10)
            )
        elif self.img_size == [240,320]:
            self.spatial_adapter = nn.Sequential(
                nn.Conv2d(768, 512, 3, padding=1),                  
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),                  
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                nn.Flatten(1),
            )
            self.adapter_2 = nn.Sequential(
                nn.Linear(128*9*11, self.output_dim),                  
                nn.LayerNorm(self.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.10)
            )
        else:
            raise ValueError(f"img_size: {self.img_size} not supported")
        print(f'trainable dino parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')

    def forward(self, pixel, pixel_key=None,lang=None):

        BT,H,W = pixel.shape[0], pixel.shape[-2], pixel.shape[-1]
        if H == 128 and W == 128:
            pixel = pixel[:,:,1:-1,1:-1]
        elif H == 480 and W == 640:
            pixel = pixel[:,:,2:-2,5:-5]
        elif H == 240 and W == 320:
            pixel = pixel[:,:,1:-1,6:-6]

        if self.norm:
            pixel = self.normalize(pixel)

        feature_assume = self.dinov2(pixel) if not self.separate_encoders else self.dinov2[pixel_key](pixel)

        patch_feature = feature_assume['x_norm_patchtokens']               
        B = patch_feature.shape[0]
        H_patch = H // 14
        W_patch = W // 14
        patch_feature = patch_feature.permute(0, 2, 1).view(B, -1, H_patch, W_patch)

        patch_feature = self.spatial_adapter(patch_feature)              
        patch_feature = self.adapter_2(patch_feature)         
        
        return patch_feature
