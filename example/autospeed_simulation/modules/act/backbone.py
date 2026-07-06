from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
                                        
                                   
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
                                                              
        try:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights="IMAGENET1K_V1",
                norm_layer=FrozenBatchNorm2d,
            )
        except TypeError:
                                                     
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True,
                norm_layer=FrozenBatchNorm2d,
            )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor):
        xs = self[0](tensor)
        out: List = []
        pos = []
        for name, x in xs.items():
            out.append(x)
                               
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(
    hidden_dim: int = 512,
    backbone: str = "resnet18",
    position_embedding: str = "sine",
    lr_backbone: float = 1e-5,
    masks: bool = False,
    dilation: bool = False,
):
    position_emb = build_position_encoding(hidden_dim, position_embedding)
    train_backbone = lr_backbone > 0
    return_interm_layers = masks
    backbone_model = Backbone(backbone, train_backbone, return_interm_layers, dilation)
    model = Joiner(backbone_model, position_emb)
    return model
