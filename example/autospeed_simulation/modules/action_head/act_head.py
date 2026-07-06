import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from modules.act.detr_vae import build_act_model


class ACTActionHead(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        num_queries,
        camera_names,

        hidden_dim=512,
        enc_layers=4,
        dec_layers=1,
        nheads=8,
        dim_feedforward=3200,
        dropout=0.1,

        backbone='resnet18',
        position_embedding='sine',
        lr_backbone=1e-5,

        latent_dim=32,
        kl_weight=10.0,

        device='cuda',
        **unused_kwargs,
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight

        self.model = build_act_model(
            state_dim=state_dim,
            action_dim=action_dim,
            num_queries=num_queries,
            camera_names=camera_names,
            hidden_dim=hidden_dim,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            backbone_name=backbone,
            position_embedding=position_embedding,
            lr_backbone=lr_backbone,
            latent_dim=latent_dim,
        )

        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
        self.normalize = T.Normalize(mean=MEAN, std=STD)

    def normalize_images(self, images):
        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)
        images = self.normalize(images)
        images = images.view(B, V, C, H, W)
        return images

    def forward(self, action_cond, actions=None, **unused_kwargs):
        qpos = action_cond['qpos'].to(self.device)
        images = action_cond['images'].to(self.device)
        if images.max() > 1.0:
            images = images / 255.0
        images = self.normalize_images(images)

        if actions is None:
            a_hat, _, _, features = self.model(
                qpos=qpos, image=images, env_state=None,
                actions=None, is_pad=None,
            )
            return a_hat, features                                       

        B, num_gt, chunk_size, D = actions.shape
        cached_features = self.model.encode_images(qpos, images)

        losses = []
        criteria = []

        for i in range(num_gt):
            actions_gt = actions[:, i]                      
            is_pad = torch.zeros((B, chunk_size), dtype=torch.bool, device=self.device)

            a_hat, is_pad_hat, (mu, logvar), features = self.model.forward_with_cached_features(
                qpos=qpos, cached_features=cached_features,
                actions=actions_gt, is_pad=is_pad,
            )

            l1_loss = F.l1_loss(a_hat, actions_gt, reduction='none')
            l1_loss = l1_loss.mean(dim=[1, 2])       

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            total_loss = l1_loss + self.kl_weight * kl_loss

            losses.append(total_loss)
            criteria.append(l1_loss)

        _, _, _, prior_features = self.model.forward_with_cached_features(
            qpos=qpos, cached_features=cached_features,
            actions=None, is_pad=None,
        )
        return {
            'loss': torch.stack(losses, dim=1),                    
            'criterion': torch.stack(criteria, dim=1),              
            'features': prior_features,
        }

    def get_features(self, action_cond):
        qpos = action_cond['qpos'].to(self.device)
        images = action_cond['images'].to(self.device)
        if images.max() > 1.0:
            images = images / 255.0
        images = self.normalize_images(images)

        with torch.no_grad():
            _, _, _, features = self.model(
                qpos=qpos, image=images, env_state=None,
                actions=None, is_pad=None,
            )
        return features                   
