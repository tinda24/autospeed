import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])          
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])            

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def reparametrize_n(mu, std, n):
    eps = Variable(std.data.new(n, *std.size()).normal_())
    return mu.unsqueeze(0) + std.unsqueeze(0) * eps


class DETRVAE(nn.Module):

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim: int,
        action_dim: int,
        num_queries: int,
        camera_names: list,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

                                  
        self.latent_dim = latent_dim
        self.cls_embed = nn.Embedding(1, hidden_dim)                             
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )                      

                                  
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)                          

    def forward(self, qpos, image, env_state=None, actions=None, is_pad=None):
        is_training = actions is not None                
        bs, _ = qpos.shape

                                                
        if is_training:
                                               
                                                                                   
            action_embed = self.encoder_action_proj(actions)                         
            qpos_embed = self.encoder_joint_proj(qpos)                    
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)                       
            cls_embed = self.cls_embed.weight                   
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )                       
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )                           
            encoder_input = encoder_input.permute(1, 0, 2)                           
                                   
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )                        
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)               
                                       
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)                          
                         
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]                        
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
                                   
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

                                                                     
        if self.backbones is not None:
                                                                
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])                                   
                features = features[0]                               
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                                     
            proprio_input = self.input_proj_robot_state(qpos)
                                                        
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            qpos_emb = self.input_proj_robot_state(qpos)
            env_state_emb = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos_emb, env_state_emb], axis=1)                  
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
                                                                     
        features = hs.mean(dim=1)                   
        return a_hat, is_pad_hat, [mu, logvar], features

    def encode_images(self, qpos, image):
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[0](image[:, cam_id])
            features = features[0]
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        proprio_input = self.input_proj_robot_state(qpos)
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        return src, pos, proprio_input

    def forward_with_cached_features(self, qpos, cached_features, actions=None, is_pad=None):
        src, pos, proprio_input = cached_features
        is_training = actions is not None
        bs, _ = qpos.shape

        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)
            cls_embed = self.cls_embed.weight
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(1, 0, 2)
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        hs = self.transformer(
            src, None, self.query_embed.weight, pos,
            latent_input, proprio_input, self.additional_pos_embed.weight,
        )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        features = hs.mean(dim=1)
        return a_hat, is_pad_hat, [mu, logvar], features

    def get_samples(self, qpos, image, env_state=None, num_samples=10, actions=None, is_pad=None):
        bs, _ = qpos.shape

                           
        mu = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
        logvar = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
        latent_sample = reparametrize_n(mu, logvar.div(2).exp(), num_samples)
        latent_sample = latent_sample.reshape(num_samples * bs, self.latent_dim)
        latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
                                                                
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                                     
            proprio_input = self.input_proj_robot_state(qpos)
                                                        
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            src = torch.repeat_interleave(src, num_samples, dim=0)
            proprio_input = torch.repeat_interleave(proprio_input, num_samples, dim=0)
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            qpos_emb = self.input_proj_robot_state(qpos)
            env_state_emb = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos_emb, env_state_emb], axis=1)
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]

        a_hat = self.action_head(hs)
        a_hat = a_hat.reshape(num_samples, bs, -1, a_hat.shape[-1])
        features = hs.mean(dim=1)                                
        features = features.reshape(num_samples, bs, -1)                                 
        is_pad_hat = None
        return a_hat, is_pad_hat, [mu, logvar], features


def build_encoder(hidden_dim, enc_layers, nheads, dim_feedforward, dropout, pre_norm):
    activation = "relu"
    encoder_layer = TransformerEncoderLayer(
        hidden_dim, nheads, dim_feedforward, dropout, activation, pre_norm
    )
    encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
    encoder = TransformerEncoder(encoder_layer, enc_layers, encoder_norm)
    return encoder


def build_act_model(
    state_dim: int,
    action_dim: int,
    num_queries: int,
    camera_names: list,
    hidden_dim: int = 512,
    enc_layers: int = 4,
    dec_layers: int = 1,
    nheads: int = 8,
    dim_feedforward: int = 3200,
    dropout: float = 0.1,
    backbone_name: str = "resnet18",
    position_embedding: str = "sine",
    lr_backbone: float = 1e-5,
    pre_norm: bool = False,
    latent_dim: int = 32,
):
                    
    backbones = []
    backbone = build_backbone(
        hidden_dim=hidden_dim,
        backbone=backbone_name,
        position_embedding=position_embedding,
        lr_backbone=lr_backbone,
        masks=False,
        dilation=False,
    )
    backbones.append(backbone)

                       
    transformer = build_transformer(
        hidden_dim=hidden_dim,
        dropout=dropout,
        nheads=nheads,
        dim_feedforward=dim_feedforward,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        pre_norm=pre_norm,
    )

                       
    encoder = build_encoder(
        hidden_dim=hidden_dim,
        enc_layers=enc_layers,
        nheads=nheads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pre_norm=pre_norm,
    )

                              
    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        action_dim=action_dim,
        num_queries=num_queries,
        camera_names=camera_names,
        latent_dim=latent_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ACT model - number of parameters: {n_parameters / 1e6:.2f}M")

    return model
