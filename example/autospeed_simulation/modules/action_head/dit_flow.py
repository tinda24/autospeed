import sys
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH, "repos"))
sys.path.append(os.path.join(ROOT_PATH, "repos", "CleanDiffuser"))

from typing import Optional, Union, Tuple, List, Dict
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.action_head.mlp import MLP

class DiTBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        use_cross_attn: bool = False,
        adaLN_on_cross_attn: bool = False,
    ):
        super().__init__()
        self._adaLN_on_cross_attn = adaLN_on_cross_attn
        self._use_cross_attn = use_cross_attn

                        
        self.sa_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sa_attn = nn.MultiheadAttention(hidden_size, n_heads, attn_dropout, batch_first=True)

                         
        if use_cross_attn:
            self.ca_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=not adaLN_on_cross_attn, eps=1e-6
            )
            self.ca_attn = nn.MultiheadAttention(
                hidden_size, n_heads, attn_dropout, batch_first=True
            )
        else:
            self.ca_norm, self.ca_attn = None, None

                      
        self.ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(ffn_dropout),
        )

               
        n_coeff = 9 if adaLN_on_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * n_coeff)
        )

    def forward(
        self,
        x: torch.Tensor,
        vec_condition: torch.Tensor,
        seq_condition: Optional[torch.Tensor] = None,
        seq_condition_mask: Optional[torch.Tensor] = None,
    ):
        adaLN_coeff = self.adaLN_modulation(vec_condition.unsqueeze(-2))
        if self._adaLN_on_cross_attn:
            (
                shift_sa,
                scale_sa,
                gate_sa,
                shift_ca,
                scale_ca,
                gate_ca,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = adaLN_coeff.chunk(9, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = adaLN_coeff.chunk(
                6, dim=-1
            )

        h = self.sa_norm(x) * (1 + scale_sa) + shift_sa
        x = x + gate_sa * self.sa_attn(h, h, h)[0]

        if self._use_cross_attn:
            if self._adaLN_on_cross_attn:
                h = self.ca_norm(x) * (1 + scale_ca) + shift_ca
            else:
                h = self.ca_norm(x)
                gate_ca = 1.0


            x = (
                x
                + gate_ca
                * self.ca_attn(
                    h, seq_condition, seq_condition, key_padding_mask=seq_condition_mask
                )[0]
            )

        h = self.ffn_norm(x) * (1 + scale_ffn) + shift_ffn
        x = x + gate_ffn * self.mlp(h)
        return x

class FinalLayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int, head_type: str = "linear"):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_size, out_dim),
            )
            nn.init.constant_(self.head[-1].weight, 0)
            nn.init.constant_(self.head[-1].bias, 0)
        else:
            self.head = nn.Linear(hidden_size, out_dim)
            nn.init.constant_(self.head.weight, 0)
            nn.init.constant_(self.head.bias, 0)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = self.modulate(self.norm(x), shift, scale)
        return self.head(x)

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import UntrainablePositionalEmbedding, set_seed

class DiT1dWithACICrossAttention(BaseNNDiffusion):
    def __init__(
        self,
        x_dim: int,
        x_seq_len: int,
        emb_dim: int,
        d_model: int ,
        n_heads: int ,
        depth: int ,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        head_type: str = "mlp",
        use_trainable_pos_emb: bool = True,
        use_cross_attn: bool = True,
        adaLN_on_cross_attn: bool = False,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
                          
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        if use_cross_attn:
            self.seq_cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        else:
            self.seq_cond_proj = None

                              
        pos_emb = UntrainablePositionalEmbedding(d_model)(torch.arange(x_seq_len))[None]
        self.pos_emb = nn.Parameter(pos_emb, requires_grad=use_trainable_pos_emb)

                                 
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    d_model, n_heads, attn_dropout, ffn_dropout, use_cross_attn, adaLN_on_cross_attn
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer1d(d_model, x_dim, head_type)
        self.initialize_weights()

        self.seq_cond_proj = None
        self.vis_cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        self.lang_cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))


    def initialize_weights(self):
                                             
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action_cond: torch.Tensor,
    ):
        t_emb = self.t_proj(self.map_noise(t))
        x_emb = self.x_proj(x) + self.pos_emb
        cond_emb = t_emb

        if action_cond is not None and self.vis_cond_proj is not None:
            action_cond = self.vis_cond_proj(action_cond)


        for i, block in enumerate(self.blocks):
            seq_condition = action_cond
            seq_condition_mask = None
            x_emb = block(x_emb, cond_emb, seq_condition, seq_condition_mask)

        x_emb = self.final_layer(x_emb, cond_emb)

        return x_emb

from cleandiffuser.diffusion.basic import DiffusionModel
from cleandiffuser.utils import (
    TensorDict,
    at_least_ndim,
    concat_zeros,
    dict_apply,
    get_sampling_scheduler,
)


class ContinuousRectifiedFlow(DiffusionModel):
    def __init__(
        self,
        nn_diffusion: BaseNNDiffusion,
        nn_condition = None,
        fix_mask: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        classifier = None,
        ema_rate: float = 0.995,
        optimizer_params: Optional[dict] = None,
        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        noise_init_varying_multispeed: str = 'same',
        criterion_type: str = 'noise',
    ):
        super().__init__(
            nn_diffusion,
            nn_condition,
            fix_mask,
            loss_weight,
            classifier,
            ema_rate,
            optimizer_params,
        )

        assert classifier is None, "Rectified Flow does not support classifier-guidance."

        self.noise_init_varying_multispeed = noise_init_varying_multispeed
        self.criterion_type = criterion_type

        self.x_max = nn.Parameter(x_max, requires_grad=False) if x_max is not None else None
        self.x_min = nn.Parameter(x_min, requires_grad=False) if x_min is not None else None

    @property
    def supported_solvers(self):
        return ["euler"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def add_noise(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
        ):
        t = torch.rand((x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps

        xt = x0 + at_least_ndim(t, x0.dim()) * (eps - x0)
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask

        return xt, t, eps

                 
    def loss(
        self,
        x0_group: torch.Tensor,
        action_cond: torch.Tensor,
        optimize_target: str = 'action',                                                             
        x1_group: torch.Tensor = None,
        ):

        num_gt, BT, chunk_size, D = x0_group.shape[1], x0_group.shape[0], x0_group.shape[2], x0_group.shape[3]
        x0_group = x0_group.to(device=self.device)

        if x1_group is None and self.noise_init_varying_multispeed == 'random':
            x1_group = torch.randn_like(x0_group)
        elif x1_group is None and self.noise_init_varying_multispeed == 'same':
            x1_group = torch.randn_like(x0_group[:,0])
            x1_group = x1_group.unsqueeze(1).repeat(1, num_gt, 1, 1)
        else:
            x1_group = x1_group.to(device=self.device)
            assert x0_group[0].shape == x1_group.shape, "x0 and x1 must have the same shape"

        t_bt = torch.rand((BT,), device=self.device)
        t = t_bt.repeat_interleave(num_gt, dim=0)

        x0_gt = x0_group.reshape(BT*num_gt, chunk_size, D)                              
        x1 = x1_group.reshape(BT*num_gt, chunk_size, D)

        xt, _, _ = self.add_noise(x0_gt, t=t, eps=x1)

                          
        action_cond = action_cond.repeat_interleave(num_gt, dim=0)
        pred_velocity = self.model["diffusion"](xt, t, action_cond)
                                                
        gt_velocity = x0_gt - x1

                            
        fm_loss = (pred_velocity - gt_velocity) ** 2
        fm_loss = fm_loss * self.loss_weight * (1 - self.fix_mask)
        loss = fm_loss                              

        if self.criterion_type == 'noise':
            criteria = fm_loss
        elif self.criterion_type == 'action':
            t_expanded = at_least_ndim(t, xt.dim())

            pred_x0 = xt + t_expanded * pred_velocity
            action_error = (pred_x0 - x0_gt) ** 2
            criteria = action_error
        else:
            raise ValueError(f"Invalid criterion type: {self.criterion_type}")

        loss = loss.reshape(BT, num_gt, chunk_size, D)                               
        loss = loss.mean(dim=[2, 3])                

        criteria = criteria.reshape(BT, num_gt, chunk_size, D)
        criteria = criteria.mean(dim=[2, 3])                

                                              
        return {
            'loss': loss,
            'criterion': criteria,
        }

    def update_diffusion(
        self,
        x0: torch.Tensor,
        condition_cfg: Optional[torch.Tensor] = None,
        update_ema: bool = True,
        x1: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return super().update_diffusion(x0, condition_cfg, update_ema, x1=x1)

    def sample(
        self,
                                                         
        prior: torch.Tensor,
        action_cond: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
                                                        
        solver: str = "euler",
        sample_steps: int = 5,
        sampling_schedule: str = "linear",
        sampling_schedule_params: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,
                                                          
        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: float = 0.0,
        condition_cg: None = None,
        w_cg: float = 0.0,
                                                     
        diffusion_x_sampling_steps: int = 0,
                                               
        warm_start_reference: Optional[torch.Tensor] = None,
        warm_start_forward_level: float = 0.3,
                                                        
        requires_grad: bool = False,
        preserve_history: bool = False,
        **kwargs,
        ):
        assert solver in self.supported_solvers, f"Solver {solver} is not supported."
        assert w_cg == 0.0 and condition_cg is None, (
            "Rectified Flow does not support classifier-guidance."
        )

                                                                    
        n_samples = prior.shape[0]
        log = {"sample_history": []}

        model = self.model if not use_ema else self.model_ema

        sampling_schedule_params = sampling_schedule_params or {}

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and 0.0 < warm_start_forward_level < 1.0:
            warm_start_reference = warm_start_reference.to(self.device)
            t_c = torch.ones_like(prior) * warm_start_forward_level
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
        else:
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"

        xt = x1
        xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = (
                model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            )

        sampling_scheduler = get_sampling_scheduler(sampling_schedule, **sampling_schedule_params)
        t_schedule = sampling_scheduler(
            sample_steps, device=self.device, **sampling_schedule_params
        )

                                                                       
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):
            t = torch.full((n_samples,), t_schedule[i], dtype=torch.float32, device=self.device)

            delta_t = t_schedule[i] - t_schedule[i - 1]
                                                     

                      
            with torch.set_grad_enabled(requires_grad):
                                              
                if w_cfg == 1.0:
                    assert condition_cfg is None
                    vel = model["diffusion"](xt, t, action_cond)

                                          
                elif w_cfg == 0.0:
                    vel = model["diffusion"](xt, t, None, None)

                else:
                    condition = dict_apply(condition_vec_cfg, concat_zeros, dim=0)

                    vel_all = model["diffusion"](
                        einops.repeat(xt, "b ... -> (2 b) ..."), t.repeat(2), condition
                    )

                    vel, vel_uncond = torch.chunk(vel_all, 2, dim=0)
                    vel = w_cfg * vel + (1 - w_cfg) * vel_uncond

                             
            xt = xt + delta_t * vel

                                                                      
            xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

                                                             
        if self.clip_pred:
                                                                                                   
            xt = xt.clip(self.x_min, self.x_max)
                                                                                                   

        log["t_schedule"] = t_schedule

        return xt, log

class ActionHead(nn.Module):
    def __init__(
                self,
                action_dim,
                hidden_dim,

                num_queries,

                num_heads,
                num_layers,
                d_model,
                timestep_emb_type,
                timestep_emb_params,

                sampling_steps,
                noise_init_varying_multispeed,
                criterion_type,
                optimize_target,
                action_recon_coef,
                device='cuda',
                **unused_kwargs,
                ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        self.sampling_steps = sampling_steps
        self.device = device

        nn_diffusion = DiT1dWithACICrossAttention(
            x_dim=action_dim if optimize_target == 'action' else hidden_dim,
            x_seq_len=num_queries,
            emb_dim=hidden_dim,
            d_model=d_model,
            n_heads=num_heads,
            depth=num_layers,
            timestep_emb_type=timestep_emb_type,
            timestep_emb_params=timestep_emb_params,
        ).to(self.device)

        self.policy = ContinuousRectifiedFlow(
            nn_diffusion=nn_diffusion,
            nn_condition=None,
            noise_init_varying_multispeed=noise_init_varying_multispeed,
            x_max=torch.full((num_queries, self.action_dim), 1.0, device=self.device),
            x_min=torch.full((num_queries, self.action_dim), -1.0, device=self.device),
            criterion_type=criterion_type,
        ).to(self.device)

        self.optimize_target = optimize_target
        self.action_recon_coef = action_recon_coef

         
        if optimize_target == 'latent_action_linear':
            self.action_encoder = nn.Linear(action_dim, hidden_dim).to(self.device)
            self.action_decoder = nn.Linear(hidden_dim, action_dim).to(self.device)
        elif optimize_target == 'latent_action_mlp':
            self.action_encoder = MLP(in_channels=action_dim,
                                    hidden_channels=[hidden_dim, hidden_dim]).to(self.device)
            self.action_decoder = MLP(in_channels=hidden_dim,
                                    hidden_channels=[hidden_dim, action_dim]).to(self.device)
        else:
            self.action_encoder = None
            self.action_decoder = None

    def forward(self, action_cond, actions=None, optimize_target = 'action', init_noise=None):
                   
        if actions is None:
            B = action_cond.shape[0]
            if self.optimize_target == 'action':
                prior = torch.zeros((B, self.num_queries, self.action_dim), device=self.device)
            elif 'latent_action' in self.optimize_target:
                prior = torch.zeros((B, self.num_queries, self.hidden_dim), device=self.device)
            else:
                raise ValueError(f"Invalid optimize target: {self.optimize_target}")
            act, _ = self.policy.sample(
                prior=prior,
                action_cond=action_cond,
                x1=init_noise,
                solver="euler",
                sample_steps=self.sampling_steps,
                use_ema=False,
                w_cfg=1.0,
            )
            return act

        else:          
                                                     
                                            
            if 'latent_action' in optimize_target:
                latent_actions = self.action_encoder(actions)

            metrics =  self.policy.loss(
                x0_group=latent_actions if 'latent_action' in optimize_target else actions,
                action_cond=action_cond,
                optimize_target=optimize_target,
                x1_group=init_noise,
            )

            if 'latent_action' in optimize_target:                                            
                reconstructed_actions = self.action_decoder(latent_actions)
                action_loss = F.mse_loss(reconstructed_actions, actions, reduction='none')
                action_loss = action_loss.mean(dim=[2, 3])

                assert action_loss.shape == metrics['loss'].shape, f"action_loss and metrics['loss'] must have the same shape, but got {action_loss.shape} and {metrics['loss'].shape}"

                metrics['loss'] = metrics['loss'] + self.action_recon_coef * action_loss
                metrics['criterion'] = metrics['criterion'] + self.action_recon_coef * action_loss

            return metrics
