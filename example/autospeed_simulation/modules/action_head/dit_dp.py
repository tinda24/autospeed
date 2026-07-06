
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH, "repos"))

from modules.action_head.dit_flow import DiT1dWithACICrossAttention
from modules.action_head.mlp import MLP


class DDIMPolicy(nn.Module):

    def __init__(
        self,
        nn_diffusion: nn.Module,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        noise_init_varying_multispeed: str = "same",
    ):
        super().__init__()
        self.model = nn.ModuleDict({"diffusion": nn_diffusion})
        self.num_train_timesteps = num_train_timesteps
        self.noise_init_varying_multispeed = noise_init_varying_multispeed

        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=False,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        self.x_max = x_max
        self.x_min = x_min

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _prepare_noise(
        self,
        x0_group: torch.Tensor,
        x1_group: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bt, num_gt, q, d = x0_group.shape

        if x1_group is None and self.noise_init_varying_multispeed == "random":
            return torch.randn_like(x0_group)

        if x1_group is None and self.noise_init_varying_multispeed == "same":
            base = torch.randn((bt, q, d), device=self.device, dtype=x0_group.dtype)
            return base.unsqueeze(1).repeat(1, num_gt, 1, 1)

        if x1_group is None:
            raise ValueError(
                f"Invalid noise_init_varying_multispeed: {self.noise_init_varying_multispeed}"
            )

        x1_group = x1_group.to(device=self.device, dtype=x0_group.dtype)
        if x1_group.dim() == 3:
            if x1_group.shape != (bt, q, d):
                raise ValueError(
                    f"x1_group shape {x1_group.shape} incompatible with {(bt, q, d)}"
                )
            x1_group = x1_group.unsqueeze(1).repeat(1, num_gt, 1, 1)

        if x1_group.shape != x0_group.shape:
            raise ValueError(
                f"x1_group shape {x1_group.shape} must match x0_group {x0_group.shape}"
            )
        return x1_group

    def loss(
        self,
        x0_group: torch.Tensor,                      
        action_cond: torch.Tensor,
        criterion_type: str = "noise",
        x1_group: Optional[torch.Tensor] = None,
    ):
        bt, num_gt, q, d = x0_group.shape
        x0_group = x0_group.to(device=self.device)
        action_cond = action_cond.to(device=self.device, dtype=x0_group.dtype)
        if action_cond.shape[0] != bt:
            raise ValueError(
                f"action_cond batch {action_cond.shape[0]} does not match x0_group BT {bt}"
            )

                            
        noise_group = self._prepare_noise(x0_group, x1_group)

                                           
        t_bt = torch.randint(
            low=0,
            high=self.num_train_timesteps,
            size=(bt,),
            device=self.device,
        ).long()
        t = t_bt.repeat_interleave(num_gt, dim=0)               

        x0 = x0_group.reshape(bt * num_gt, q, d)
        noise = noise_group.reshape(bt * num_gt, q, d)
        action_cond_expanded = action_cond.repeat_interleave(num_gt, dim=0)

        noisy_x = self.scheduler.add_noise(x0, noise, t)

        t_input = t.float() / float(self.num_train_timesteps)
        pred_noise = self.model["diffusion"](noisy_x, t_input, action_cond_expanded)

        noise_error = (pred_noise - noise) ** 2                     

        if criterion_type == "noise":
            criteria_error = noise_error
        elif criterion_type == "action":
            alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
            alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1)
            sqrt_alpha = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
            pred_x0 = (noisy_x - sqrt_one_minus * pred_noise) / (sqrt_alpha + 1e-8)
            criteria_error = (pred_x0 - x0) ** 2
        else:
            raise ValueError(f"Invalid criterion_type: {criterion_type}")

        loss = noise_error.view(bt, num_gt, q, d).mean(dim=[2, 3])
        criteria = criteria_error.view(bt, num_gt, q, d).mean(dim=[2, 3])

        return {
            "loss": loss,
            "criterion": criteria,
        }

    @torch.no_grad()
    def sample(
        self,
        prior: torch.Tensor,
        action_cond: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        sample_steps: int = 20,
        temperature: float = 1.0,
    ):
        b = prior.shape[0]
        action_cond = action_cond.to(device=self.device, dtype=prior.dtype)
        if action_cond.shape[0] != b:
            raise ValueError(
                f"action_cond batch {action_cond.shape[0]} does not match prior batch {b}"
            )

        if x1 is None:
            x = torch.randn_like(prior) * temperature
        else:
            if x1.shape != prior.shape:
                raise ValueError(f"x1 shape {x1.shape} must match prior shape {prior.shape}")
            x = x1.to(device=self.device, dtype=prior.dtype)

        self.scheduler.set_timesteps(sample_steps, device=self.device)

        for t_int in self.scheduler.timesteps:
            t_norm = t_int.float() / float(self.num_train_timesteps)
            t_batch = t_norm.expand(b)
            pred_noise = self.model["diffusion"](x, t_batch, action_cond)
            x = self.scheduler.step(pred_noise, t_int, x).prev_sample

        if self.x_max is not None and self.x_min is not None:
            x = x.clip(self.x_min, self.x_max)

        return x, {}


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
        num_diffusion_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        device="cuda",
        **unused_kwargs,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.sampling_steps = sampling_steps
        self.device = device
        self.optimize_target = optimize_target
        self.action_recon_coef = action_recon_coef
        self.criterion_type = criterion_type
        self.diffusion_dim = action_dim if optimize_target == "action" else hidden_dim

        nn_diffusion = DiT1dWithACICrossAttention(
            x_dim=self.diffusion_dim,
            x_seq_len=num_queries,
            emb_dim=hidden_dim,
            d_model=d_model,
            n_heads=num_heads,
            depth=num_layers,
            timestep_emb_type=timestep_emb_type,
            timestep_emb_params=timestep_emb_params,
        ).to(self.device)

        x_max = None
        x_min = None
        if optimize_target == "action":
            x_max = torch.full((num_queries, action_dim), 1.0, device=self.device)
            x_min = torch.full((num_queries, action_dim), -1.0, device=self.device)

        self.policy = DDIMPolicy(
            nn_diffusion=nn_diffusion,
            num_train_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            x_max=x_max,
            x_min=x_min,
            noise_init_varying_multispeed=noise_init_varying_multispeed,
        ).to(self.device)

        if optimize_target == "latent_action_linear":
            self.action_encoder = nn.Linear(action_dim, hidden_dim).to(self.device)
            self.action_decoder = nn.Linear(hidden_dim, action_dim).to(self.device)
        elif optimize_target == "latent_action_mlp":
            self.action_encoder = MLP(
                in_channels=action_dim,
                hidden_channels=[hidden_dim, hidden_dim],
            ).to(self.device)
            self.action_decoder = MLP(
                in_channels=hidden_dim,
                hidden_channels=[hidden_dim, action_dim],
            ).to(self.device)
        else:
            self.action_encoder = None
            self.action_decoder = None

    def _resolve_optimize_target(self, optimize_target: Optional[str]) -> str:
        if optimize_target is None:
            target = self.optimize_target
        else:
            target = optimize_target

        valid_targets = {"action", "latent_action_linear", "latent_action_mlp"}
        if target not in valid_targets:
            raise ValueError(f"Invalid optimize_target: {target}")
        if target != self.optimize_target:
            raise ValueError(
                f"Runtime optimize_target ({target}) differs from init optimize_target "
                f"({self.optimize_target}). Keep them consistent to avoid decoder mismatch."
            )
        return target

    def _convert_init_noise_to_target(
        self,
        init_noise: Optional[torch.Tensor],
        optimize_target: str,
    ) -> Optional[torch.Tensor]:
        if init_noise is None:
            return None

        model_dtype = next(self.policy.parameters()).dtype
        init_noise = init_noise.to(device=self.policy.device, dtype=model_dtype)
        target_dim = self.action_dim if optimize_target == "action" else self.hidden_dim

        if init_noise.shape[-1] == target_dim:
            return init_noise

        if (
            "latent_action" in optimize_target
            and init_noise.shape[-1] == self.action_dim
            and self.action_encoder is not None
        ):
            return self.action_encoder(init_noise)

        raise ValueError(
            f"init_noise last dim {init_noise.shape[-1]} is incompatible with "
            f"target dim {target_dim} under optimize_target={optimize_target}"
        )

    def forward(self, action_cond, actions=None, optimize_target=None, init_noise=None):
        optimize_target = self._resolve_optimize_target(optimize_target)
        model_dtype = next(self.policy.parameters()).dtype
        action_cond = action_cond.to(device=self.policy.device, dtype=model_dtype)

                   
        if actions is None:
            b = action_cond.shape[0]
            prior = torch.zeros((b, self.num_queries, self.diffusion_dim), device=self.policy.device)
            init_noise = self._convert_init_noise_to_target(init_noise, optimize_target)

            act, _ = self.policy.sample(
                prior=prior,
                action_cond=action_cond,
                x1=init_noise,
                sample_steps=self.sampling_steps,
            )
            if "latent_action" in optimize_target:
                act = self.action_decoder(act)
            return act

                  
        actions = actions.to(device=self.policy.device, dtype=model_dtype)
        init_noise = self._convert_init_noise_to_target(init_noise, optimize_target)
        if "latent_action" in optimize_target:
            latent_actions = self.action_encoder(actions)
        else:
            latent_actions = None

        metrics = self.policy.loss(
            x0_group=latent_actions if latent_actions is not None else actions,
            action_cond=action_cond,
            criterion_type=self.criterion_type,
            x1_group=init_noise,
        )

        if latent_actions is not None:
            reconstructed_actions = self.action_decoder(latent_actions)
            action_loss = F.mse_loss(reconstructed_actions, actions, reduction="none")
            action_loss = action_loss.mean(dim=[2, 3])
            metrics["loss"] = metrics["loss"] + self.action_recon_coef * action_loss
            metrics["criterion"] = metrics["criterion"] + self.action_recon_coef * action_loss

        return metrics
                                                 
DitHead = ActionHead
