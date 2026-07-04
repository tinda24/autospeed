# S3 CORE MODULE
import torch
import torch.nn as nn
import torch.nn.functional as F
import os,sys
import numpy as np
from torch import log
import random
import math
import torch_dct

from dataclasses import dataclass


# option: for gen model, 3 stage style, mentioned in sec2.4.1
def freeze_except_action_head(policy: nn.Module, head_attr: str = "action_head"):
    assert hasattr(policy.model, head_attr), f"policy has no attribute `{head_attr}`"
    head = getattr(policy.model, head_attr)
    assert isinstance(head, nn.Module), f"policy.{head_attr} must be an nn.Module"

    for p in policy.model.parameters():
        p.requires_grad_(False)

    for p in head.parameters():
        p.requires_grad_(True)

    head.train()
    

# region compress, echo sec 2.3 Motion Speed Transform
def compress(actions, acc_ratio=1.0, action_overcollect_ratio=2, num_queries=32):
    """Resample an action sequence via DCT compression and IDCT reconstruction.

    Args:
        actions: Input action sequence, shape ``[B, N, D]`` where ``B`` is batch
            size, ``N`` is sequence length, and ``D`` is action dimension.
        acc_ratio: Acceleration ratio controlling resampled time spacing.
        action_overcollect_ratio: Oversampling rate of the action (relative to the observation sampling rate) in your collected datasets.
        num_queries: Output sequence length ``M``.

    Returns:
        Resampled action sequence of shape ``[B, M, D]``.

    Notes:
        The pipeline is:
        1. DCT: map the time-domain signal to frequency coefficients.
        2. Temporal resampling: place ``M`` query points under ``acc_ratio``.
        3. IDCT: reconstruct the signal at the new times with cosine basis functions.
    """
    B, N, D = actions.shape

    assert isinstance(actions, torch.Tensor) and actions.dim() == 3  and acc_ratio > 0

    M = num_queries

    t_new = (torch.arange(M, device=actions.device, dtype=torch.float32) *
             float(acc_ratio) * float(action_overcollect_ratio)).unsqueeze(1)

    k = torch.arange(N, device=actions.device, dtype=torch.float32).unsqueeze(0)
    basis = torch.cos(torch.pi * k * (t_new + 0.5) / float(N))

    basis[:, 0] *= 1.0 / math.sqrt(2.0)
    basis *= math.sqrt(2.0 / float(N))

    actions_t = actions.transpose(1, 2)  # [B, D, N]
    C_t = torch_dct.dct(actions_t, norm='ortho')  # [B, D, N] DCT coefficients
    C = C_t.transpose(1, 2)  # [B, N, D] restore original layout

    acc_action = torch.einsum('mn,bnd->bmd', basis, C)

    return acc_action


# region AutoSpeedConfig
@dataclass
class AutoSpeedConfig:
    num_queries: int = -1  # must be set externally before use

    speed_range: list = None
    speed_interval: float = 0.1
    action_overcollect_ratio: float = 1.0

    s1_till_steps: int = 300
    s2_till_steps: int = 5000

    step_loss_type: str = 'log_denominator'  # speed weight curve: linear_denominator, log_denominator, linear_numerator, log_numerator
    step_loss_ratio: float = 1.3  # max/min speed weight ratio for stage-2 optimization

    # Ratio Head
    input_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 3

    ratio_head_lr: float = 1e-4

    def __post_init__(self):
        if self.speed_range is None:
            self.speed_range = [1.0, 2.0]

# region modulator
class MotionModulator:
    def __init__(self, autospeedconfig,device = 'cuda'):
        super().__init__()

        self.config = autospeedconfig
        self.ratio_head = RatioHead(input_dim=autospeedconfig.input_dim, 
                                    hidden_dim=autospeedconfig.hidden_dim, 
                                    num_layers=autospeedconfig.num_layers, 
                                    speed_range=autospeedconfig.speed_range, 
                                    speed_interval=autospeedconfig.speed_interval,
                                    device='cuda')
        self.device = device
        self.num_queries = autospeedconfig.num_queries
        assert self.num_queries > 0, "num_queries must be greater than 0"

        print(f"num_queries {self.num_queries}")
        self.speed_range = autospeedconfig.speed_range
        self.speed_min, self.speed_max = self.speed_range[0], self.speed_range[1]
        self.speed_interval = autospeedconfig.speed_interval
        self.action_overcollect_ratio = autospeedconfig.action_overcollect_ratio

        self.s1_till_steps = autospeedconfig.s1_till_steps
        self.s2_till_steps = autospeedconfig.s2_till_steps

        self.step_loss_type = autospeedconfig.step_loss_type
        self.step_loss_ratio = autospeedconfig.step_loss_ratio

        self.speed_num_steps = int(round((self.speed_max - self.speed_min) / self.speed_interval)) + 1
        self.speed_num_steps = max(self.speed_num_steps, 1)
        self.a_candidates = torch.linspace(self.speed_min, self.speed_max, self.speed_num_steps, device=self.device)
        self.a_candidates_list = [round(float(v), 1) for v in torch.linspace(self.speed_min, self.speed_max, self.speed_num_steps).tolist()]
        print(f"a_candidates_list: {self.a_candidates_list}")

        self.ration_head_optimizer = self.init_ratio_head_optimizer()
        
    def ratio_head_update(self, latent_features, ratio_gt):
        """
        Input:
            latent_features : [B, L, D]
            ratio_gt : [B]
        Output:
            LOSS of the softmax Ratio Head
        """
        latent_features = latent_features.detach().clone()
        ratio_gt = ratio_gt.detach().clone()
        with torch.enable_grad():
            loss = self.ratio_head.loss(latent_features,ratio_gt)
        return loss
    
    def ratio_head_prediction(self, latent_features):
        """
        Input:
            latent_features : [B, L, D]
        Output:
            ratio: [B]
        """
        with torch.no_grad():  # no gradients for inference
            return self.ratio_head.forward(latent_features)
    
    def init_ratio_head_optimizer(self):
        optim_ratio_head = torch.optim.AdamW(
                    self.ratio_head.parameters(),
                    lr=self.config.ratio_head_lr,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=1e-4)
        return optim_ratio_head


    def criterion_and_refresh_ratiohead(self, prediction_error, loss, training_step, latent_features, eval_mode=False):
        """
        Build the stage-dependent training objective and optionally update ratio_head.

        1. Turn per-speed prediction_error into a selection criterion.
        2. Pick the corresponding action loss.
        3. Gate ratio-head optimization by training_step (disabled in stage 3).
        4. Update ratio_head (treated as an internal speed estimator).

        Args:
            prediction_error: Per-speed errors, shape ``[B, N]`` where ``N`` is the
                number of speed candidates.
        """
        assert training_step >= 0
        criterion = prediction_error

        criterion = prediction_error
        if training_step < self.s1_till_steps:  # s1
            action_loss = loss.mean()
            
            total_loss = action_loss

            return total_loss, 0.0
        
        elif training_step < self.s2_till_steps:   # stage 2: selective speed optimization

            criterion = criterion.detach().clone() # [B, num_gt]

            # step weight
            loss_step = torch.log(self.a_candidates+1) if 'log' in self.step_loss_type else self.a_candidates
            max_loss_step, min_loss_step = loss_step.max(), loss_step.min()
            if 'denominator' in self.step_loss_type:  # use curve as denominator; tune max/min weight gap
                w = (max_loss_step - self.step_loss_ratio * min_loss_step) / (self.step_loss_ratio - 1.0)
                loss_step = 1 / (w + loss_step)
            elif 'numerator' in self.step_loss_type:  # use curve as numerator; tune max/min weight gap
                w = (self.step_loss_ratio * max_loss_step - min_loss_step) / (self.step_loss_ratio - 1.0)
                loss_step = w - loss_step
            else:
                raise ValueError(f'Invalid step loss type: {self.step_loss_type}')

            # aggregate criterion across speed candidates
            whole_criterion =  criterion * loss_step # * other_coef  + loss_ratio_refer * criterion    # [B, Na]

            selected_loss_id = whole_criterion.argmin(dim=1)  # [B]

            # selected action loss
            loss = loss[torch.arange(loss.shape[0]), selected_loss_id] # [B]    
            a_selected = [self.a_candidates[selected_loss_id[i]] for i in range(loss.shape[0])] # [B]

            # ratio head loss
            a_selected = torch.stack(a_selected, dim=0) # list to tensor

            latent_features = latent_features.float()

            ratio_head_loss = self.ratio_head_update(latent_features, a_selected)

            # update ratio head locally
            if not eval_mode:
                self.ration_head_optimizer.zero_grad(set_to_none=True)
                ratio_head_loss.backward()
                self.ration_head_optimizer.step()

            ratio_head_loss_to_return = ratio_head_loss.detach().clone()

            return loss.mean(), ratio_head_loss_to_return


        
        else:                                         # s3
            action_loss = loss.mean()  
            total_loss = action_loss
            return total_loss, 0.0
    
    # non-generative mode: return the full speed set
    def generate_action_group_nongenerative(self, action_data, is_pad, training_step, s1_random_speed = False, s3_latent_features = None):
        """
        Location:
            data batch ---> action head supervision

        Input:
            actions: [Batch, ceil(chunk_size*speed_max*action_overcollect_ratio) , action_dim]
            random_speed: If True, return one random speed only; cannot be used with specific_speed.
            #specific_speed: [Batch]
            training_step: Controls how many compressed variants are returned.

        Output:
            action_group(compressed_gt_actions): [Batch, 1 or N, chunk_size, action_dim]

        Build action_group from a fixed speed or from the full speed candidate list.
        
        """   
        M = math.ceil(self.action_overcollect_ratio*self.num_queries*self.speed_max)
        assert action_data.dim() == 3, f"action_data should be [B,L,D], got {action_data.shape}"
        B, L, D = action_data.shape

        if is_pad is None:
            is_pad = torch.zeros((B, L), dtype=torch.bool, device=action_data.device)  # all valid
        else:  # shape check
            assert is_pad.shape == (B, L), f"is_pad should be [L], got {is_pad.shape}"
            assert is_pad.dtype == torch.bool, "is_pad must be bool"

        out = action_data.new_empty((B, M, D))

        for b in range(B):
            valid_mask = ~is_pad[b]  # [L], True = valid
            idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)  # [K]
            K = idx.numel()

            if K == 0:
                out[b].zero_()
                continue

            take_idx = idx[:min(K, M)]
            picked = action_data[b].index_select(dim=0, index=take_idx)  # [min(K,M), D]

            if K >= M:
                out[b] = picked
            else:
                last = action_data[b, idx[-1], :].unsqueeze(0)           # [1, D]
                pad = last.expand(M - K, D)                               # [M-K, D]
                out[b] = torch.cat([picked, pad], dim=0)                 # [M, D]

        actions = out

        # standard path after padding/trimming
        return self.generate_action_group(actions=actions,training_step=training_step,s1_random_speed=s1_random_speed, s3_latent_features=s3_latent_features)

    # generative mode: return the action group for different stages
    def generate_action_group(self, actions, training_step, s1_random_speed = False, s3_latent_features = None):
        """
        Location:
            data batch ---> action head supervision

        Input:
            actions: [Batch, ceil(chunk_size*speed_max*action_overcollect_ratio) , action_dim]
            random_speed: If True, return one random speed only; cannot be used with specific_speed.
            specific_speed: [Batch]

        Output:
            action_group(compressed_gt_actions): [Batch, 1 or N, chunk_size, action_dim]

        Build action_group from a fixed speed or from the full speed candidate list.
        
        """   
        actions = actions.float()          # BF16 -> FP32

        compressed_gt_actions = []

        if training_step < self.s1_till_steps and s1_random_speed:   # stage 1 random-speed variant
            a = random.choice(self.a_candidates_list)
            compressed_gt_actions.append(compress(actions=actions,
                                                acc_ratio=a,
                                                action_overcollect_ratio=self.action_overcollect_ratio,
                                                num_queries=self.num_queries))
            action_group = torch.stack(compressed_gt_actions, dim=1)  #[Batch, 1, chunk_size, action_dim] 
            return action_group
        
        elif training_step > self.s2_till_steps:                      # s3  
            assert s3_latent_features!=None
            s3_latent_features = s3_latent_features.float()
            with torch.no_grad():
                ratio_output = self.ratio_head(s3_latent_features) #[B]
            
            compressed_gt_action = [compress(actions=actions[j:j+1],
                                        acc_ratio=ratio_output[j],
                                        action_overcollect_ratio=self.action_overcollect_ratio,
                                        num_queries=self.num_queries)  for j in range(len(ratio_output))]  # per-batch speed; each sample may differ
            compressed_gt_action = torch.cat(compressed_gt_action, dim=0)
            compressed_gt_actions.append(compressed_gt_action)
            action_group = torch.stack(compressed_gt_actions, dim=1)  #[Batch, 1, chunk_size, action_dim] 
            return action_group
        else:  # stage 1 mean and stage 2 return the full speed set
            for a in self.a_candidates_list:
                compressed_gt_actions.append(compress(actions=actions,
                                                     acc_ratio=a,
                                                     action_overcollect_ratio=self.action_overcollect_ratio,
                                                     num_queries=self.num_queries))
            action_group = torch.stack(compressed_gt_actions, dim=1)  #[Batch, N, chunk_size, action_dim] 
            return action_group

    def ratio_head_save_ckpt(
        self,
        save_path: str,
        step: int = None,
        extra: dict | None = None,
        save_optimizer: bool = True,
     ):
        """
        Save ratio_head only (optionally with optimizer state).

        Args:
            save_path: e.g. "ckpts/ratio_head.pt" or "ckpts/ratio_head_step_1000.pt"
            step: Optional step number stored in checkpoint metadata.
            extra: Optional extra metadata stored in the checkpoint.
            save_optimizer: Whether to save ration_head_optimizer state.
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        ckpt = {
            "ratio_head_state_dict": self.ratio_head.state_dict(),
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "speed_range": self.config.speed_range,
                "speed_interval": self.config.speed_interval,
            },
            "step": step,
            "extra": extra or {},
        }

        if save_optimizer and hasattr(self, "ration_head_optimizer") and self.ration_head_optimizer is not None:
            ckpt["ratio_head_optim_state_dict"] = self.ration_head_optimizer.state_dict()

        torch.save(ckpt, save_path)
        return save_path

    def ratio_head_load_ckpt(
        self,
        ckpt_path: str,
        map_location: str | torch.device | None = None,
        strict: bool = True,
        load_optimizer: bool = False,
        assert_config_match: bool = True,
     ):
        """
        Load ratio_head (optionally with optimizer state).

        Args:
            ckpt_path: Path to the saved .pt checkpoint.
            map_location: Defaults to self.device when None.
            strict: Passed to load_state_dict.
            load_optimizer: Whether to load ration_head_optimizer state.
            assert_config_match: Whether to verify structural hyperparameters match.

        Returns:
            ckpt: Raw checkpoint dict (useful for reading step/extra).
        """
        if map_location is None:
            map_location = self.device

        ckpt = torch.load(ckpt_path, map_location=map_location)

        # 1) Optional config consistency check (avoid loading after changing num_classes/layers)
        if assert_config_match and "config" in ckpt:
            cfg = ckpt["config"]
            current = {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "speed_range": self.config.speed_range,
                "speed_interval": self.config.speed_interval,
            }
            # speed_range may be list/tuple; normalize before comparing
            cfg_sr = list(cfg.get("speed_range", []))
            cur_sr = list(current.get("speed_range", []))
            if (cfg.get("input_dim") != current["input_dim"] or
                cfg.get("hidden_dim") != current["hidden_dim"] or
                cfg.get("num_layers") != current["num_layers"] or
                cfg_sr != cur_sr or
                float(cfg.get("speed_interval")) != float(current["speed_interval"])):
                raise ValueError(
                    f"[ratio_head_load_ckpt] Config mismatch!\n"
                    f"  ckpt:    {cfg}\n"
                    f"  current: {current}\n"
                    f"Hint: instantiate MotionModulator with the same config, "
                    f"or set assert_config_match=False to force load (not recommended)."
                )

        # 2) Load model weights
        self.ratio_head.load_state_dict(ckpt["ratio_head_state_dict"], strict=strict)

        # 3) Ensure correct device (especially cpu -> cuda)
        self.ratio_head.to(self.device)

        # 4) Optional optimizer restore
        if load_optimizer and "ratio_head_optim_state_dict" in ckpt:
            # Optimizer must already be initialized in __init__ via init_ratio_head_optimizer
            if not hasattr(self, "ration_head_optimizer") or self.ration_head_optimizer is None:
                self.ration_head_optimizer = self.init_ratio_head_optimizer()

            self.ration_head_optimizer.load_state_dict(ckpt["ratio_head_optim_state_dict"])

            # Optimizer state tensors must be moved to device (PyTorch does not do this automatically)
            for state in self.ration_head_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)


# region RatioHead
class RatioHead(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim, 
                 num_layers,
                 speed_range,
                 speed_interval,
                 device = 'cuda',
                 ):
        super(RatioHead, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.speed_range = speed_range
        self.speed_step = speed_interval
        self.a_max, self.a_min = self.speed_range[1], self.speed_range[0]

        # Number of speed classes from a_min to a_max with speed_step spacing
        self.num_classes = int(round((self.a_max - self.a_min) / self.speed_step)) + 1
        print()
        print(f"ratio_head num_classes: {self.num_classes}")

        # Actual speed value for each class index
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

        total = sum(p.numel() for p in self.parameters())
        print(f"Ratio Head parameter count total={total:,}")

    
    def forward(self,
                action_features,
            ):
        if len(action_features.shape)==3:
            action_features = action_features.mean(dim=1)  # [B, D]
        else:
            assert len(action_features.shape)==2
        logits = self.decoder(action_features)  # [B, num_classes]

        # Softmax over speed classes
        probs = F.softmax(logits, dim=-1)  # [B, num_classes]

        # Predicted class index
        pred_class = torch.argmax(probs, dim=-1)  # [B]

        # Map class index to speed value
        pred_a = self.speed_values[pred_class]  # [B]

        return pred_a

    def loss(self, action_features, targets):
        """
        Cross-entropy loss for softmax speed classification.

        Args:
            action_features: Input features, shape ``[B, input_dim]`` or ``[B, T, input_dim]``.
            targets: Target speed values, shape ``[B]``.

        Returns:
            loss: Cross-entropy loss.
            metrics: Dict with auxiliary metrics such as loss_a, loss_b, etc.
        """
        # Forward logits
        if len(action_features.shape)==3:
            action_features = action_features.mean(dim=1)  # [B, D]
        else:
            assert len(action_features.shape)==2
        logits = self.decoder(action_features)  # [B, num_classes]

        # Convert target speed values to class labels (nearest speed bin)
        targets = targets.unsqueeze(-1)  # [B, 1]
        distances = torch.abs(self.speed_values.unsqueeze(0) - targets)  # [B, num_classes]
        target_classes = torch.argmin(distances, dim=-1)  # [B]

        # Cross-entropy
        loss_a = F.cross_entropy(logits, target_classes)

        # Total loss
        loss = loss_a

        return loss