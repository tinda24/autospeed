import math
import einops
import random
import os
import numpy as np
import torch
import torch_dct
from torch import nn
from torchvision import transforms as T
import utils.agent_utils as utils
from utils.mlp import MLP
from utils.log_print import log_print_params as log_print

from agent.img_encoder import static_dino_encoder_offline
from agent.backbone import GPT, GPTConfig

def compress(actions, acc_ratio=1.0, action_overcollect_ratio=2, num_queries=32):
    B, N, D = actions.shape

    assert isinstance(actions, torch.Tensor) and actions.dim() == 3  and acc_ratio > 0

    M = num_queries

    t_new = (torch.arange(M, device=actions.device, dtype=torch.float32) *
             float(acc_ratio) * float(action_overcollect_ratio)).unsqueeze(1)
    k = torch.arange(N, device=actions.device, dtype=torch.float32).unsqueeze(0)

    basis = torch.cos(torch.pi * k * (t_new + 0.5) / float(N))
    basis[:, 0] *= 1.0 / math.sqrt(2.0)
    basis *= math.sqrt(2.0 / float(N))
    actions_t = actions.transpose(1, 2)

    C_t = torch_dct.dct(actions_t, norm='ortho')
    C = C_t.transpose(1, 2)
    acc_action = torch.einsum('mn,bnd->bmd', basis, C)

    return acc_action

class Actor(nn.Module):
    def __init__(
        self,

        action_head,
        ratio_head,
        action_dim,
        proprio_dim,

        img_size,
        num_views,

        pixel_keys,
        proprio_key,
        lang_key,

        hidden_dim,
        loss_coef,
        num_queries,

        group_loss_window,
        s1_till_steps,
        s1_strategy,
        s2_till_steps,
        syn_optimize_ratio,                # not used now
        action_overcollect_ratio,
        optimize_target,

        step_loss_type,
        step_loss_ratio,
        constrain_speed_via_ratio_head,
        constrain_from_ratio_coef,
        constrain_coef_linear_up,
        constrain_other_coef_linear_down,
        constrain_start_step,

        new_loss_args,
        device = 'cuda',

        **unused_kwargs,
    ):
        super().__init__()

        self.device = device

        self.action_head = action_head.to(device)
        self.ratio_head = ratio_head.to(device)

        self.action_dim = action_dim
        self.proprio_dim = proprio_dim

        self.img_size = img_size
        self.num_views = num_views

        self.pixel_keys = pixel_keys
        self.proprio_key = proprio_key
        self.lang_key = lang_key
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.loss_coef = loss_coef

        self.group_loss_window = group_loss_window
        self.s1_till_steps = s1_till_steps
        self.s1_strategy = s1_strategy
        self.s2_till_steps = s2_till_steps
        self.constrain_coef_linear_up = constrain_coef_linear_up
        self.constrain_other_coef_linear_down = constrain_other_coef_linear_down
        self.constrain_start_step = constrain_start_step
        self.action_overcollect_ratio = action_overcollect_ratio
        self.optimize_target = optimize_target

        self.constrain_speed_via_ratio_head = constrain_speed_via_ratio_head
        self.constrain_from_ratio_coef = constrain_from_ratio_coef
        self.step_loss_type = step_loss_type
        self.step_loss_ratio = step_loss_ratio
        
        self.new_loss_args = new_loss_args
        speed_range = new_loss_args.get('speed_range')
        speed_step = new_loss_args.get('speed_step')
        self.speed_max, self.speed_min = speed_range[1], speed_range[0]
        self.speed_num_steps = int(round((self.speed_max - self.speed_min) / speed_step)) + 1
        self.speed_num_steps = max(self.speed_num_steps, 1)
        self.a_candidates = torch.linspace(self.speed_min, self.speed_max, self.speed_num_steps, device=self.device)
        self.a_candidates_list = [round(float(v), 1) for v in torch.linspace(self.speed_min, self.speed_max, self.speed_num_steps).tolist()]
        log_print(f"a_candidates_list: {self.a_candidates_list}")

        self.language_projector = MLP(384,hidden_channels=[self.hidden_dim, self.hidden_dim],).to(device)
        self.language_projector.apply(utils.weight_init)

        self.proprio_projector = MLP(self.proprio_dim,hidden_channels=[self.hidden_dim, self.hidden_dim],).to(device)
        self.proprio_projector.apply(utils.weight_init)

        self.img_encoder = static_dino_encoder_offline(dinov2_type="vitb").to(device)
        for p in self.img_encoder.parameters():
            p.requires_grad = False
        
        self.spatial_adapter = self.load_spatial_adapter().to(device)
        
        MEAN = torch.tensor([0.485, 0.456, 0.406], device=self.device)
        STD = torch.tensor([0.229, 0.224, 0.225], device=self.device)
        self.normalize = T.Normalize(mean=MEAN, std=STD)

        gptconfig = GPTConfig(
            block_size=65,
            input_dim=512,
            output_dim=512,
            n_layer=12,
            n_head=8,
            n_embd=512,
        )
        self.backbone = GPT(gptconfig)
        self.backbone.to(device)
        self._action_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim).to(device))

        self.all_time_actions = torch.zeros(
            [
                1000,
                1000 + self.num_queries,
                self.action_dim,
            ]
        ).to(self.device)
    
    def load_spatial_adapter(self):
        patch_size = 14
        effective_img_h = (self.img_size[0] // patch_size) * patch_size
        effective_img_w = (self.img_size[1] // patch_size) * patch_size
        patch_h = max(1, effective_img_h // patch_size)
        patch_w = max(1, effective_img_w // patch_size)

        adapter_out_h = (patch_h + 1) // 2
        adapter_out_w = (patch_w + 1) // 2

        adapter_flat_dim = 128 * adapter_out_h * adapter_out_w

        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(768, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.Flatten(1),
            nn.Linear(adapter_flat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10)
        )
        return self.spatial_adapter
    
    def crop_image(self, image):
        assert len(image.shape) == 4
        H, W = image.shape[-2:]
        det_H = H%14
        det_W = W%14
        high = det_H//2
        low = det_H - high
        left = det_W//2
        right = det_W - left

        if low == 0:
            low = -10000
        if right == 0:
            right = -10000
        return image[:, :, high:-low, left:-right]

    def stage3_freeze_backbone(self, train_step):
        log_print(f"stage3_freeze_backbone: train_step: {train_step}")
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.spatial_adapter.parameters():
            p.requires_grad = False
        for p in self.proprio_projector.parameters():
            p.requires_grad = False
        for p in self.language_projector.parameters():
            p.requires_grad = False


    def get_gt_action_group(self, actions, sampling_strategy='random', s3_specified_speed=None): #full or align
        assert actions.shape[1] == math.ceil(self.num_queries * self.action_overcollect_ratio * self.speed_max), f'actions.shape: {actions.shape}, self.num_queries: {self.num_queries}, self.action_overcollect_ratio: {self.action_overcollect_ratio}'
      
        compressed_gt_actions = []

        if sampling_strategy == 'random':
            a = random.choice(self.a_candidates_list)
            compressed_gt_actions.append(compress(actions=actions,
                                                 acc_ratio=a,
                                                 action_overcollect_ratio=self.action_overcollect_ratio,
                                                 num_queries=self.num_queries))

        elif sampling_strategy == 'full' or sampling_strategy == 'mean':
            for a in self.a_candidates_list:
                compressed_gt_actions.append(compress(actions=actions,
                                                     acc_ratio=a,
                                                     action_overcollect_ratio=self.action_overcollect_ratio,
                                                     num_queries=self.num_queries))
        elif sampling_strategy == 'align':
            assert s3_specified_speed != None
            compressed_gt_action = [compress(actions=actions[j:j+1],
                                        acc_ratio=s3_specified_speed[j],
                                        action_overcollect_ratio=self.action_overcollect_ratio,
                                        num_queries=self.num_queries)  for j in range(len(s3_specified_speed))]
            compressed_gt_action = torch.cat(compressed_gt_action, dim=0)
            compressed_gt_actions.append(compressed_gt_action)

        return compressed_gt_actions

    def optimize_strategy(self, loss, criterion, sampling_strategy, training_step, pred_speed = None, is_eval = False):
        if sampling_strategy != 'full':
            return loss.mean(), 0
        
        if sampling_strategy == 'full':              
            criterion = criterion.detach().clone()
            # step weight
            loss_step = torch.log(self.a_candidates+1) if 'log' in self.step_loss_type else self.a_candidates
            max_loss_step, min_loss_step = loss_step.max(), loss_step.min()
            if 'denominator' in self.step_loss_type:
                w = (max_loss_step - self.step_loss_ratio * min_loss_step) / (self.step_loss_ratio - 1.0)
                loss_step = 1 / (w + loss_step)
            elif 'numerator' in self.step_loss_type:
                w = (self.step_loss_ratio * max_loss_step - min_loss_step) / (self.step_loss_ratio - 1.0)
                loss_step = w - loss_step
            else:
                raise ValueError(f'Invalid step loss type: {self.step_loss_type}')

            loss_ratio_refer = 0.0
            other_coef = 1.0
            if pred_speed is not None and self.constrain_speed_via_ratio_head and training_step >= self.constrain_start_step:
                centers = pred_speed.unsqueeze(1)                   # [BT, 1]
                a_candidate = torch.tensor(self.a_candidates, device=self.device).unsqueeze(0)  # [1, Na]
                diff = torch.abs(a_candidate - centers)             # [BT, Na]
                loss_ratio_refer = diff
                linear_ratio = min(1.0, max(0.0, (training_step - self.constrain_start_step) / (self.s2_till_steps - self.constrain_start_step)))
                if self.constrain_coef_linear_up:
                    loss_ratio_refer = loss_ratio_refer * linear_ratio
                if self.constrain_coef_linear_up and self.constrain_other_coef_linear_down:
                    other_coef = 1.0 - linear_ratio

            whole_criterion = other_coef * criterion * loss_step + loss_ratio_refer * criterion    # [B*T , Na]

            whole_criterion = einops.rearrange(whole_criterion, '(b T) n -> b T n',T = self.group_loss_window if not is_eval else 1)
            whole_criterion = whole_criterion.mean(dim=1) # [B, Na]
            
            selected_loss_id = whole_criterion.argmin(dim=1) # [B]

            loss = einops.rearrange(loss, '(b T) n -> b T n',T = self.group_loss_window if not is_eval else 1)
            loss = loss.mean(dim=1) # [B, Na]
            loss = loss[torch.arange(loss.shape[0]), selected_loss_id] # [B]

            a_selected = [self.a_candidates[selected_loss_id[i]] for i in range(loss.shape[0])] # [B]

            return loss.mean(), a_selected
        
    # region update
    def update(self, expert_replay_iter, train_step):

        if train_step == self.s2_till_steps:
            self.stage3_freeze_backbone(train_step)

        batch, task_name, episode_id, sample_idx = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)
         
        all_pixels = []
        for key in self.pixel_keys:
            pixel = data[key]  # [B, T, C, H, W]
            all_pixels.append(pixel)
        
        all_pixels = torch.stack(all_pixels, dim=2) # [B,T,V, C, H, W]
        B,T,V,C,H,W = all_pixels.shape
        assert T == self.group_loss_window and V == len(self.pixel_keys)

        all_pixels = einops.rearrange(all_pixels, 'b t v c h w -> (b t v) c h w')

        all_pixels = self.normalize(all_pixels)
        all_pixels = self.crop_image(all_pixels)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feature_assume = self.img_encoder(all_pixels)
        patch_feature = feature_assume['x_norm_patchtokens']

        patch_feature = patch_feature.permute(0, 2, 1).view(B*T*V, -1, H//14, W//14)
        patch_feature = self.spatial_adapter(patch_feature)

        img_features = patch_feature.view(B*T, V, self.hidden_dim)
        
        proprio = data[self.proprio_key].float()
        proprio = self.proprio_projector(proprio)
        proprio = proprio.view(B*T, 1, self.hidden_dim)
        img_features = torch.cat([img_features, proprio], dim=1)

        lang_features = data[self.lang_key].float().view(B, 1, -1)   # b 1 d
        lang_features = self.language_projector(lang_features)
        lang_features = lang_features.repeat_interleave(T, dim=0)
        features = torch.cat([lang_features, img_features], dim=1)

        action_token = self._action_token.repeat(B*T, 1, 1)
        features = torch.cat([features, action_token], dim=1)
        features = self.backbone(features)
        action_features = features[:, -1:, :]  # [B*T, 1, hidden_dim]
      
        action_future = data["action_future"].float()   # [B T L D]
        action_future = einops.rearrange(action_future, 'b t l d -> (b t) l d')

        with torch.no_grad():
            pred_speed = self.ratio_head(action_features)  # [B*T]

        sampling_strategy = self.s1_strategy if train_step < self.s1_till_steps else ('full' if train_step < self.s2_till_steps else 'align')

        if sampling_strategy != 'align':   # stage1&2
            action_future_group = self.get_gt_action_group(action_future, sampling_strategy=sampling_strategy) 
        elif sampling_strategy == 'align': # stage3
            action_future_group = self.get_gt_action_group(action_future, sampling_strategy=sampling_strategy, s3_specified_speed=pred_speed)
        action_group = torch.stack(action_future_group, dim=1) # [B*T, num_gt , chunk_size, D]

        loss_dict = self.action_head(action_features, actions = action_group, optimize_target = self.optimize_target) #[B*T, num_gt]
        
        loss = loss_dict['loss']
        criterion = loss_dict['criterion']
        
        loss, a_selected = self.optimize_strategy(loss, criterion, sampling_strategy, train_step, pred_speed = pred_speed)

        stage = 's1' if train_step < self.s1_till_steps else ('s2' if train_step < self.s2_till_steps else 's3')

        loss_ratio = 0.0
        if stage == 's2':         
            a_selected = torch.stack(a_selected, dim=0) # list to tensor
            a_selected_rep = a_selected.repeat_interleave(T, dim=0) # to match action_features
            loss_ratio = self.ratio_head.loss(action_features, a_selected_rep)

        loss = loss * self.loss_coef
        
        metrics = {
            'loss': loss,
            'loss_ratio': loss_ratio,
        }

        if stage == 's3':
            a_selected = einops.rearrange(pred_speed, '(b T) -> b T',T = self.group_loss_window).mean(dim=1)

        return metrics, task_name, episode_id, sample_idx, a_selected, stage

    # region act
    def act(self, obs, proprio, lang_emb, norm_stats = None, step=None,return_action_features_for_eval=False, norm_to_minor=False,nospeed_temporal_agg=False):
        test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])

        all_pixels = []
        for key in self.pixel_keys:
            pixel = test_aug(obs[key]).cuda()
            pixel = pixel.unsqueeze(0)
            all_pixels.append(pixel)

        B = all_pixels[0].shape[0]
        V = len(self.pixel_keys)
        
        all_pixels = torch.stack(all_pixels, dim=1) # [B,V, C, H, W]
        all_pixels = einops.rearrange(all_pixels, 'b v c h w -> (b v) c h w')

        all_pixels = self.normalize(all_pixels)
        all_pixels = self.crop_image(all_pixels)
        patch_h = all_pixels.shape[-2] // 14
        patch_w = all_pixels.shape[-1] // 14
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feature_assume = self.img_encoder(all_pixels)

        patch_feature = feature_assume['x_norm_patchtokens']
        patch_feature = patch_feature.permute(0, 2, 1).view(B*V, -1, patch_h, patch_w)

        patch_feature = self.spatial_adapter(patch_feature)  # [BV, hidden_dim]
        img_features = patch_feature.view(B, V, self.hidden_dim) # B,V, hidden_dim
        
        if norm_stats is not None:
            min_proprio = torch.tensor(norm_stats['min'], device=self.device)
            max_proprio = torch.tensor(norm_stats['max'], device=self.device)
            if norm_to_minor:
                proprio = 2 * (proprio - min_proprio) / (max_proprio - min_proprio + 1e-5) - 1 # [-1 ,1]
            else:
                proprio = (proprio - min_proprio) / (max_proprio - min_proprio + 1e-5) # [0 ,1]

        proprio = self.proprio_projector(proprio)
        proprio = proprio.view(B,1, self.hidden_dim)
        img_features = torch.cat([img_features, proprio], dim=1) # B,V+1 d

        if not isinstance(lang_emb, torch.Tensor):
            lang_emb = torch.tensor(lang_emb, device=self.device)
        lang_features = lang_emb.to(self.device).float().view(1,1, -1)
        if return_action_features_for_eval: 
            lang_features = lang_features.repeat(B, 1, 1)

        lang_features = self.language_projector(lang_features)
        features = torch.cat([lang_features, img_features], dim=1) # B,1+V+1 d

        action_token = self._action_token.repeat(B, 1, 1)
        features = torch.cat([features, action_token], dim=1) # B,1+V+1+1 d
        features = self.backbone(features)
        action_features = features[:, -1:] # B,1,d

        pred_a = self.ratio_head(action_features) # [B] float tensor
        action = self.action_head(action_features) # [B, chunk_size, D]
        if norm_stats is not None:
            min_action = torch.tensor(norm_stats['min'], device=self.device)
            max_action = torch.tensor(norm_stats['max'], device=self.device)
            action = min_action + action * (max_action - min_action)

        if nospeed_temporal_agg:
            action = action.view(-1, self.num_queries, self.action_dim)
            self.all_time_actions[[step], step : step + self.num_queries] = action
            actions_for_curr_step = self.all_time_actions[:, step]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01 # 0.01 0.02
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            return action.cpu().numpy()[0], pred_a
        
        else:
            return action, pred_a # [B, chunk_size, D], [B]
        

    def eval(self, episode, norm_stats, t_eval_sampling_times, training_step):
        obs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in episode['obs'].items()}
        action_chunk = episode['action_chunks']
        lang_emb = episode['task_emb']

        proprio = obs[self.proprio_key].float()

        pred_actions, ratio_pred, action_features = self.act(obs, proprio, lang_emb, norm_stats, return_action_features_for_eval=True)

        action_future = action_chunk.to(self.device)   # [B L D]
        action_future_group = self.get_gt_action_group(action_future, sampling_strategy='full') 
        action_group = torch.stack(action_future_group, dim=1) # [B, num_gt , chunk_size, D]

        ratio_pred_list = ratio_pred.tolist()
        speed_index = [self.a_candidates_list.index(round(float(a), 1)) for a in ratio_pred_list] #B
        gt_action = [action_group[item,idx] for item,idx in enumerate(speed_index)]
        gt_action = torch.stack(gt_action, dim=0) # [B, chunk_size, D]

        a_selected_list = []
        for i in range(t_eval_sampling_times):
            loss_dict = self.action_head(action_features, actions = action_group, optimize_target = self.optimize_target) #[B, num_gt]
            loss = loss_dict['loss']
            criterion = loss_dict['criterion']
            loss, a_selected = self.optimize_strategy(loss, criterion, sampling_strategy='full', training_step = training_step, pred_speed = ratio_pred, is_eval = True)
            a_selected = torch.stack(a_selected, dim=0) # [B]
            a_selected = a_selected.tolist()
            a_selected_list.append(torch.tensor(a_selected, device=self.device))
            
        
        a_selected_group = torch.stack(a_selected_list, dim=1) # [B,t]

        print(pred_actions.shape, gt_action.shape, ratio_pred.shape, a_selected_group.shape)
        return pred_actions, gt_action, ratio_pred, a_selected_group


    def save_snapshot(self):
        model_keys = ["img_encoder", "spatial_adapter", "proprio_projector", 
                    "language_projector", "backbone", "action_head", "ratio_head"]
    
        payload = {
            k: self.__dict__['_modules'][k].state_dict() for k in model_keys
        }
        payload["_action_token"] = self._action_token.detach().cpu()

        return payload

    def load_snapshot(self, payload):
        model_keys = ["img_encoder", "spatial_adapter", "proprio_projector", 
                    "language_projector", "backbone", "action_head", "ratio_head"]
        
        for k in model_keys:
            if k in payload:
                self.__dict__['_modules'][k].load_state_dict(payload[k])
                
        token = payload.get("_action_token")
        if torch.is_tensor(token) and token.shape == self._action_token.shape:
            self._action_token.data.copy_(
                token.to(self._action_token.device, self._action_token.dtype))