import math
import random
import einops
import torch
import torch_dct
from torch import nn
import utils.agent_utils as utils

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


class ActorACTSpeed(nn.Module):
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
        syn_optimize_ratio,
        action_overcollect_ratio,
        step_loss_type,
        step_loss_ratio,
        constrain_speed_via_ratio_head,
        constrain_from_ratio_coef,
        constrain_coef_linear_up,
        constrain_other_coef_linear_down,
        constrain_start_step,
        new_loss_args,
        device="cuda",
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
        self.syn_optimize_ratio = syn_optimize_ratio
        self.constrain_coef_linear_up = constrain_coef_linear_up
        self.constrain_other_coef_linear_down = constrain_other_coef_linear_down
        self.constrain_start_step = constrain_start_step
        self.action_overcollect_ratio = action_overcollect_ratio

        self.constrain_speed_via_ratio_head = constrain_speed_via_ratio_head
        self.constrain_from_ratio_coef = constrain_from_ratio_coef
        self.step_loss_type = step_loss_type
        self.step_loss_ratio = step_loss_ratio

        self.new_loss_args = new_loss_args
        speed_range = new_loss_args.get("speed_range")
        speed_step = new_loss_args.get("speed_step")
        self.speed_max, self.speed_min = speed_range[1], speed_range[0]
        self.speed_num_steps = (
            int(round((self.speed_max - self.speed_min) / speed_step)) + 1
        )
        self.speed_num_steps = max(self.speed_num_steps, 1)
        self.a_candidates = torch.linspace(
            self.speed_min,
            self.speed_max,
            self.speed_num_steps,
            device=self.device,
        )
        self.a_candidates_list = [
            round(float(value), 1)
            for value in torch.linspace(
                self.speed_min, self.speed_max, self.speed_num_steps
            ).tolist()
        ]

        self.spatial_adapter = nn.Identity()
        self.proprio_projector = nn.Identity()
        self.language_projector = nn.Identity()
        self.backbone = nn.Identity()

    def get_gt_action_group(
        self, actions, sampling_strategy="random", s3_specified_speed=None
    ):
        expected_length = math.ceil(
            self.num_queries * self.action_overcollect_ratio * self.speed_max
        )
        assert actions.shape[1] == expected_length, (
            f"actions.shape: {actions.shape}, self.num_queries: {self.num_queries}, "
            f"self.action_overcollect_ratio: {self.action_overcollect_ratio}"
        )

        compressed_gt_actions = []

        if sampling_strategy == "random":
            speed = random.choice(self.a_candidates_list)
            compressed_gt_actions.append(
                compress(
                    actions=actions,
                    acc_ratio=speed,
                    action_overcollect_ratio=self.action_overcollect_ratio,
                    num_queries=self.num_queries,
                )
            )
        elif sampling_strategy in {"full", "mean"}:
            for speed in self.a_candidates_list:
                compressed_gt_actions.append(
                    compress(
                        actions=actions,
                        acc_ratio=speed,
                        action_overcollect_ratio=self.action_overcollect_ratio,
                        num_queries=self.num_queries,
                    )
                )
        elif sampling_strategy == "align":
            assert s3_specified_speed is not None
            compressed_gt_action = [
                compress(
                    actions=actions[index : index + 1],
                    acc_ratio=s3_specified_speed[index],
                    action_overcollect_ratio=self.action_overcollect_ratio,
                    num_queries=self.num_queries,
                )
                for index in range(len(s3_specified_speed))
            ]
            compressed_gt_actions.append(torch.cat(compressed_gt_action, dim=0))

        return compressed_gt_actions

    def optimize_strategy(
        self,
        loss,
        criterion,
        sampling_strategy,
        training_step,
        pred_speed=None,
        is_eval=False,
    ):
        if sampling_strategy != "full":
            return loss.mean(), 0

        criterion = criterion.detach().clone()
        loss_step = (
            torch.log(self.a_candidates + 1)
            if "log" in self.step_loss_type
            else self.a_candidates
        )
        max_loss_step, min_loss_step = loss_step.max(), loss_step.min()

        if "denominator" in self.step_loss_type:
            offset = (
                max_loss_step - self.step_loss_ratio * min_loss_step
            ) / (self.step_loss_ratio - 1.0)
            loss_step = 1 / (offset + loss_step)
        elif "numerator" in self.step_loss_type:
            offset = (
                self.step_loss_ratio * max_loss_step - min_loss_step
            ) / (self.step_loss_ratio - 1.0)
            loss_step = offset - loss_step
        else:
            raise ValueError(f"Invalid step loss type: {self.step_loss_type}")

        loss_ratio_refer = 0.0
        other_coef = 1.0
        if (
            pred_speed is not None
            and self.constrain_speed_via_ratio_head
            and training_step >= self.constrain_start_step
        ):
            centers = pred_speed.unsqueeze(1)
            speed_candidates = self.a_candidates.unsqueeze(0)
            loss_ratio_refer = torch.abs(speed_candidates - centers)
            linear_ratio = min(
                1.0,
                max(
                    0.0,
                    (training_step - self.constrain_start_step)
                    / (self.s2_till_steps - self.constrain_start_step),
                ),
            )
            if self.constrain_coef_linear_up:
                loss_ratio_refer = loss_ratio_refer * linear_ratio
            if (
                self.constrain_coef_linear_up
                and self.constrain_other_coef_linear_down
            ):
                other_coef = 1.0 - linear_ratio

        whole_criterion = (
            other_coef * criterion * loss_step + loss_ratio_refer * criterion
        )
        window_size = 1 if is_eval else self.group_loss_window
        whole_criterion = einops.rearrange(
            whole_criterion, "(b t) n -> b t n", t=window_size
        ).mean(dim=1)
        selected_loss_id = whole_criterion.argmin(dim=1)

        loss = einops.rearrange(
            loss, "(b t) n -> b t n", t=window_size
        ).mean(dim=1)
        batch_indices = torch.arange(loss.shape[0], device=loss.device)
        loss = loss[batch_indices, selected_loss_id]
        selected_speeds = [
            self.a_candidates[index] for index in selected_loss_id
        ]

        return loss.mean(), selected_speeds

    def update(self, expert_replay_iter, train_step):
        batch, task_name, episode_id, sample_idx = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)

        all_pixels = [data[key] for key in self.pixel_keys]
        all_pixels = torch.stack(all_pixels, dim=2)
        B, T, V, C, H, W = all_pixels.shape
        assert T == self.group_loss_window and V == len(self.pixel_keys)

        images = einops.rearrange(
            all_pixels, "b t v c h w -> (b t) v c h w"
        )
        proprio = data[self.proprio_key].float()
        qpos = einops.rearrange(proprio, "b t d -> (b t) d")
        action_cond = {"qpos": qpos, "images": images}

        action_future = data["action_future"].float()
        action_future = einops.rearrange(
            action_future, "b t l d -> (b t) l d"
        )

        if train_step < self.s1_till_steps:
            sampling_strategy = self.s1_strategy
        elif train_step < self.s2_till_steps:
            sampling_strategy = "full"
        else:
            sampling_strategy = "align"

        prior_features = self.action_head.get_features(action_cond)
        if sampling_strategy == "align":
            pred_speed = self.ratio_head(prior_features.unsqueeze(1))
            action_future_group = self.get_gt_action_group(
                action_future,
                sampling_strategy=sampling_strategy,
                s3_specified_speed=pred_speed,
            )
        else:
            action_future_group = self.get_gt_action_group(
                action_future, sampling_strategy=sampling_strategy
            )

        action_group = torch.stack(action_future_group, dim=1)
        loss_dict = self.action_head(action_cond, actions=action_group)
        loss = loss_dict["loss"]
        criterion = loss_dict["criterion"]
        action_features = loss_dict["features"]

        with torch.no_grad():
            pred_speed = (
                self.ratio_head(action_features.unsqueeze(1))
                if self.syn_optimize_ratio
                else None
            )

        loss, selected_speeds = self.optimize_strategy(
            loss,
            criterion,
            sampling_strategy,
            train_step,
            pred_speed=pred_speed,
        )

        if train_step < self.s1_till_steps:
            stage = "s1"
        elif train_step < self.s2_till_steps:
            stage = "s2"
        else:
            stage = "s3"

        loss_ratio = 0.0
        if stage == "s2" and self.syn_optimize_ratio:
            selected_speeds = torch.stack(selected_speeds, dim=0)
            selected_speeds_rep = selected_speeds.repeat_interleave(T, dim=0)
            loss_ratio = self.ratio_head.loss(
                action_features.unsqueeze(1), selected_speeds_rep
            )

        loss = loss * self.loss_coef
        metrics = {"loss": loss, "loss_ratio": loss_ratio}

        if stage == "s3":
            selected_speeds = einops.rearrange(
                pred_speed, "(b t) -> b t", t=self.group_loss_window
            ).mean(dim=1)

        return (
            metrics,
            task_name,
            episode_id,
            sample_idx,
            selected_speeds,
            stage,
        )

    def act(
        self,
        obs,
        proprio,
        lang_emb=None,
        norm_stats=None,
        return_action_features_for_eval=False,
        norm_to_minor=True,
    ):
        all_pixels = [obs[key] for key in self.pixel_keys]
        B = all_pixels[0].shape[0]
        all_pixels = torch.stack(all_pixels, dim=1)

        if norm_stats is not None:
            min_proprio = torch.tensor(norm_stats["min"], device=self.device)
            max_proprio = torch.tensor(norm_stats["max"], device=self.device)
            proprio = (
                2 * (proprio - min_proprio) / (max_proprio - min_proprio + 1e-5)
                - 1
            )

        qpos = proprio.float()
        action_cond = {"qpos": qpos, "images": all_pixels}

        eval_batch_size = 50
        with torch.no_grad():
            if B <= eval_batch_size:
                action, action_features = self.action_head(action_cond)
            else:
                action_chunks = []
                feature_chunks = []
                for start in range(0, B, eval_batch_size):
                    end = min(start + eval_batch_size, B)
                    mini_cond = {
                        "qpos": qpos[start:end],
                        "images": all_pixels[start:end],
                    }
                    action_chunk, feature_chunk = self.action_head(mini_cond)
                    action_chunks.append(action_chunk)
                    feature_chunks.append(feature_chunk)
                action = torch.cat(action_chunks, dim=0)
                action_features = torch.cat(feature_chunks, dim=0)

        pred_speed = self.ratio_head(action_features.unsqueeze(1))
        if return_action_features_for_eval:
            return action, pred_speed, action_features.unsqueeze(1)
        return action, pred_speed

    def eval(self, episode, norm_stats, t_eval_sampling_times, training_step):
        obs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in episode["obs"].items()
        }
        action_chunk = episode["action_chunks"]
        lang_emb = episode["task_emb"]
        proprio = obs[self.proprio_key].float()

        pred_actions, ratio_pred, action_features = self.act(
            obs,
            proprio,
            lang_emb,
            norm_stats,
            return_action_features_for_eval=True,
        )

        action_future = action_chunk.to(self.device)
        action_future_group = self.get_gt_action_group(
            action_future, sampling_strategy="full"
        )
        action_group = torch.stack(action_future_group, dim=1)

        speed_index = [
            int(torch.argmin(torch.abs(self.a_candidates - float(speed))).item())
            for speed in ratio_pred.tolist()
        ]
        gt_action = torch.stack(
            [action_group[item, index] for item, index in enumerate(speed_index)],
            dim=0,
        )

        all_pixels = torch.stack([obs[key] for key in self.pixel_keys], dim=1)
        eval_batch_size = 50
        selected_speed_groups = []

        for _ in range(t_eval_sampling_times):
            with torch.no_grad():
                B = all_pixels.shape[0]
                if B <= eval_batch_size:
                    action_cond = {"qpos": proprio, "images": all_pixels}
                    loss_dict = self.action_head(
                        action_cond, actions=action_group
                    )
                    loss = loss_dict["loss"]
                    criterion = loss_dict["criterion"]
                else:
                    loss_chunks = []
                    criterion_chunks = []
                    for start in range(0, B, eval_batch_size):
                        end = min(start + eval_batch_size, B)
                        mini_cond = {
                            "qpos": proprio[start:end],
                            "images": all_pixels[start:end],
                        }
                        loss_dict = self.action_head(
                            mini_cond, actions=action_group[start:end]
                        )
                        loss_chunks.append(loss_dict["loss"])
                        criterion_chunks.append(loss_dict["criterion"])
                    loss = torch.cat(loss_chunks, dim=0)
                    criterion = torch.cat(criterion_chunks, dim=0)

            loss, selected_speeds = self.optimize_strategy(
                loss,
                criterion,
                sampling_strategy="full",
                training_step=training_step,
                pred_speed=ratio_pred,
                is_eval=True,
            )
            selected_speed_groups.append(torch.stack(selected_speeds, dim=0))

        selected_speed_groups = torch.stack(selected_speed_groups, dim=1)
        print(
            pred_actions.shape,
            gt_action.shape,
            ratio_pred.shape,
            selected_speed_groups.shape,
        )
        return pred_actions, gt_action, ratio_pred, selected_speed_groups

    def save_snapshot(self):
        return {
            key: self.__dict__["_modules"][key].state_dict()
            for key in ["action_head", "ratio_head"]
        }

    def load_snapshot(self, payload):
        for key in ["action_head", "ratio_head"]:
            if key in payload:
                self.__dict__["_modules"][key].load_state_dict(payload[key])
