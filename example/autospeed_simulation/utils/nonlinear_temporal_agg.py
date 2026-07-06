import logging
import math
from typing import List, Optional

import torch

def setup_logger(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

class NonlinearTemporalAgg:
    def __init__(
        self,
        window_range=1.0,
        recency_decay=0.01,
        max_candidates=320,
        eps=1e-8,
        debug=False,
    ):
        self.window_range = float(window_range)
        self.recency_decay = float(recency_decay)
        self.max_candidates = int(max_candidates)
        self.eps = float(eps)
        self.debug = debug

        self.action_buffer = []
        self.timespan_buffer = [0.0]
        self.current_progress = 0.0

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _as_float(self, value):
        if torch.is_tensor(value):
            value = value.detach().flatten()
            if value.numel() == 0:
                return 0.0
            return float(value[0].item())
        return float(value)

    def _dct_sample(self, chunk: torch.Tensor, tau: float) -> torch.Tensor:
        if chunk.shape[-2] == 1:
            return chunk.select(dim=-2, index=0)

        n = chunk.shape[-2]
        tau = float(max(0.0, min(float(n - 1), tau)))
        device = chunk.device
        dtype = chunk.dtype

        time_idx = torch.arange(n, device=device, dtype=dtype)
        freq_idx = torch.arange(n, device=device, dtype=dtype)
        alpha = torch.full((n,), math.sqrt(2.0 / n), device=device, dtype=dtype)
        alpha[0] = math.sqrt(1.0 / n)

        dct_basis = alpha[:, None] * torch.cos(
            math.pi / n * (time_idx[None, :] + 0.5) * freq_idx[:, None]
        )
        eval_basis = alpha * torch.cos(
            math.pi / n * (torch.as_tensor(tau, device=device, dtype=dtype) + 0.5) * freq_idx
        )

        moved = chunk.movedim(-2, 0)
        rest_shape = moved.shape[1:]
        flat = moved.reshape(n, -1)
        coeff = dct_basis @ flat
        sampled = eval_basis @ coeff
        return sampled.reshape(rest_shape)

    def _entry_progress_bounds(self, entry):
        speed = entry["speed"]
        length = entry["length"]
        start = entry["base_progress"]
        end = start + speed * max(length - 1, 0)
        return (min(start, end), max(start, end))

    def _prune_buffer(self):
        min_progress = self.current_progress - self.window_range
        max_progress = self.current_progress + self.window_range
        new_buffer = []
        for entry in self.action_buffer:
            start, end = self._entry_progress_bounds(entry)
            if end >= min_progress and start <= max_progress:
                new_buffer.append(entry)
        self.action_buffer = new_buffer[-self.max_candidates :]

    def _sample_entry_at_current_progress(self, entry) -> Optional[torch.Tensor]:
        speed = entry["speed"]
        if abs(speed) < self.eps:
            tau = 0.0
            distance = abs(self.current_progress - entry["base_progress"])
        else:
            tau = (self.current_progress - entry["base_progress"]) / speed
            distance = 0.0
            if tau < 0.0:
                distance = abs(tau * speed)
            elif tau > entry["length"] - 1:
                distance = abs((tau - (entry["length"] - 1)) * speed)

        if distance > self.window_range:
            return None
        return self._dct_sample(entry["actions"], tau)

    def _get_current_actions(self, step):
        candidates: List[torch.Tensor] = []
        weights = []

        for entry in self.action_buffer[-self.max_candidates :]:
            action = self._sample_entry_at_current_progress(entry)
            if action is None:
                continue

            recency = max(0.0, float(step - entry["step"]))
            weights.append(math.exp(-self.recency_decay * recency))
            candidates.append(action)

        if not candidates:
            if not self.action_buffer:
                raise RuntimeError("No action candidates are available for temporal aggregation.")
            latest = self.action_buffer[-1]
            return latest["actions"].select(dim=-2, index=0)

        stacked_actions = torch.stack(candidates, dim=0)
        weights_tensor = torch.as_tensor(
            weights,
            device=stacked_actions.device,
            dtype=stacked_actions.dtype,
        )
        norm_weights = weights_tensor / weights_tensor.sum().clamp_min(self.eps)
        view_shape = (len(candidates),) + (1,) * (stacked_actions.dim() - 1)
        aggregated = (stacked_actions * norm_weights.view(view_shape)).sum(dim=0)

        if self.debug:
            self.logger.debug(
                "aggregated %d candidates at step=%s progress=%.4f",
                len(candidates),
                step,
                self.current_progress,
            )

        return aggregated

    def record_and_get_current_actions(self, actions, speed, step):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        if actions.dim() < 2:
            raise ValueError("actions must have a time dimension at dim=-2")

        last_timespan = self.timespan_buffer[-1]
        self.current_progress += last_timespan
        self._prune_buffer()

        speed_value = self._as_float(speed)
        entry = {
            "step": int(step),
            "base_progress": self.current_progress,
            "speed": speed_value,
            "length": int(actions.shape[-2]),
            "actions": actions,
        }
        self.action_buffer.append(entry)
        self.action_buffer = self.action_buffer[-self.max_candidates :]
        self.timespan_buffer.append(speed_value)

        return self._get_current_actions(step)

    def reset(self):
        self.action_buffer = []
        self.timespan_buffer = [0.0]
        self.current_progress = 0.0


if __name__ == "__main__":
    setup_logger(debug=True)
    nonlinear_temporal_agg = NonlinearTemporalAgg(debug=True)

    actions = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    print(nonlinear_temporal_agg.record_and_get_current_actions(actions, 1, 0))

    actions = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    print(nonlinear_temporal_agg.record_and_get_current_actions(actions, 2, 1))

    actions = torch.tensor([[3.0], [3.5], [4.0], [4.5]])
    print(nonlinear_temporal_agg.record_and_get_current_actions(actions, 0.5, 2))

    actions = torch.tensor([[4.0], [5.0], [6.0], [7.0]])
    print(nonlinear_temporal_agg.record_and_get_current_actions(actions, 1, 3))