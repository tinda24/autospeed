import random
import numpy as np
import torch


def _worker_init_fn(worker_id):
    seed = int(np.random.get_state()[1][0]) + int(worker_id)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def make_expert_replay_loader(iterable, batch_size, num_workers):
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
    )
    if num_workers and num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
        )

    loader = torch.utils.data.DataLoader(iterable, **loader_kwargs)
    return loader
