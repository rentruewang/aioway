# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
import contextlib as ctxl
from collections import abc as cabc

from torch import nn, optim
from torchrl import collectors as rlcol
from torchrl import data as rldata
from torchrl import envs as rlenv
from torchrl import objectives as rlobj

__all__ = ["vec_env", "collector"]


@ctxl.contextmanager
def vec_env(name: str, processes: int) -> cabc.Generator[rlenv.ParallelEnv]:
    env = rlenv.GymEnv(name)
    parallel = rlenv.ParallelEnv(processes, lambda: env)

    try:
        yield parallel
    finally:
        parallel.close()


@ctxl.contextmanager
def collector(
    env: rlenv.EnvBase, policy: nn.Module, frames_per_batch: int, total_frames: int
) -> cabc.Generator[rlcol.Collector]:
    collector = rlcol.Collector(
        create_env_fn=env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

    try:
        yield collector
    finally:
        collector.shutdown()


def train_rl(
    loss_fn: rlobj.LossModule,
    optimizer: optim.Optimizer,
    collector: rlcol.Collector,
    buffer: rldata.ReplayBuffer,
    batch_size: int,
):
    for i, batch in enumerate(collector):
        buffer.extend(batch)

        # Sample a batch.
        sample = buffer.sample(torch.Size([batch_size]))
        loss = loss_fn(sample)

        # Standard PyTorch optimization step
        optimizer.zero_grad()
        loss["loss"].backward()
        optimizer.step()
