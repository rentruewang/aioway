# Copyright (c) AIoWay Authors - All Rights Reserved

from debugpy.launcher.debuggee import process
from torchrl import envs as trl_envs
from torchrl import collectors as trl_cols
from torchrl import objectives as trl_objs
from torchrl import data as trl_data

import contextlib as ctxl


from torch import nn, optim

from collections import abc as cabc

__all__ = ["vec_env", "collector"]


@ctxl.contextmanager
def vec_env(name: str, processes: int) -> cabc.Generator[trl_envs.ParallelEnv]:
    env = trl_envs.GymEnv(name)
    parallel = trl_envs.ParallelEnv(processes, lambda: env)

    try:
        yield parallel
    finally:
        parallel.close()


@ctxl.contextmanager
def collector(
    env: trl_envs.EnvBase, policy: nn.Module, frames_per_batch: int, total_frames: int
) -> cabc.Generator[trl_cols.Collector]:
    collector = trl_cols.Collector(
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
    loss_fn: trl_objs.LossModule,
    optimizer: optim.Optimizer,
    collector: trl_cols.Collector,
    buffer: trl_data.ReplayBuffer,
    batch_size: int,
):
    for i, batch in enumerate(collector):
        buffer.extend(batch)

        # Sample a batch.
        sample = buffer.sample(batch_size)
        loss = loss_fn(sample)

        # Standard PyTorch optimization step
        optimizer.zero_grad()
        loss["loss"].backward()
        optimizer.step()
