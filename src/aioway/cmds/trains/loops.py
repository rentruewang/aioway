# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib
import dataclasses as dcls

import torch
from torch import nn, optim

from aioway.io import Dset

from .callbacks import Callback, State

__all__ = ["Train", "Eval"]


# train ====


@dcls.dataclass(frozen=True)
class Train:
    """One pass of supervised training over a `Dset`: per batch, loss(model(input), target), backward, step.
    More epochs, validation and early stopping are for whoever calls it again."""

    batch_size: int
    callbacks: tuple[Callback, ...] = ()

    def __call__(
        self, model: nn.Module, optimizer: optim.Optimizer, loss, dset: Dset
    ) -> None:
        loader = dset.to_dataloader(
            batch_size=self.batch_size, shuffle=True, collate_fn=dset.collate_fn
        )

        with _set_training_mode(model, to=True):
            for step, batch in enumerate(loader, 1):
                value = loss(model(batch.input), batch.target)
                optimizer.zero_grad()
                value.backward()
                optimizer.step()

                state = State(model, optimizer, step, value.item())
                for cb in self.callbacks:
                    cb.on_step_end(state)


# eval ====


@dcls.dataclass(frozen=True)
class Eval:
    "The loss's mean over a `Dset`'s batches, in eval mode, without gradients: validation and test alike."

    batch_size: int

    def __call__(self, model: nn.Module, loss, dset: Dset) -> float:
        loader = dset.to_dataloader(
            batch_size=self.batch_size, shuffle=False, collate_fn=dset.collate_fn
        )

        with _set_training_mode(model, to=False), torch.no_grad():
            losses = [loss(model(batch.input), batch.target).item() for batch in loader]
        return sum(losses) / len(losses)


# helpers ====


@contextlib.contextmanager
def _set_training_mode(model: nn.Module, to: bool):
    "The model in training or eval mode inside, its old mode back after."
    was = model.training
    model.train(to)
    try:
        yield
    finally:
        model.train(was)
