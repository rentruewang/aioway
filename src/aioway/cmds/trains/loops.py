# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib
import dataclasses as dcls

import torch
from torch import nn, optim

from .callbacks import Callback, State

__all__ = ["Train", "Eval"]


# train ====


@dcls.dataclass(frozen=True)
class Train:
    """Supervised training: descend objective(model, *batch) over the data's batches for `steps` steps, cycling
    the data. Every `every` steps, the objective's mean on valid (if given). Callbacks watch, and may stop it.
    """

    steps: int
    lr: float = 3e-4
    every: int = 100
    callbacks: tuple[Callback, ...] = ()

    def __call__(self, model: nn.Module, objective, data, valid=None) -> None:
        state = State(model, optim.AdamW(model.parameters(), lr=self.lr))
        batches = _cycle(data)

        with _mode(model, training=True):
            while state.step < self.steps and not state.stop:
                loss = objective(model, *next(batches))
                state.optimizer.zero_grad()
                loss.backward()
                state.optimizer.step()

                state.step += 1
                state.loss = loss.item()
                for cb in self.callbacks:
                    cb.on_step_end(state)

                if valid is not None and state.step % self.every == 0:
                    state.valid_loss = Eval()(model, objective, valid)
                    for cb in self.callbacks:
                        cb.on_valid_end(state)


# eval ====


@dcls.dataclass(frozen=True)
class Eval:
    "The objective's mean over the data's batches, in eval mode, without gradients: validation and test alike."

    def __call__(self, model: nn.Module, objective, data) -> float:
        with _mode(model, training=False), torch.no_grad():
            losses = [objective(model, *batch).item() for batch in data]
        return sum(losses) / len(losses)


# helpers ====


def _cycle(data):
    "The data's batches, over and over (iterated again each pass, so a shuffling loader reshuffles)."
    while True:
        yield from data


@contextlib.contextmanager
def _mode(model: nn.Module, training: bool):
    "The model in training or eval mode inside, its old mode back after."
    was = model.training
    model.train(training)
    try:
        yield
    finally:
        model.train(was)
