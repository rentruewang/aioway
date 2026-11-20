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
    every: int = 100
    callbacks: tuple[Callback, ...] = ()

    def __call__(
        self, model: nn.Module, optimizer: optim.Optimizer, objective, data, valid=None
    ) -> None:
        state = State(model, optimizer)
        batches = _cycle(data)

        with _set_training_mode(model, to=True):
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
        with _set_training_mode(model, to=False), torch.no_grad():
            losses = [objective(model, *batch).item() for batch in data]
        return sum(losses) / len(losses)


# helpers ====


def _cycle(data):
    "The data's batches, over and over (iterated again each pass, so a shuffling loader reshuffles)."
    while True:
        yield from data


@contextlib.contextmanager
def _set_training_mode(model: nn.Module, to: bool):
    "The model in training or eval mode inside, its old mode back after."
    was = model.training
    model.train(to)
    try:
        yield
    finally:
        model.train(was)
