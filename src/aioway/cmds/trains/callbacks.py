# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

from torch import nn, optim

__all__ = ["State", "Callback", "PrintLoss"]


# State ====


@dcls.dataclass(frozen=True)
class State:
    "What a callback sees after a training step: built fresh each step, cheaply."

    model: nn.Module
    optimizer: optim.Optimizer
    step: int
    loss: float


# Callbacks ====


class Callback:
    "Every hook does nothing by default: override the ones needed."

    def on_step_end(self, state: State) -> None:
        pass


@dcls.dataclass(frozen=True)
class PrintLoss(Callback):
    "Print the training loss every `every` steps."

    every: int = 100

    def on_step_end(self, state):
        if state.step % self.every == 0:
            print(f"step {state.step}: loss {state.loss:.4f}")
