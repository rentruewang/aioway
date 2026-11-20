# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import math

from torch import nn, optim

__all__ = ["State", "Callback", "EarlyStopping", "PrintLoss"]


# state ====


@dcls.dataclass
class State:
    "What changes while training, one version at a time: callbacks read it, and may set `stop`."

    model: nn.Module
    optimizer: optim.Optimizer
    step: int = 0
    loss: float = math.nan
    valid_loss: float = math.nan
    stop: bool = False


# callbacks ====


class Callback:
    "Every hook does nothing by default: override the ones needed."

    def on_step_end(self, state: State) -> None:
        pass

    def on_valid_end(self, state: State) -> None:
        pass


@dcls.dataclass
class EarlyStopping(Callback):
    "Stop once the valid loss has not improved for `patience` checks. Not reset between runs: one per `Train` call."

    patience: int
    best: float = dcls.field(default=math.inf, init=False)
    bad: int = dcls.field(default=0, init=False)

    def on_valid_end(self, state):
        if state.valid_loss < self.best:
            self.best, self.bad = state.valid_loss, 0
        else:
            self.bad += 1

        if self.bad >= self.patience:
            state.stop = True


@dcls.dataclass
class PrintLoss(Callback):
    "Print the training loss every `every` steps, and the valid loss whenever it is checked."

    every: int = 100

    def on_step_end(self, state):
        if state.step % self.every == 0:
            print(f"step {state.step}: loss {state.loss:.4f}")

    def on_valid_end(self, state):
        print(f"step {state.step}: valid {state.valid_loss:.4f}")
