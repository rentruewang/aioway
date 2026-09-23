# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import enum
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch import nn
from torch.nn import functional as F

from .tasks import BatchIter, Task

__all__ = ["SupervisedTask", "TaskKind", "LossFn"]

type LossFn = cabc.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class TaskKind(enum.StrEnum):
    """
    The kind of supervised problem. Decides the default loss
    and how targets are coerced before the loss is computed.
    """

    REGRESSION = "regression"
    "Real-valued targets, same shape as predictions. Loss: MSE."

    BINARY = "binary"
    "One logit per sample, targets in {0, 1}. Loss: BCE with logits."

    MULTICLASS = "multiclass"
    "Logits of shape (N, C), integer class targets of shape (N,). Loss: cross entropy."

    MULTILABEL = "multilabel"
    "Logits of shape (N, C), 0/1 targets of shape (N, C). Loss: BCE with logits."

    @property
    def default_loss(self) -> LossFn:
        "The default loss for this kind of task."

        match self:
            case TaskKind.REGRESSION:
                return _regression
            case TaskKind.BINARY:
                return _binary
            case TaskKind.MULTICLASS:
                return _multiclass
            case TaskKind.MULTILABEL:
                return _multilabel
            case _:
                typing.assert_never(self)


def _regression(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target.to(pred.dtype).view_as(pred))


def _binary(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Accept (N,) or (N, 1) logits against (N,) or (N, 1) targets.
    pred = pred.reshape(-1)
    target = target.reshape(-1).to(pred.dtype)
    return F.binary_cross_entropy_with_logits(pred, target)


def _multiclass(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == pred.ndim:
        # Probabilities / soft labels: cross_entropy supports these as float.
        return F.cross_entropy(pred, target.to(pred.dtype))
    return F.cross_entropy(pred, target.long())


def _multilabel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred, target.to(pred.dtype))


@dcls.dataclass
class SupervisedTask(Task[td.TensorDict]):
    """
    A standard supervised learning task.

    Each batch is a `TensorDict` holding an input under `input_key`
    and a target under `target_key`. A step runs
    forward -> loss -> backward -> optimizer update.

    The loss is chosen from `kind` unless `loss_fn` is given explicitly.
    The model is expected to output raw logits for classification kinds.
    """

    model: nn.Module
    "The model being trained."

    optimizer: torch.optim.Optimizer
    "Optimizer over `model.parameters()`."

    data: BatchIter[td.TensorDict]
    "Source of real training batches."

    fake_fn: cabc.Callable[[], td.TensorDict]
    "Produces one cheap, correctly-shaped fake batch."

    kind: TaskKind | str
    "The kind of problem, which decides the default loss."

    loss_fn: LossFn | None = None
    "Overrides the loss chosen by `kind` when set."

    input_key: str = "input"
    target_key: str = "target"

    grad_clip: float | None = None
    "If set, clip the global gradient norm to this value."

    last_loss: torch.Tensor | None = dcls.field(default=None, init=False)
    "Detached loss of the most recent step."

    _fake_mode: bool = dcls.field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.kind = TaskKind(self.kind)
        if self.loss_fn is None:
            self.loss_fn = self.kind.default_loss

    def fake(self) -> td.TensorDict:
        batch = self.fake_fn()
        self._check_keys(batch)
        return batch

    def iterator(self) -> BatchIter[td.TensorDict]:
        return self.data

    def step(self, batch: td.TensorDict, /) -> None:
        self._check_keys(batch)
        inputs = batch[self.input_key]
        target = batch[self.target_key]
        loss_fn = typing.cast(LossFn, self.loss_fn)

        if self._fake_mode:
            # Cheap path: forward + loss only, no gradients, no parameter updates.
            with torch.no_grad():
                pred = self.model(inputs)
                self.last_loss = loss_fn(pred, target).detach()
            return

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pred = self.model(inputs)
        loss = loss_fn(pred, target)
        loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.last_loss = loss.detach()

    @typing.override
    def fake_step(self) -> None:
        with self._fake():
            super().fake_step()

    @contextlib.contextmanager
    def _fake(self) -> cabc.Generator[None]:
        prev, self._fake_mode = self._fake_mode, True
        try:
            yield
        finally:
            self._fake_mode = prev

    def _check_keys(self, batch: td.TensorDict) -> None:
        missing = {self.input_key, self.target_key} - set(batch.keys())
        if missing:
            raise KeyError(f"Batch is missing keys: {sorted(missing)}")
