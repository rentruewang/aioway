# Copyright (c) AIoWay Authors - All Rights Reserved

"The explainer based on `captum.influence`."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from captum import influence
from torch import nn
from torch.utils import data as du

from aioway.ml.tasks import Task

__all__ = [
    "CaptumInfluenceTask",
    "build_influence_dcls",
    "BuildTracInCP",
    "BuildTracInCPFast",
    "BuildNaiveInfluenceFunction",
]


@typing.dataclass_transform(frozen_default=True, kw_only_default=True)
def build_influence_dcls(cls):
    return dcls.dataclass(frozen=True, kw_only=True)(cls)


# The base class ====


@dcls.dataclass
class CaptumInfluenceTask(Task):
    """
    The explainer based on `captum.influence.DataInfluence`.
    """

    @build_influence_dcls
    class Strategy(abc.ABC):
        """
        Strategy for building a `captum.influence.DataInfluence`
        for a module and a training corpus.

        Subclasses set `_CAPTUM_CLASS` (subclass of `DataInfluence`)
        `self` must decompose to kwargs of its constructor.

        The module's current weights are the only checkpoint.
        """

        _CAPTUM_CLASS: typing.ClassVar[cabc.Callable[..., influence.DataInfluence]]
        "The `captum` class to build."

        _CHECKPOINTS: typing.ClassVar[dict[str, typing.Any]] = {
            "checkpoints": ["current"]
        }
        "The constructor's checkpoint argument. Overridable."

        batch_size: int = 64
        "The batch size over the training corpus."

        def __call__(
            self, module: nn.Module, corpus: du.Dataset
        ) -> influence.DataInfluence:
            return self._CAPTUM_CLASS(
                model=module,
                train_dataset=corpus,
                checkpoints_load_func=_keep_weights,
                **self._CHECKPOINTS,
                **self.as_kwargs(),
            )

        def total(
            self, method: influence.DataInfluence, loader: du.DataLoader
        ) -> torch.Tensor:
            "Each corpus example's influence, summed over the loader: one corpus pass per batch."
            inf = [method.influence((x, y)).sum(0) for x, y in loader]
            result = sum(inf)
            assert isinstance(result, torch.Tensor)
            return result

        def as_kwargs(self) -> dict[str, typing.Any]:
            "The fields, as keyword arguments to the constructor."
            return {f.name: getattr(self, f.name) for f in dcls.fields(self)}

    cfg: Strategy
    "The influence build function."

    corpus: du.Dataset
    "The training examples to score, read in order: result[j] belongs to corpus[j]."

    batch_size: int = 256
    "The batch size to use to iterate over the dataset."

    def __call__(self, dset: du.Dataset, module: nn.Module) -> torch.Tensor:
        """
        Each corpus example's influence on the whole `dset`, summed. Shape (len(corpus),).
        Positive: training on that example lowers the loss on `dset`.

        Args:
            dset: The examples to explain, (x, y) pairs.
            module: An `nn.Module` that outputs one score per class, shape (batch, classes).
        """

        method = self.cfg(module, self.corpus)
        loader = du.DataLoader(dset, batch_size=self.batch_size)

        return self.cfg.total(method=method, loader=loader)


# The strategy implementations ====


def _keep_weights(module: nn.Module, checkpoint: str) -> float:
    "The only checkpoint is the module as passed in: nothing to load. Returns the learning rate."

    return 1.0


@build_influence_dcls
class BuildTracInCP(CaptumInfluenceTask.Strategy):
    _CAPTUM_CLASS = influence.TracInCP

    layers: list[str] | None = None
    "The layers whose weights' gradients are compared. `None`: all."

    loss_fn: nn.Module = dcls.field(
        default_factory=lambda: nn.CrossEntropyLoss(reduction="sum")
    )
    "Reduction `sum` with `sample_wise_grads_per_batch=True`, `none` with `False`."

    sample_wise_grads_per_batch: bool = True

    def total(self, method, loader):
        # Only 1 corpus pass, don't use `sum`.
        return method.influence(loader, aggregate=True)[0]


@build_influence_dcls
class BuildTracInCPFast(CaptumInfluenceTask.Strategy):
    _CAPTUM_CLASS = influence.TracInCPFast

    final_fc_layer: str
    "The name of the last linear layer, the only one whose gradients are compared."

    loss_fn: nn.Module = dcls.field(
        default_factory=lambda: nn.CrossEntropyLoss(reduction="sum")
    )
    "Reduction `sum` or `mean`."


@build_influence_dcls
class BuildNaiveInfluenceFunction(CaptumInfluenceTask.Strategy):
    _CAPTUM_CLASS = influence.NaiveInfluenceFunction
    _CHECKPOINTS = {"checkpoint": "current"}

    layers: list[str]
    "The layers whose weights enter the Hessian, (P x P) for their P weights: keep it small."

    loss_fn: nn.Module = dcls.field(
        default_factory=lambda: nn.CrossEntropyLoss(reduction="none")
    )
    "Reduction `none`."

    projection_dim: int | None = 50
    hessian_reg: float = 1e-6
