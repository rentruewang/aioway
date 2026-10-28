# Copyright (c) AIoWay Authors - All Rights Reserved

"The explainer based on `captum.attr`."

import abc
import dataclasses as dcls
import typing

import torch
from captum import attr
from torch import nn
from torch.utils import data as du

from aioway.ml.tasks import Task

from ._utils import AttrAdaptor

__all__ = [
    "CaptumAttrTask",
    "build_attr_dcls",
    "BuildSaliency",
    "BuildIntegratedGradients",
    "BuildDeepLift",
    "BuildFeatureAblation",
    "BuildKernelShap",
]


@typing.dataclass_transform(frozen_default=True)
def build_attr_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


# The base class ====


@dcls.dataclass
class CaptumAttrTask(Task):
    """
    The explainer based on `captum.attr.Attribution`.

    Output is of shape `input.shape[1:]` (features).
    """

    @build_attr_dcls
    class Strategy(abc.ABC):
        """
        Builds a `captum.attr.Attribution` for a module.

        Subclasses set `_CAPTUM_CLASS`
        `self` must decompose to kwargs of its `.attribute`.
        """

        _CAPTUM_CLASS: typing.ClassVar[type[attr.Attribution]]
        "The `captum` class to build."

        def __call__(self, module: nn.Module) -> attr.Attribution:
            return self._CAPTUM_CLASS(module)

        def as_kwargs(self) -> dict[str, typing.Any]:
            "The fields, as keyword arguments to `.attribute`."
            return {f.name: getattr(self, f.name) for f in dcls.fields(self)}

    cfg: Strategy
    "The attr build function."

    batch_size: int = 256
    "The batch size to use to iterate over the dataset."

    def __call__(self, dset: du.Dataset, module: nn.Module) -> torch.Tensor:
        """
        This function corresponds to `captum.attr.*`'s `.attribute` method

        Args:
            dataset: Inputs that will be fed to the `module`.
            module: An `nn.Module`, must output a scalar tenosr.
        """

        adaptor = AttrAdaptor(module)
        attr = self.cfg(adaptor)
        total = 0.0

        n = 0
        for x, y in du.DataLoader(dset, batch_size=self.batch_size):
            tensors, non_tensors = adaptor.split(x)
            result: tuple[torch.Tensor, ...] = attr.attribute(
                tensors,
                additional_forward_args=(non_tensors,),
                target=y,
                **self.kwargs,
            )
            n += len(y)

            for tensor in result:
                total += tensor.abs().sum(0).flatten()

        assert isinstance(total, torch.Tensor)
        return total / n

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return self.cfg.as_kwargs()


# The strategy implementations ====


@build_attr_dcls
class BuildSaliency(CaptumAttrTask.Strategy):
    _CAPTUM_CLASS = attr.Saliency

    abs: bool = False


@build_attr_dcls
class BuildIntegratedGradients(CaptumAttrTask.Strategy):
    _CAPTUM_CLASS = attr.IntegratedGradients

    n_steps: int = 32


@build_attr_dcls
class BuildDeepLift(CaptumAttrTask.Strategy):
    _CAPTUM_CLASS = attr.DeepLift


@build_attr_dcls
class BuildFeatureAblation(CaptumAttrTask.Strategy):
    _CAPTUM_CLASS = attr.FeatureAblation

    perturbations_per_eval: int = 8


@build_attr_dcls
class BuildKernelShap(CaptumAttrTask.Strategy):
    _CAPTUM_CLASS = attr.KernelShap

    n_samples: int = 128
