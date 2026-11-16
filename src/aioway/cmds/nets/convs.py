# Copyright (c) AIoWay Authors - All Rights Reserved

"Convolution emitters."

import functools

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway._utils import is_list_of
from aioway.io.tspecs import TSpec
from aioway.ops import GridOp, OpFunc, SameShapeOp
from aioway.t import fake_mode

from ._utils import Activation, linear_tree, observ_op
from .emitters import Emitter, emitter_dcls

__all__ = ["ImageRegressorEmitter"]


@emitter_dcls
class ImageRegressorEmitter(Emitter):
    """
    The emitter for images -> regression.
    """

    channels: int | list[int]
    "The channel sizes."

    kernels: int | list[int]
    "The kernel sizes."

    strides: int | list[int]
    "The stride sizes."

    activation: str = "relu"
    "The non linear activation to use."

    def _validate(self) -> None:
        _ = self._size

    def __len__(self) -> int:
        return self._size

    def __call__(self, observ: TSpec, action: TSpec) -> OpFunc:
        if not isinstance(observ, tspecs.BoundedContinuous):
            return NotImplemented

        if not isinstance(action, tspecs.Unbounded):
            return NotImplemented

        assert observ.ndim == 3, f"Observation should be 3D. Got {observ.shape=}."

        activation = Activation(self.activation).nn_type

        out_channels = self._as_list(self.channels)
        in_channels = [observ.shape[0], *out_channels[:-1]]
        kernels = self._as_list(self.kernels)
        strides = self._as_list(self.strides)

        op = observ_op()

        for i in range(len(self)):
            conv = nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=kernels[i],
                stride=strides[i],
            )
            op = SameShapeOp(activation(), GridOp(conv, op))

        # Run a fake sample through the convs, for their output shape.
        with fake_mode():
            sim_out = OpFunc(op)(torch.zeros(1, *observ.shape))

        # Flattens, then a linear final layer.
        return OpFunc(linear_tree(op, sim_out.shape[1:], action.shape))

    @functools.cached_property
    def _size(self) -> int:

        len_list = [
            self._get_len(s)
            for s in [
                self.channels,
                self.kernels,
                self.strides,
            ]
        ]
        len_set = {l for l in len_list if l > 0}

        if len(len_set) != 1:
            raise ValueError("Unequal sizes.")

        return list(len_set)[0]

    @staticmethod
    def _get_len(item) -> int:
        if isinstance(item, int):
            return 0
        if is_list_of(int)(item):
            return len(item)
        raise ValueError

    def _as_list(self, item: int | list[int]) -> list[int]:
        if isinstance(item, list):
            assert len(item) == self._size
            return item

        if isinstance(item, int):
            return [item] * self._size

        raise RuntimeError("Unreachable.")
