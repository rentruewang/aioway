# Copyright (c) AIoWay Authors - All Rights Reserved

"Convolution emitters."

import functools

from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway._utils import is_list_of
from aioway.nets import sample_from_space
from aioway.schemas import Shape
from aioway.tspecs import Space, TSpecSpace, unbounded_box_space

from ._utils import Activation, activation_module
from .emitters import Emitter, emitter_dcls
from .linear import linear_regression

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

    activation: Activation = "relu"

    def _validate(self) -> None:
        _ = self._size

    def __len__(self) -> int:
        return self._size

    def __call__(self, observ: Space, action: Space) -> nn.Sequential:
        if not isinstance(observ, TSpecSpace):
            return NotImplemented
        if not isinstance(action, TSpecSpace):
            return NotImplemented

        if not observ.cast_spec(tspecs.BoundedContinuous):
            return NotImplemented

        if not action.cast_spec(tspecs.Unbounded):
            return NotImplemented

        activation = activation_module(self.activation)

        modules: list[nn.Module] = []

        channels = self._as_list(self.channels)
        kernels = self._as_list(self.kernels)
        strides = self._as_list(self.strides)
        for i in range(len(self)):
            modules.append(
                nn.LazyConv2d(
                    out_channels=channels[i],
                    kernel_size=kernels[i],
                    stride=strides[i],
                )
            )

            if activation is not NotImplemented:
                modules.append(activation)

        seq = nn.Sequential(*modules, nn.Flatten())

        sim_in = sample_from_space(observ)
        sim_out = seq(sim_in)

        # Emits a linear final layer, that uses our `linear_shape` logic.
        linear = linear_regression(
            unbounded_box_space(shape=Shape.parse(sim_out.shape[1:])), action
        )

        seq.append(linear)

        return seq

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
