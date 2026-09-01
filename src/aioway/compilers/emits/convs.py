# Copyright (c) AIoWay Authors - All Rights Reserved

"Convolution emitters."

import functools

from torchrl.data import tensor_specs as tspecs

from aioway._utils import is_list_of
from aioway.instrs import Activation, Conv2d, Flatten, Instr, Sequential
from aioway.schemas import Shape
from aioway.tspecs import TSpec, sample_from_tspec, unbounded_box_tspec

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

    activation: str = "relu"
    "The non linear activation to use."

    def _validate(self) -> None:
        _ = self._size

    def __len__(self) -> int:
        return self._size

    def __call__(self, observ: TSpec, action: TSpec) -> Sequential:
        if not isinstance(observ, tspecs.BoundedContinuous):
            return NotImplemented

        if not isinstance(action, tspecs.Unbounded):
            return NotImplemented

        assert observ.ndim == 3

        activation = Activation(self.activation)

        modules: list[Instr] = []

        def sequential():
            "The sequential function. Defined s.t. it updates with `modules`."
            return Sequential(*modules)

        out_channels = self._as_list(self.channels)
        in_channels = [observ.shape[0], *out_channels[:-1]]
        kernels = self._as_list(self.kernels)
        strides = self._as_list(self.strides)

        for i in range(len(self)):
            modules.append(
                Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernels[i],
                    stride=strides[i],
                )
            )

            if activation is not NotImplemented:
                modules.append(activation.instr_cls())

        modules.append(Flatten())

        sim_in = sample_from_tspec(observ)
        sim_out = sequential().module()(sim_in)

        # Emits a linear final layer, that uses our `linear_regression` logic.
        linear = linear_regression(
            unbounded_box_tspec(shape=Shape.parse(sim_out.shape[1:])), action
        )
        modules.append(linear)

        return sequential()

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
