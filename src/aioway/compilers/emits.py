# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway.builders import TensorBuilder
from aioway.dsets import Dset, LoaderOpt, TensorStream
from aioway.hop import Linear, ListHop
from aioway.sinks import Sink
from aioway.tags import AttrTag

__all__ = ["Emitter", "emitter_dcls", "JustLinearEmitter"]


@typing.dataclass_transform(frozen_default=True)
def emitter_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@emitter_dcls
class Emitter(abc.ABC):
    @abc.abstractmethod
    def __call__(self) -> ListHop:
        """
        Compiles from a list of input and output tags.
        Returns `NotImplemented` if the current builder does not support the inputs outputs.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def inputs(self) -> cabc.Iterator[Dset]:
        raise NotImplementedError

    @abc.abstractmethod
    def outputs(self) -> cabc.Iterator[Sink]:
        raise NotImplementedError


@emitter_dcls
class JustLinearEmitter(Emitter):
    """
    A builder that outputs 1 linear layer, supporting 1 input and 1 output.
    """

    in_dset: TensorStream
    "The input shape."

    out_dset: Sink
    "The output shape."

    opts: LoaderOpt = LoaderOpt()
    "The default data loader options."

    def __post_init__(self) -> None:
        assert isinstance(self.in_dset, Dset)
        assert isinstance(self.out_dset, Sink)

    def __call__(self) -> ListHop:
        try:
            in_attr = self.in_attr.to_attr()
            out_attr = self.out_attr.to_attr()
        except TypeError:
            return NotImplemented

        input_node = TensorBuilder(self.in_dset(self.opts))
        linear_node = input_node.apply_layer(
            Linear(in_attr.shape[-1], out_attr.shape[-1]),
        )
        return ListHop([linear_node.hop])

    @property
    def in_attr(self) -> AttrTag:
        result = self.in_dset.tags[AttrTag.NAME]
        assert isinstance(result, AttrTag)
        return result

    @property
    def out_attr(self) -> AttrTag:
        result = self.out_dset.tags[AttrTag.NAME]
        assert isinstance(result, AttrTag)
        return result

    def inputs(self) -> cabc.Iterator[Dset]:
        yield self.in_dset

    def outputs(self) -> cabc.Iterator[Sink]:
        yield self.out_dset
