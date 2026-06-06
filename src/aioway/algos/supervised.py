# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

from aioway._streams import TensorStream
from aioway.compilers import Builder, JustLinearBuilder, builder_dcls
from aioway.hop import HopGraph, MSELoss, TensorStreamHop
from aioway.tags import AttrTag, TagDict

__all__ = ["SupervisedAlgo"]


@builder_dcls
class SupervisedAlgo(Builder):
    """
    The supervised learning algorithm. Currently supports 1 input 1 output.
    """

    input_data: TensorStream
    target_data: TensorStream

    @typing.no_type_check
    def __call__(self) -> HopGraph:
        builder = self.just_linear
        dag = builder()

        [output_node] = dag.output_nodes
        stream_node = TensorStreamHop(self.target_data)
        loss_node = MSELoss().apply(output_node, stream_node)

        return HopGraph(loss_node)

    @property
    def just_linear(self) -> JustLinearBuilder:
        return JustLinearBuilder(
            AttrTag.from_attr(self.input_data.attr),
            AttrTag.from_attr(self.target_data.attr),
        )

    def inputs(self) -> cabc.Iterator[TagDict]:
        yield TagDict.from_tags(AttrTag.from_attr(self.input_data.attr))

    def outputs(self) -> cabc.Iterator[TagDict]:
        yield TagDict.from_tags(AttrTag.from_attr(self.target_data.attr))
