# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

from aioway.compilers import Emitter, JustLinearEmitter, emitter_dcls
from aioway.dsets import Dset, TensorStream
from aioway.hop import ListHop
from aioway.nn import MSELoss
from aioway.sinks import Sink

__all__ = ["SupervisedAlgo"]


@emitter_dcls
class SupervisedAlgo(Emitter):
    """
    The supervised learning algorithm. Currently supports 1 input 1 output.
    """

    input_data: TensorStream
    target_data: Sink

    @typing.no_type_check
    def __call__(self) -> ListHop:
        builder = self.just_linear
        dag = builder()

        # Getting the `ListHop` (which are the outputs)'s immediate dependencies.
        [output_node] = dag.deps()
        loss_node = MSELoss().apply(output_node, self.target_data)

        return ListHop([loss_node])

    @property
    def just_linear(self) -> JustLinearEmitter:
        return JustLinearEmitter(self.input_data, self.target_data)

    def inputs(self) -> cabc.Iterator[Dset]:
        yield self.input_data

    def outputs(self) -> cabc.Iterator[Sink]:
        yield self.target_data
