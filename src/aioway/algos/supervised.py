# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

from aioway.compilers import Emitter, JustLinearEmitter, emitter_dcls
from aioway.hop import ListHop, TensorHop
from aioway.nn import MSELoss
from aioway.tags import TagDict

__all__ = ["SupervisedAlgo"]


@emitter_dcls
class SupervisedAlgo(Emitter):
    """
    The supervised learning algorithm. Currently supports 1 input 1 output.
    """

    input_data: TensorHop
    target_data: TensorHop

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
        raise NotImplementedError
        # return JustLinearBuilder(
        #     AttrTag.from_attr(self.input_data.attr),
        #     AttrTag.from_attr(self.target_data.attr),
        # )

    def inputs(self) -> cabc.Iterator[TagDict]:
        raise NotImplementedError
        # yield TagDict.from_tags(AttrTag.from_attr(self.input_data.attr))

    def outputs(self) -> cabc.Iterator[TagDict]:
        raise NotImplementedError
        # yield TagDict.from_tags(AttrTag.from_attr(self.target_data.attr))
