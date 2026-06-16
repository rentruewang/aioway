# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway.compilers import Emitter, JustLinearEmitter, emitter_dcls
from aioway.hop import ListHop, MSELoss, TensorHop
from aioway.spaces import ShapeSpace

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
        return JustLinearEmitter(
            self.input_data, ShapeSpace(self.target_data.attr.shape)
        )
