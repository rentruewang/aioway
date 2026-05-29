# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway.compilers import Builder, builder_dcls
from aioway.tags import AttrTag

__all__ = ["SupervisedAlgo"]


@builder_dcls
class SupervisedAlgo(Builder):
    """
    The supervised learning algorithm. Currently supports 1 input 1 output.
    """

    input_space: AttrTag
    target_space: AttrTag

    @typing.no_type_check
    def __call__(self) -> typing.Any:
        raise NotImplementedError
        # builder = JustLinearBuilder(input_space, outputsapc)
        # dag = just_linear_builder([input_space], [target_space])
        # output_nodes = dag.output_nodes

        # # FIXME: this is a bug.
        # loss_nodes = [MSELoss().apply(node) for node in output_nodes]
        # all_hop_nodes = list(dag)
        # return HopDag.from_list_of_nodes([*all_hop_nodes, *loss_nodes])
