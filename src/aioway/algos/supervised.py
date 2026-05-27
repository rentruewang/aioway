# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

from aioway.compilers import just_linear_builder
from aioway.hop import HopDag, MSELoss
from aioway.spaces import Space

__all__ = ["SupervisedAlgo"]


class SupervisedAlgo(abc.ABC):
    """
    The supervised learning algorithm. Currently supports 1 input 1 output.
    """

    def __call__(self, input_space: Space, target_space: Space) -> typing.Any:
        dag = just_linear_builder([input_space], [target_space])
        output_nodes = dag.output_nodes

        # FIXME: this is a bug.
        loss_nodes = [MSELoss().apply(node) for node in output_nodes]
        all_hop_nodes = list(dag)
        return HopDag.from_list_of_nodes([*all_hop_nodes, *loss_nodes])
