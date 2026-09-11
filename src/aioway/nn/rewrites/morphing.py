# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from .rewrites import Rewriter

__all__ = ["NetMorphDeeper"]


class NetMorphDeeper(Rewriter):
    @typing.override
    def rewrite(self, module: nn.Module) -> nn.Module:
        raise NotImplementedError
