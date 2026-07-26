# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for NMF, temporarily."

import typing

import tensordict as td

from aioway.attrs import AttrDict
from aioway.errors import re_raise_func
from aioway.spaces import TdictSpace, TensorSpace, space_dcls


@typing.final
@space_dcls
class NMFSpace(TdictSpace):
    """
    NMF is 1 space duplicated, or 2 spaces of the same type.
    """

    space: TensorSpace
    "The space for both input and target. They should be the same."

    @typing.override
    @re_raise_func(AssertionError, ValueError)
    def _check_attrs(self, attrs: AttrDict) -> None:
        assert len(attrs) == 2

        for attr in attrs.values():
            assert attr.to_fake_tensor() in self.space

    def _check_data(self, data: td.TensorDict):
        pass

    @typing.override
    def _sample_n(self, n: int):
        sample = self.space.sample(n)
        return td.TensorDict({"input": sample, "target": sample})
