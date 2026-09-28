# Copyright (c) AIoWay Authors - All Rights Reserved

"The DAG that supports analysis."

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch

from aioway.t import (
    find_nested_tensors,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

from .vars import VarInfo, VarList

__all__ = ["DoneThunk", "Dag"]


@dcls.dataclass(frozen=True)
class DoneThunk[F: cabc.Callable]:
    """
    Stores the thunk's arguments, function, and output.
    """

    _: dcls.KW_ONLY

    func: F
    "The callable that the thunk calls."

    args: tuple[typing.Any, ...]
    "The arguments fed to the function."

    kwargs: dict[str, typing.Any]
    "The keyword arguments fed to the function."

    result: typing.Any = dcls.MISSING
    "The result. If `dcls.MISSING`, the thunk is not called yet."

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise TypeError(f"{self.func} is not callable.")

    @typing.override
    def __repr__(self) -> str:
        result = str(replace_tensors_with_attr(self.result))
        thunk = render_tensor_func_short(str(self.func), self.args, self.kwargs)
        return thunk + " -> " + result

    def upstream(self) -> cabc.Generator[torch.Tensor]:
        "Get the dependencies of the current thunk."

        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)

    def downstream(self) -> cabc.Generator[torch.Tensor]:
        "Get the output of the current thunk."

        yield from find_nested_tensors(self.result)

    @property
    def done(self) -> bool:
        return self.result is dcls.MISSING


class Dag[F: cabc.Callable](cabc.Sequence[DoneThunk[F]]):
    """
    A dag is a sequence of callables, that are linked by fake tensors.
    """

    def __init__(self, thunks: cabc.Iterable[DoneThunk[F]]) -> None:
        self._thunks = tuple(thunks)
        self._var_list = self._compute_local_vars()
        self._inputs = tuple(self._input_fake_vars())

    def __len__(self) -> int:
        return len(self._thunks)

    @typing.overload
    def __getitem__(self, idx: int) -> DoneThunk[F]: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Self: ...

    def __getitem__(self, idx):
        match idx:
            case slice():
                return self._getitem_slice(idx)
            case int():
                return self._getitem_int(idx)

    def _getitem_slice(self, idx: slice) -> typing.Self:
        return type(self)(self._thunks[idx])

    def _getitem_int(self, idx: int) -> DoneThunk[F]:
        return self._thunks[idx]

    def __iter__(self) -> cabc.Iterator[DoneThunk[F]]:
        return iter(self._thunks)

    def inputs(self) -> tuple[torch.Tensor, ...]:
        "The fake tensor inputs, from order or definition."

        return self._inputs

    def var_list(self) -> VarList:
        return self._var_list

    def _input_fake_vars(self) -> cabc.Generator[torch.Tensor]:
        for var in self._var_list.values():
            if self._var_list[var.fake].is_input:
                yield var.fake

    def _compute_local_vars(self) -> VarList:
        unique_vars: dict[int, VarInfo] = {}

        for idx, thunk in enumerate(self):
            self._add_inputs(idx, thunk, unique_vars)
            self._add_output(idx, thunk, unique_vars)

        return VarList(unique_vars.values())

    def _add_inputs(
        self, idx: int, thunk: DoneThunk, locals: dict[int, VarInfo]
    ) -> None:
        for input in thunk.upstream():
            if (input_id := id(input)) not in locals:
                info = VarInfo.input_var(input)
                locals[input_id] = info

            info = locals[input_id]
            info.add_consumers(idx)

    def _add_output(
        self, idx: int, thunk: DoneThunk, locals: dict[int, VarInfo]
    ) -> None:
        # Output must be unique, so it's always new.
        for output in thunk.downstream():
            info = VarInfo(producer=idx, fake=output)

            if id(info.fake) in locals:
                raise KeyError(f"Output produced is not unique.")

            locals[id(output)] = info
