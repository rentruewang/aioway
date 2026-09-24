# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking and creating thunks."

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import torch

from aioway._utils import Sign
from aioway.torch import (
    find_nested_tensors,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

from .vars import LocalVars, VarInfo

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

        # Try binding, if not this would fail with `TypeError`.
        _ = self._signature.bind(*self.args, **self.kwargs)

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

    @functools.cached_property
    def _signature(self) -> Sign:
        return Sign.from_callable(self.func)


class Dag[F: cabc.Callable = typing.Any](cabc.Sequence[DoneThunk[F]]):
    """
    This is the DAG responsible for executing a traced thunk list on real data.
    """

    def __init__(self, thunks: cabc.Sequence[DoneThunk[F]]) -> None:
        if not thunks:
            raise ValueError("DAG is empty.")

        self._thunks = tuple(thunks)
        self._locals = self._compute_local_vars()
        self._inputs = tuple(self._input_vars())

    def __len__(self) -> int:
        return len(self._thunks)

    @typing.overload
    def __getitem__(self, idx: int) -> DoneThunk[F]: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Self: ...

    def __getitem__(self, idx):
        match idx:
            case int():
                return self._thunks[idx]
            case slice():
                return type(self)(self._thunks[idx])

    def __iter__(self) -> cabc.Generator[DoneThunk[F]]:
        yield from self._thunks

    def __call__(self, *inputs: torch.Tensor) -> typing.Any:
        try:
            self._locals.update(self._inputs, inputs)
        except ValueError as err:
            raise TypeError from err

        for idx, thunk in enumerate(self._thunks):
            args, kwargs = self._locals.map([thunk.args, thunk.kwargs])
            real = thunk.func(*args, **kwargs)
            self._locals.update(thunk.result, real)
            self._locals.expire(idx)

        result = self._locals.map(self._thunks[-1].result)
        self._locals.clear()
        return result

    def inputs(self) -> tuple[torch.Tensor, ...]:
        "The fake tensor inputs, from order or definition."

        return self._inputs

    def _input_vars(self) -> cabc.Generator[torch.Tensor]:
        for var in self._locals.tracked():
            if self._locals.info(var).is_input:
                yield var

    def _compute_local_vars(self) -> LocalVars:
        unique_vars: dict[int, VarInfo] = {}

        for idx, thunk in enumerate(self._thunks):
            self._add_inputs(idx, thunk, unique_vars)
            self._add_output(idx, thunk, unique_vars)

        return LocalVars(unique_vars)

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
