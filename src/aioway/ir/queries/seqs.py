# Copyright (c) AIoWay Authors - All Rights Reserved

"Queries for a chain of instrs `a -> b -> c`, like `nn.Sequential`."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway.ir.instrs import Instr
from aioway.ir.queries import OrderedIndex, Query

if typing.TYPE_CHECKING:
    from aioway.ir import Program

__all__ = ["SequentialQuery", "ExactSequential", "TypeSequential"]


class _SeqMatchFailed(RuntimeError):
    "Our custom error to raise / catch when matching failed."


@dcls.dataclass(frozen=True)
class SequentialQuery[I: Instr = typing.Any](Query[I]):
    """
    Search for a chain in reverse.

    How `Instr` matches the given pattern can be defined by key.

    The chain can sit anywhere in the program, like `re.search` not `re.match`.
    """

    pattern: tuple[typing.Any, ...]
    "What each instr in the chain must match, in order."

    key: cabc.Callable[[I], typing.Any]
    "Gets the value to match from an instr, e.g. `lambda thunk: thunk.func`."

    def __call__(self, program: Program[I], /) -> OrderedIndex:
        # Try every end, walking back from it. The first few steps can't end
        # a chain, as it needs `len(pattern) - 1` steps before its end.
        # A partial match that fails just moves on to the next.

        for end in range(len(self.pattern) - 1, len(program)):
            try:
                chain = self._chain(program, end)
            except _SeqMatchFailed:
                continue
            else:
                return OrderedIndex.from_list_int(chain)

        raise LookupError(f"No match for {self.pattern}.")

    def _chain(self, program: Program[I], end: int) -> list[int]:
        "The chain ending at `end`. Raises `_SeqMatchFailed` if it doesn't match."

        step = end
        reverse_chain: list[int] = []

        for item in reversed(self.pattern):
            # Stop if does not match.
            if not self._match(self.key(program[step]), item):
                raise _SeqMatchFailed

            reverse_chain.append(step)

            if len(reverse_chain) == len(self.pattern):
                break

            step = _prev_in_chain(program, step)

        return reverse_chain[::-1]

    @abc.abstractmethod
    def _match(self, value: typing.Any, pat: typing.Any) -> bool:
        raise NotImplementedError


class ExactSequential[I: Instr = typing.Any](SequentialQuery[I]):
    "Matches with `==`, for functions, e.g. `(torch.relu, torch.sigmoid)`."

    def _match(self, value: typing.Any, pat: typing.Any) -> bool:
        return value == pat


class TypeSequential[I: Instr = typing.Any](SequentialQuery[I]):
    "Matches with `isinstance`, for modules, e.g. `(nn.Linear, nn.ReLU)`."

    def _match(self, value: typing.Any, pat: typing.Any) -> bool:
        return isinstance(value, pat)


def _prev_in_chain[I: Instr](program: Program[I], step: int) -> int:
    """
    The step before `step` in a chain. Raises `_SeqMatchFailed` if there is none.

    `step` must have one input, made by an instr rather than given to the program.
    """

    inputs = program.instrs[step].inputs

    if len(inputs) != 1:
        raise _SeqMatchFailed

    if (prev := program.output_of_step(inputs[0])) >= 0:
        return prev

    # `output_of_step` gives -1 for program inputs.
    raise _SeqMatchFailed
