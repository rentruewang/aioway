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


@dcls.dataclass(frozen=True)
class SequentialQuery[I: Instr = typing.Any](Query[I]):
    """
    Selects the first chain where the n-th instr matches `pattern[n]`,
    and each instr's output is consumed only by the next one.
    """

    pattern: tuple[typing.Any, ...]
    "What each instr in the chain must match, in order."

    key: cabc.Callable[[I], typing.Any]
    "Gets the value to match from an instr, e.g. `lambda i: i.fn`."

    def __call__(self, program: Program[I], /) -> OrderedIndex:
        for start in range(len(program)):
            if chain := self._chain(program, start):
                return OrderedIndex.from_list_int(chain)

        raise LookupError(f"No match for {self.pattern}.")

    def _chain(self, program: Program[I], step: int) -> list[int]:
        "The chain starting at `step`, or empty if it doesn't match."

        chain: list[int] = []

        for pat in self.pattern:
            if chain:
                # Step to the next instr: the only consumer of this one's output.
                (out,) = program[step].outputs
                consumers = program.input_to_step(out)

                if len(consumers) != 1:
                    return []

                (step,) = consumers

            if not self._match(self.key(program.instrs[step]), pat):
                return []

            chain.append(step)

        return chain

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
