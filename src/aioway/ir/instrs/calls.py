# Copyright (c) AIoWay Authors - All Rights Reserved

"The instructions themselves."

import typing
from collections import abc as cabc

from torch import nn

from aioway.t import (
    BoundArgs,
    GetAttrKey,
    Sign,
    TList,
    find_nested_tensors,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

from .instrs import Instr

__all__ = ["FuncCall", "ModuleCall"]


class FuncCall[F: cabc.Callable](Instr):
    """
    An instruction representing a function call.
    """

    _PY_TREE_KEY = GetAttrKey

    def __init__(
        self,
        *,
        func: F,
        args: tuple,
        kwargs: dict[str, typing.Any],
        result=NotImplemented,
    ) -> None:
        self._func = func
        self._bounded = Sign.from_callable(func).bind(*args, **kwargs)
        self._result = result

        if not callable(self.func):
            raise TypeError(f"{self.func} is not callable.")

    def __eq__(self, other) -> bool:
        """
        `FCall` will not implement `__eq__`.
        """

        return NotImplemented

    @typing.override
    def __repr__(self) -> str:
        def maybe_name(func):
            try:
                return func.__name__
            except AttributeError:
                return repr(func)

        result = str(replace_tensors_with_attr(self.result))
        thunk = render_tensor_func_short(maybe_name(self.func), self.args, self.kwargs)
        return thunk + " -> " + result

    @typing.override
    def _inputs(self) -> TList:
        "Get the (unique) dependencies of the current thunk."
        return TList.deduplicate(find_nested_tensors(self.bounded))

    @typing.override
    def _outputs(self) -> TList:
        "Get the output list of (unique) tensors of the current thunk."
        return TList.deduplicate(find_nested_tensors(self.result))

    @property
    def args(self):
        return self.bounded.args

    @property
    def kwargs(self):
        return self.bounded.kwargs

    @property
    def func(self) -> F:
        "The callable that the thunk calls."
        return self._func

    @property
    def bounded(self) -> BoundArgs:
        "The bounded arguments."
        return self._bounded

    @property
    def sign(self) -> Sign:
        "The signature."
        return self.bounded.sign

    @property
    def result(self) -> typing.Any:
        "The result. If `NotImplemented`, the thunk is not called yet."
        return self._result

    @property
    def done(self) -> bool:
        return self.result is not NotImplemented

    @typing.override
    def _tree_dict(self) -> dict[str, typing.Any]:
        return {"bounded": self.bounded, "result": self.result}

    @property
    @typing.override
    def _tree_context(self) -> typing.Any:
        return self.func

    @classmethod
    def _tree_from_dict(cls, mapping: dict, context) -> typing.Self:
        assert callable(context)

        assert len(mapping) == 2
        assert "bounded" in mapping
        assert "result" in mapping
        assert isinstance(bounded := mapping["bounded"], BoundArgs)

        return cls(
            func=context,
            args=bounded.args,
            kwargs=bounded.kwargs,
            result=mapping["result"],
        )


class ModuleCall(FuncCall[nn.Module]):
    """
    The `nn.Module` version of `FCall` instruction.
    Handles the hierarchy of `nn.Module`s.
    """

    _PY_TREE_KEY = GetAttrKey

    def __init__(
        self,
        *,
        func: nn.Module,
        args: tuple,
        kwargs: dict[str, typing.Any],
        result=NotImplemented,
        parents: tuple[nn.Module, ...] = (),
    ) -> None:
        super().__init__(func=func, args=args, kwargs=kwargs, result=result)
        self._parents = parents

    def __repr__(self) -> str:
        return super().__repr__() + f" [{len(self.parents)} parents]"

    @property
    def parents(self) -> tuple[nn.Module, ...]:
        "The parent modules that calls this current thunk. It's a stack."
        return self._parents

    @property
    @typing.override
    def _tree_context(self) -> typing.Any:
        return self.func, self.parents

    @classmethod
    def _tree_from_dict(cls, mapping: dict, context) -> typing.Self:
        module, parents = context

        assert len(mapping) == 2
        assert "bounded" in mapping
        assert "result" in mapping
        assert isinstance(bounded := mapping["bounded"], BoundArgs)
        assert isinstance(module, nn.Module)

        return cls(
            func=module,
            args=bounded.args,
            kwargs=bounded.kwargs,
            result=mapping["result"],
            parents=parents,
        )
