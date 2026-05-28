# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import abc
import dataclasses as dcls
import re
import typing

import tensordict as td
import torch

from aioway.attrs import Attr, AttrDict

__all__ = ["Tag", "TensorTag", "tags_dcls", "attach_tags", "extract_tags"]

_TAG_NAME = re.compile(r"^__aioway_[a-zA-Z0-9_]+__$")


@typing.dataclass_transform(frozen_default=True)
def tags_dcls(cls):
    return dcls.dataclass(frozen=True, slots=True)(cls)


@tags_dcls
class Tag[T = object](abc.ABC):
    """
    The base class for tags. A tag describes a `torch.Tensor`,
    and is piggybacked onto the tensor to pass around `torch` APIs.

    It is also a filtering system.

    Each tag type (subclass) defines a tag name `cls.TAG`,
    corresponding to the attribute name it is set on the tensor.

    This means each tensor would have 1 tag of the same type,
    multiple tags would have multiple types. Therefore, an instance of tag
    is expected to fully describe a tensor in that tag's expertise.
    For example, a tag about batch dimensino should describe all dimensions at once,
    because that tag type can only have a singleton on each tensor.
    """

    NAME: typing.ClassVar[str]
    """
    The name of the tag. Must be of the format `__aioway_*__`.
    """

    def __init_subclass__(cls) -> None:
        if not _TAG_NAME.fullmatch(cls.NAME):
            raise ValueError(
                f"Tag name should be of the format `__aioway_*__`. Got '{cls.NAME}'."
            )

    @typing.final
    def __post_init__(self):
        # Validate `self`. The reason this is marked `@typing.final` is
        # because it can be easy to forget to call `super().__post_init__()`.

        self._check_self()

    def _check_self(self) -> None:
        """
        Validate the data of the tags on its own (not in relation to the `torch.Tensor`).
        """

    @abc.abstractmethod
    def check(self, value: T, /) -> bool:
        """
        Perform some checks on the value you are going to attach on.
        If the tests pass, return `True`, else return `False`.
        """

        raise NotImplementedError

    def attach(self, item: T, /, overwrite: bool = True) -> None:
        """
        Validate then set `self` on the `item`.

        If `overwrite` is `False` (defaults to `True`), and the tag already exists,
        raise a `AttributeError` and do not set the attribute.
        """

        self.check(item)

        if not overwrite and hasattr(item, self.NAME):
            raise AttributeError(f"`{self.NAME}` already exists on {item=}.")

        setattr(item, self.NAME, self)

    @classmethod
    def extract(cls, tensor: T) -> typing.Self | None:
        """
        Get the tag currently stored on the `tensor`.
        """

        if (tag := getattr(tensor, cls.NAME, None)) is None:
            return None

        if not isinstance(tag, cls):
            raise TypeError(f"The tag {tag} is of type {type(tag)}, expected {cls}.")

        return tag


@tags_dcls
class TensorTag(Tag[torch.Tensor]):
    "A `Tag` that tags itself onto a `torch.Tensor`."

    @typing.override
    def check(self, value: torch.Tensor, /) -> bool:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{type(value)} is not a `torch.Tensor`.")

        attr = Attr.parse(value)

        try:
            self._check_attr(attr)
            self._check_data(value)
        except ValueError:
            return False
        else:
            return True

    def _check_attr(self, attr: Attr, /) -> None:
        """
        Raise `ValueError` if `self` cannot attach to tensor with `attr`.
        """

    def _check_data(self, tensor: torch.Tensor, /) -> None:
        """
        Raise `ValueError` if `self` is not valid or cannot attach to `tensor`.
        """


@tags_dcls
class TdictTag(Tag[td.TensorDict]):
    "A `Tag` that tags `td.TensorDict`."

    @typing.override
    def check(self, value: td.TensorDict, /) -> bool:
        if not isinstance(value, td.TensorDict):
            raise TypeError(f"{type(value)} is not a `td.TensorDict`.")

        attrs = AttrDict.parse(value)

        try:
            self._check_attrs(attrs)
            self._check_data(value)
        except ValueError:
            return False
        else:
            return True

    def _check_attrs(self, attrs: AttrDict, /) -> None:
        """
        Raise `ValueError` if `self` cannot attach to tdict with `attrs`.
        """

    def _check_data(self, tdict: td.TensorDict, /) -> None:
        """
        Raise `ValueError` if `self` is not valid or cannot attach to `tdict`.
        """


def attach_tags[T](item: T, *tags: Tag[T]) -> None:
    for tag in tags:
        tag.attach(item)


def extract_tags[T](tensor: T, /) -> dict[str, Tag[T]]:
    """
    Extract all the tags on the `torch.Tensor`.

    This assumes that no one else uses the `__aioway_*__` namespace. Hopefully.
    """

    tag_names = [attr_name for attr_name in dir(tensor) if _TAG_NAME.match(attr_name)]
    return {name: _get_tag(tensor, name) for name in tag_names}


def _get_tag[T](item: T, tag_name: str, /) -> Tag[T]:
    if isinstance(attr := getattr(item, tag_name, None), Tag):
        return attr

    raise AttributeError(
        f"The attribute '{tag_name}' is either not found or not a tag."
    )
