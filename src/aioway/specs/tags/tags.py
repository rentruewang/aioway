# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import abc
import dataclasses as dcls
import types
import typing
from collections import abc as cabc

import torch

__all__ = ["Tag", "tags_dcls", "TagsDict"]


@typing.dataclass_transform(frozen_default=True)
def tags_dcls(cls):
    return dcls.dataclass(frozen=True, slots=True)(cls)


@tags_dcls
class Tag(abc.ABC):
    """
    The base class for tags. A tag describes a `torch.Tensor`,
    and is piggybacked onto the tensor to pass around `torch` APIs.

    Each tag type (subclass) defines a tag name `cls.TAG`,
    corresponding to the attribute name it is set on the tensor.

    This means each tensor would have 1 tag of the same type,
    multiple tags would have multiple types. Therefore, an instance of tag
    is expected to fully describe a tensor in that tag's expertise.
    For example, a tag about batch dimensino should describe all dimensions at once,
    because that tag type can only have a singleton on each tensor.
    """

    TAG: typing.ClassVar[str]
    """
    The name of the tag. Must be of the format `__aioway_*__`.
    """

    @abc.abstractmethod
    def _validate(self, tensor: torch.Tensor) -> None:
        """
        Execute additional validation for subclasses. Subclasses can override this,
        and raise an error if `self` is not valid or cannot attach to `tensor`.
        """

    def attach(self, tensor: torch.Tensor, /, overwrite: bool = True) -> None:
        """
        Tag the instance on another tensor. Validate then set `self` on the `tensor`.

        If `overwrite` is `False` (defaults to `True`), and the tag already exists,
        raise a `ValueError` and do not set the attribute.
        """

        if (tags_dict := TagsDict.extract(tensor)) is None:
            raise AttributeError("You must set ")

        if not overwrite and hasattr(tensor, self.TAG):
            raise ValueError(f"`{self.TAG}` already exists on {tensor=}.")

        self._validate(tensor)
        setattr(tensor, self.TAG, self)

    @classmethod
    def extract(cls, tensor: torch.Tensor) -> typing.Self | None:
        """
        Get the tag currently stored on the `tensor`.
        """

        if (tag_dict := TagsDict.extract(tensor)) is None:
            return None

        if (result := tag_dict.get(cls)) is None:
            return None

        if not isinstance(result, cls):
            raise AssertionError(f"Wrong tag extracted {result}!")

        return result


@typing.final
class TagsDict(cabc.MutableMapping[type[Tag], Tag]):
    """
    `TagsDict` is the dict storing all the tags, attached to a `torch.Tensor`.
    """

    __FIELD: typing.ClassVar[str] = "__aioway_tags__"

    def __init__(
        self, mapping: cabc.Mapping[type[Tag], Tag] = types.MappingProxyType({}), /
    ):
        self.__mapping: dict[type[Tag], Tag] = dict(mapping)
        """
        The underlying mapping storing the data.
        """

    @typing.override
    def __repr__(self) -> str:
        return repr(self.__mapping)

    @typing.override
    def __len__(self) -> int:
        return len(self.__mapping)

    @typing.override
    def __getitem__(self, key: type[Tag], /) -> Tag:
        self.__check_key(key)
        val = self.__mapping[key]
        self.__check_val(val)
        return val

    @typing.override
    def __setitem__(self, key: type[Tag], val: Tag, /) -> None:
        self.__check_key(key)
        self.__check_val(val)

        self.__mapping[key] = val

    @typing.override
    def __delitem__(self, key: type[Tag], /) -> None:
        self.__check_key(key)
        del self.__mapping[key]

    @typing.override
    def __iter__(self) -> cabc.Iterator[type[Tag]]:
        yield from self.__mapping

    @typing.override
    def __contains__(self, key: object, /) -> bool:
        return key in self.__mapping

    def __check_key(self, key: type[Tag]):
        if not isinstance(key, type) and issubclass(key, Tag):
            raise KeyError(f"The {key=} should be a subtype of `Tag`.")

    def __check_val(self, val: Tag):
        if not isinstance(val, Tag):
            raise ValueError(f"The value extracted: {val} is not a `Tag`.")

    def attach(self, tensor: torch.Tensor, *, overwrite: bool = False):
        """
        Attach `self` onto the `tensor`.
        If not `overwrite`, raise an error if tag already exists.
        """

        if not overwrite and hasattr(tensor, self.__FIELD):
            raise AttributeError(f"`{self.__FIELD}` already exists!")

        setattr(tensor, self.__FIELD, self)

    @classmethod
    def extract(cls, tensor: torch.Tensor) -> typing.Self | None:
        """
        Extract the tag dict if exists, or else `None`.
        """

        return getattr(tensor, cls.__FIELD, None)
