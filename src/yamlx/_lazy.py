from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["lazy_getattr"]


def lazy_getattr(package: str, exports: Mapping[str, str], name: str) -> object:
    try:
        source = exports[name]
    except KeyError:
        message = f"module {package!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    return getattr(import_module(source, package), name)
