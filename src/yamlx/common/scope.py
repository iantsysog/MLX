from contextlib import contextmanager, suppress
from threading import Lock
from typing import TYPE_CHECKING

__all__ = ["Disposable", "ResourceScope"]

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


class Disposable:
    __slots__ = ("_closer", "_pending", "_scope")

    def __init__(self, scope: ResourceScope, closer: Callable[[], None]) -> None:
        self._scope = scope
        self._closer = closer
        self._pending = True

    def dispose(self) -> None:
        if not self._pending:
            return
        self._pending = False
        self._scope._remove(self)
        self._closer()

    def discard(self) -> None:
        if not self._pending:
            return
        self._pending = False
        self._scope._remove(self)


class ResourceScope:
    __slots__ = ("_closed", "_disposables", "_lock")

    def __init__(self) -> None:
        self._disposables: list[Disposable] = []
        self._closed = False
        self._lock = Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    def defer(self, closer: Callable[[], None]) -> Disposable:
        handle = Disposable(self, closer)
        with self._lock:
            registered = not self._closed
            if registered:
                self._disposables.append(handle)
        if not registered:
            handle._pending = False
            closer()
        return handle

    def _remove(self, handle: Disposable) -> None:
        with self._lock, suppress(ValueError):
            self._disposables.remove(handle)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            handles = self._disposables
            self._disposables = []
            for handle in handles:
                handle._pending = False
        error: BaseException | None = None
        for handle in reversed(handles):
            try:
                handle._closer()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    @contextmanager
    def open(self) -> Generator[ResourceScope]:
        try:
            yield self
        finally:
            self.close()
