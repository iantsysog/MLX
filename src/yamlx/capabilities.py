from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from importlib.metadata import entry_points
from threading import RLock
from typing import TYPE_CHECKING

from yamlx.common.errors import ModelNotCachedError

if TYPE_CHECKING:
    from pathlib import Path

lazy from yamlx.speech.asr.qwen3.runtime.load import resolve_model_path
lazy from yamlx.speech.asr.qwen3.runtime.session import close, load_session

__all__ = [
    "Availability",
    "CapabilitySpec",
    "Dispose",
    "Domain",
    "Requirement",
    "RequirementKind",
    "Task",
    "available",
    "checkpoint_requirement",
    "extra_requirement",
    "get",
    "module_requirement",
    "refresh",
    "register",
]


class Domain(StrEnum):
    LANGUAGE = "language"
    SPATIAL = "spatial"
    SPEECH = "speech"
    VISION = "vision"


class Task(StrEnum):
    ASR = "asr"
    TTS = "tts"
    AUDIO = "audio"


class RequirementKind(StrEnum):
    DEPENDENCY = "dependency"
    CHECKPOINT = "checkpoint"
    EXTRA = "extra"


type Key = tuple[Domain, Task, str]
type Loader = Callable[..., object]
type Closer = Callable[..., None]
type Dispose = Callable[[], None]
type Check = Callable[[], str | None]


@dataclass(frozen=True)
class Requirement:
    kind: RequirementKind
    name: str
    check: Check


def module_requirement(
    module: str,
    *,
    kind: RequirementKind = RequirementKind.DEPENDENCY,
) -> Requirement:
    def check() -> str | None:
        try:
            import_module(module)
        except ImportError:
            return f"missing module '{module}'"
        return None

    return Requirement(kind=kind, name=module, check=check)


def extra_requirement(module: str) -> Requirement:
    return module_requirement(module, kind=RequirementKind.EXTRA)


def checkpoint_requirement(
    source: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> Requirement:
    def check() -> str | None:
        try:
            resolve_model_path(source, cache_dir=cache_dir)
        except ModelNotCachedError as error:
            return str(error)
        return None

    return Requirement(kind=RequirementKind.CHECKPOINT, name=str(source), check=check)


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    domain: Domain
    task: Task
    loader: Loader
    closer: Closer | None = None
    requires: tuple[Requirement, ...] = ()

    @property
    def key(self) -> Key:
        return (self.domain, self.task, self.name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.loader(*args, **kwargs)

    def close(self, capability: object) -> None:
        if self.closer is not None:
            self.closer(capability)


@dataclass(frozen=True)
class Availability:
    spec: CapabilitySpec
    reasons: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.reasons


@dataclass
class _Registry:
    lock: RLock = field(default_factory=RLock)
    specs: dict[Key, CapabilitySpec] = field(default_factory=dict)
    plugins: dict[Key, CapabilitySpec] = field(default_factory=dict)
    builtins_loaded: bool = False
    entry_points_loaded: bool = False


_registry = _Registry()

_BUILTIN_KEY: Key = (Domain.SPEECH, Task.ASR, "qwen3-asr")
_BUILTIN_REQUIRES: tuple[Requirement, ...] = (
    module_requirement("mlx"),
    module_requirement("numpy"),
    module_requirement("regex"),
)


def _conflict(spec: CapabilitySpec, reason: str = "already registered") -> ValueError:
    return ValueError(
        f"Capability '{spec.name}' for {spec.domain}/{spec.task} is {reason}."
    )


def register(spec: CapabilitySpec) -> Dispose:
    key = spec.key
    with _registry.lock:
        existing = _registry.specs.get(key)
        if existing is not None and existing is not spec:
            raise _conflict(spec)
        _registry.specs[key] = spec

    def dispose() -> None:
        with _registry.lock:
            if _registry.specs.get(key) is spec:
                del _registry.specs[key]

    return dispose


def _load_builtins() -> None:
    with _registry.lock:
        if _registry.builtins_loaded:
            return
        _registry.builtins_loaded = True
        if _BUILTIN_KEY in _registry.specs:
            return
        _registry.specs[_BUILTIN_KEY] = CapabilitySpec(
            name="qwen3-asr",
            domain=Domain.SPEECH,
            task=Task.ASR,
            loader=load_session,
            closer=close,
            requires=_BUILTIN_REQUIRES,
        )


def _discover_entry_points() -> list[CapabilitySpec]:
    specs: list[CapabilitySpec] = []
    for entry_point in entry_points(group="yamlx.capabilities"):
        plugin = entry_point.load()
        if isinstance(plugin, CapabilitySpec):
            specs.append(plugin)
    return specs


def _install_plugins_locked(specs: list[CapabilitySpec]) -> None:
    incoming: dict[Key, CapabilitySpec] = {}
    for spec in specs:
        if spec.key in incoming:
            raise _conflict(spec, "registered by more than one entry point")
        incoming[spec.key] = spec
    for key, spec in incoming.items():
        existing = _registry.specs.get(key)
        if existing is not None and _registry.plugins.get(key) is not existing:
            raise _conflict(spec)
    for key in _registry.plugins:
        _registry.specs.pop(key, None)
    _registry.plugins = incoming
    _registry.specs.update(incoming)


def _load_entry_points() -> None:
    with _registry.lock:
        if _registry.entry_points_loaded:
            return
        _install_plugins_locked(_discover_entry_points())
        _registry.entry_points_loaded = True


def refresh() -> None:
    with _registry.lock:
        _install_plugins_locked(_discover_entry_points())
        _registry.entry_points_loaded = True


def _ensure_loaded() -> None:
    _load_builtins()
    _load_entry_points()


def _availability(spec: CapabilitySpec) -> Availability:
    return Availability(
        spec=spec,
        reasons=tuple(
            reason
            for requirement in spec.requires
            if (reason := requirement.check()) is not None
        ),
    )


def get(domain: Domain, task: Task, name: str) -> CapabilitySpec:
    _ensure_loaded()
    key: Key = (domain, task, name)
    with _registry.lock:
        spec = _registry.specs.get(key)
        known = sorted(item.name for item in _registry.specs.values())
    if spec is None:
        message = (
            f"No capability '{name}' registered for {domain}/{task}. Available: {known}"
        )
        raise LookupError(message)
    availability = _availability(spec)
    if availability.reasons:
        message = (
            f"Capability '{name}' for {domain}/{task} is unavailable: "
            + "; ".join(availability.reasons)
        )
        raise LookupError(message)
    return spec


def available(
    *,
    domain: Domain | None = None,
    task: Task | None = None,
) -> frozendict[Key, Availability]:
    _ensure_loaded()
    with _registry.lock:
        items = list(_registry.specs.items())
    return frozendict({
        key: _availability(spec)
        for key, spec in items
        if (domain is None or key[0] is domain) and (task is None or key[1] is task)
    })
