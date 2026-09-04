"""Strict TOML configuration loading for the disk cache."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tomllib
from typing import Any, Mapping

from .errors import ConfigurationError


CONFIG_ENV = "COMFYUI_MODEL_DISK_CACHE_CONFIG"
_GIB = 1024**3
_MIB = 1024**2

_CACHE_KEYS = {
    "enabled",
    "root",
    "required_mountpoint",
    "max_size_gib",
    "min_free_gib",
    "min_file_size_mib",
    "eviction",
    "validation",
    "touch_on_hit",
    "fail_open",
    "miss_policy",
    "compatibility_policy",
}
_SOURCE_KEYS = {"name", "root"}


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """One authoritative model-file tree."""

    name: str
    root: Path


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Validated runtime configuration expressed in bytes and absolute paths."""

    enabled: bool
    root: Path
    required_mountpoint: Path
    max_size_bytes: int
    min_free_bytes: int
    min_file_size_bytes: int
    eviction: str
    validation: str
    touch_on_hit: bool
    fail_open: bool
    miss_policy: str
    compatibility_policy: str
    sources: tuple[SourceConfig, ...]


def discover_config_path(user_directory: Path | None = None) -> Path:
    """Return the explicit path, ComfyUI user path, or local fallback path."""

    configured = os.environ.get(CONFIG_ENV)
    if configured:
        return Path(configured).expanduser()

    if user_directory is None:
        try:
            import folder_paths  # type: ignore[import-not-found]

            getter = getattr(folder_paths, "get_user_directory", None)
            if callable(getter):
                user_directory = Path(getter())
        except (ImportError, AttributeError, TypeError):
            user_directory = None

    if user_directory is not None:
        return Path(user_directory) / "model-disk-cache.toml"

    return Path(__file__).resolve().parents[1] / "config.toml"


def load_config(path: str | os.PathLike[str]) -> CacheConfig:
    """Load and validate a disk-cache TOML file."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigurationError(
            f"Configuration file not found: {config_path}"
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"Invalid TOML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read configuration {config_path}: {exc}"
        ) from exc

    unknown_top_level = set(document) - {"cache", "sources"}
    if unknown_top_level:
        raise ConfigurationError(
            f"Unknown top-level configuration keys: {_format_keys(unknown_top_level)}"
        )

    cache_data = _mapping(document.get("cache"), "[cache]")
    source_data = document.get("sources")
    if not isinstance(source_data, list) or not source_data:
        raise ConfigurationError("At least one [[sources]] entry is required")

    unknown_cache = set(cache_data) - _CACHE_KEYS
    if unknown_cache:
        raise ConfigurationError(
            f"Unknown [cache] keys: {_format_keys(unknown_cache)}"
        )

    required_cache = {"root", "required_mountpoint", "max_size_gib", "min_free_gib"}
    missing_cache = required_cache - set(cache_data)
    if missing_cache:
        raise ConfigurationError(
            f"Missing required [cache] keys: {_format_keys(missing_cache)}"
        )

    root = _absolute_path(cache_data["root"], "cache.root")
    mountpoint = _absolute_path(
        cache_data["required_mountpoint"], "cache.required_mountpoint"
    )
    if root == mountpoint or not _is_relative_to(root, mountpoint):
        raise ConfigurationError(
            "cache.root must be a child of cache.required_mountpoint"
        )

    max_size_bytes = _size_bytes(cache_data["max_size_gib"], _GIB, "cache.max_size_gib")
    if max_size_bytes <= 0:
        raise ConfigurationError("cache.max_size_gib must be greater than zero")
    min_free_bytes = _size_bytes(
        cache_data["min_free_gib"], _GIB, "cache.min_free_gib"
    )
    min_file_size_bytes = _size_bytes(
        cache_data.get("min_file_size_mib", 0),
        _MIB,
        "cache.min_file_size_mib",
    )

    sources: list[SourceConfig] = []
    names: set[str] = set()
    roots: set[Path] = set()
    for index, raw_source in enumerate(source_data):
        label = f"sources[{index}]"
        source = _mapping(raw_source, label)
        unknown_source = set(source) - _SOURCE_KEYS
        if unknown_source:
            raise ConfigurationError(
                f"Unknown {label} keys: {_format_keys(unknown_source)}"
            )
        missing_source = _SOURCE_KEYS - set(source)
        if missing_source:
            raise ConfigurationError(
                f"Missing {label} keys: {_format_keys(missing_source)}"
            )

        name = _nonempty_string(source["name"], f"{label}.name")
        source_root = _absolute_path(source["root"], f"{label}.root")
        if name in names:
            raise ConfigurationError(f"Duplicate source name: {name}")
        if source_root in roots:
            raise ConfigurationError(f"Duplicate source root: {source_root}")
        if _paths_overlap(source_root, root):
            raise ConfigurationError(
                f"Source root and cache root must not overlap: {source_root}"
            )
        names.add(name)
        roots.add(source_root)
        sources.append(SourceConfig(name=name, root=source_root))

    return CacheConfig(
        enabled=_boolean(cache_data.get("enabled", True), "cache.enabled"),
        root=root,
        required_mountpoint=mountpoint,
        max_size_bytes=max_size_bytes,
        min_free_bytes=min_free_bytes,
        min_file_size_bytes=min_file_size_bytes,
        eviction=_choice(cache_data.get("eviction", "lru"), {"lru"}, "cache.eviction"),
        validation=_choice(
            cache_data.get("validation", "stat"), {"stat"}, "cache.validation"
        ),
        touch_on_hit=_boolean(
            cache_data.get("touch_on_hit", True), "cache.touch_on_hit"
        ),
        fail_open=_boolean(cache_data.get("fail_open", True), "cache.fail_open"),
        miss_policy=_choice(
            cache_data.get("miss_policy", "copy_then_load"),
            {"copy_then_load"},
            "cache.miss_policy",
        ),
        compatibility_policy=_choice(
            cache_data.get("compatibility_policy", "guarded"),
            {"guarded", "strict"},
            "cache.compatibility_policy",
        ),
        sources=tuple(sources),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a TOML table")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    text = _nonempty_string(value, label)
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ConfigurationError(f"{label} must be an absolute path: {text}")
    return path.resolve(strict=False)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty string")
    return value.strip()


def _size_bytes(value: Any, multiplier: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConfigurationError(f"{label} must be a non-negative finite number")
    return int(number * multiplier)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{label} must be true or false")
    return value


def _choice(value: Any, choices: set[str], label: str) -> str:
    text = _nonempty_string(value, label)
    if text not in choices:
        expected = ", ".join(sorted(choices))
        raise ConfigurationError(f"{label} must be one of: {expected}")
    return text


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_relative_to(first, second) or _is_relative_to(second, first)


def _format_keys(keys: set[str]) -> str:
    return ", ".join(sorted(keys))
