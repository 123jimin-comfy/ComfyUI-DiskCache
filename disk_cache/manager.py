"""On-demand, byte-bounded local file cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BufferedReader, BufferedWriter
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import time
from types import TracebackType
from typing import Callable, NamedTuple, Protocol
import uuid

from .config import CacheConfig, SourceConfig
from .errors import CacheCapacityError, CacheUnavailableError, SourceChangedError
from .index import CacheEntry, CacheIndex
from .locking import InterProcessLock


_LOG = logging.getLogger("comfyui_disk_cache")
_COPY_BUFFER_SIZE = 8 * 1024 * 1024
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9_-]{1,32}$")


class DiskUsage(Protocol):
    free: int


class SourceSnapshot(NamedTuple):
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "SourceSnapshot":
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True, slots=True)
class CacheResolution:
    """The path selected for one loader invocation."""

    source_path: str
    path: str
    key: str | None
    outcome: str

    @property
    def cached(self) -> bool:
        return self.key is not None


@dataclass(frozen=True, slots=True)
class CacheStats:
    entries: int
    indexed_bytes: int
    free_bytes: int
    max_size_bytes: int
    min_free_bytes: int


class CacheLease:
    """Keep a cache object locked until its loader call completes."""

    def __init__(
        self,
        manager: "CacheManager",
        resolution: CacheResolution,
        lock: InterProcessLock | None = None,
    ) -> None:
        self.manager = manager
        self.resolution = resolution
        self._lock = lock
        self._successful = False

    @property
    def path(self) -> str:
        return self.resolution.path

    @property
    def outcome(self) -> str:
        return self.resolution.outcome

    def mark_success(self) -> None:
        self._successful = True

    def __enter__(self) -> "CacheLease":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._successful and self.resolution.key is not None:
                self.manager._record_access(self.resolution)
        finally:
            if self._lock is not None:
                self._lock.release()
                self._lock = None


class CacheManager:
    """Materialize authoritative files into a disposable local cache."""

    def __init__(
        self,
        config: CacheConfig,
        *,
        clock: Callable[[], int] = time.time_ns,
        mount_probe: Callable[[Path], bool] | None = None,
        disk_usage: Callable[[Path], DiskUsage] = shutil.disk_usage,
        copy_file: Callable[
            [BufferedReader, BufferedWriter, int], None
        ] = shutil.copyfileobj,
    ) -> None:
        self.config = config
        self._clock = clock
        self._mount_probe = mount_probe or (lambda path: os.path.ismount(path))
        self._disk_usage = disk_usage
        self._copy_file = copy_file
        self._source_roots: tuple[tuple[SourceConfig, Path], ...] = ()
        self._device: int | None = None
        self._started = False
        self._objects = config.root / "objects"
        self._temporary = config.root / "tmp"
        self._locks = config.root / "locks"
        self._index = CacheIndex(config.root / "index.sqlite3")

    def start(self) -> None:
        """Validate storage and create cache-owned state."""

        if self._started:
            return
        source_roots = self._resolve_sources()
        mountpoint, expected_device = self._validate_storage_before_create()

        self.config.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for directory in (self._objects, self._temporary, self._locks):
            directory.mkdir(mode=0o700, exist_ok=True)
        self._validate_owned_directory(self.config.root, expected_device)
        for directory in (self._objects, self._temporary, self._locks):
            self._validate_owned_directory(directory, expected_device)

        self._source_roots = source_roots
        self._device = expected_device
        self._validate_state_file(self._index.path, expected_device, required=False)
        self._index.initialize()
        self._validate_state_file(self._index.path, expected_device, required=True)
        with self._maintenance_lock():
            self._reconcile()
        self._started = True
        _LOG.info(
            "Disk cache ready: root=%s mount=%s max=%d bytes min_free=%d bytes",
            self.config.root,
            mountpoint,
            self.config.max_size_bytes,
            self.config.min_free_bytes,
        )

    def acquire(self, path: str | os.PathLike[str]) -> CacheLease:
        """Return a lease containing either a cached or authoritative path."""

        if not self._started:
            raise CacheUnavailableError("Cache manager has not been started")
        self._validate_active_storage()

        original = os.fspath(path)
        try:
            canonical = Path(original).resolve(strict=True)
            source = self._match_source(canonical)
            source_stat = canonical.stat()
        except (FileNotFoundError, OSError, RuntimeError):
            return self._bypass(original)

        if source is None or not stat.S_ISREG(source_stat.st_mode):
            return self._bypass(original)
        if source_stat.st_size < self.config.min_file_size_bytes:
            return self._bypass(original)

        snapshot = SourceSnapshot.from_stat(source_stat)
        suffix = _cache_suffix(canonical)
        key = _cache_key(source.name, canonical, suffix)

        object_lock = self._object_lock(key)
        object_lock_held = False
        try:
            object_lock.acquire()
            object_lock_held = True
            entry = self._index.get(key)
            if entry is not None and self._entry_is_valid(entry, snapshot, suffix):
                path_text = str(self._object_path(key, suffix))
                _LOG.info("HIT %s -> %s", canonical, path_text)
                lease = CacheLease(
                    self,
                    CacheResolution(str(canonical), path_text, key, "hit"),
                    object_lock,
                )
                object_lock_held = False
                return lease
        finally:
            if object_lock_held:
                object_lock.release()

        with self._maintenance_lock():
            object_lock = self._object_lock(key)
            object_lock_held = False
            try:
                object_lock.acquire()
                object_lock_held = True
                self._validate_active_storage()
                current_canonical = Path(original).resolve(strict=True)
                current_source = self._match_source(current_canonical)
                if current_canonical != canonical or current_source != source:
                    return self._bypass(original)
                canonical = current_canonical
                current_stat = canonical.stat()
                snapshot = SourceSnapshot.from_stat(current_stat)
                if not stat.S_ISREG(current_stat.st_mode):
                    return self._bypass(original)

                entry = self._index.get(key)
                if entry is not None and self._entry_is_valid(entry, snapshot, suffix):
                    path_text = str(self._object_path(key, suffix))
                    _LOG.info("HIT %s -> %s", canonical, path_text)
                    lease = CacheLease(
                        self,
                        CacheResolution(str(canonical), path_text, key, "hit"),
                        object_lock,
                    )
                    object_lock_held = False
                    return lease

                if entry is not None:
                    self._remove_entry(entry)
                self._ensure_capacity(snapshot.size, excluded_key=key)
                cached_path = self._populate(source, canonical, snapshot, key, suffix)
                _LOG.info("READY %s -> %s", canonical, cached_path)
                lease = CacheLease(
                    self,
                    CacheResolution(str(canonical), str(cached_path), key, "miss"),
                    object_lock,
                )
                object_lock_held = False
                return lease
            finally:
                if object_lock_held:
                    object_lock.release()

    def stats(self) -> CacheStats:
        self._require_started()
        self._validate_active_storage()
        usage = self._disk_usage(self.config.root)
        return CacheStats(
            entries=self._index.count(),
            indexed_bytes=self._index.total_size(),
            free_bytes=usage.free,
            max_size_bytes=self.config.max_size_bytes,
            min_free_bytes=self.config.min_free_bytes,
        )

    def prune(self) -> CacheStats:
        """Evict LRU entries until the configured limits are satisfied."""

        self._require_started()
        with self._maintenance_lock():
            self._validate_active_storage()
            self._ensure_capacity(0, excluded_key=None)
        return self.stats()

    def clear(self) -> None:
        """Remove all indexed objects, leaving lock files and schema intact."""

        self._require_started()
        with self._maintenance_lock():
            self._validate_active_storage()
            entries = self._index.lru_entries()
            locked: list[InterProcessLock] = []
            try:
                for entry in entries:
                    object_lock = self._object_lock(entry.key)
                    if not object_lock.acquire(blocking=False):
                        raise CacheUnavailableError(
                            f"Cannot clear busy cache object: {entry.key}"
                        )
                    locked.append(object_lock)
                for entry in entries:
                    self._remove_entry(entry)
            finally:
                for object_lock in reversed(locked):
                    object_lock.release()

    def _record_access(self, resolution: CacheResolution) -> None:
        try:
            self._validate_active_storage()
            now = self._clock()
            self._index.touch(resolution.key or "", now)
            if self.config.touch_on_hit:
                path = Path(resolution.path)
                if path.is_file() and not path.is_symlink():
                    os.utime(path, ns=(now, now))
        except Exception as exc:
            _LOG.warning(
                "Could not record cache access for %s: %s", resolution.path, exc
            )

    def _populate(
        self,
        source: SourceConfig,
        canonical: Path,
        expected: SourceSnapshot,
        key: str,
        suffix: str,
    ) -> Path:
        destination = self._object_path(key, suffix)
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        self._validate_object_directory(destination.parent)
        temporary = self._temporary / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        started = time.monotonic()
        _LOG.info("MISS %s; copying %d bytes", canonical, expected.size)

        try:
            with (
                canonical.open("rb") as source_handle,
                temporary.open("xb") as target_handle,
            ):
                source_value = os.fstat(source_handle.fileno())
                target_value = os.fstat(target_handle.fileno())
                opened = SourceSnapshot.from_stat(source_value)
                if opened != expected or not stat.S_ISREG(source_value.st_mode):
                    raise SourceChangedError(f"Source changed before copy: {canonical}")
                if (
                    self._device is None
                    or target_value.st_dev != self._device
                    or not stat.S_ISREG(target_value.st_mode)
                ):
                    raise CacheUnavailableError(
                        f"Temporary cache file is on the wrong device: {temporary}"
                    )
                self._copy_file(source_handle, target_handle, _COPY_BUFFER_SIZE)
                target_handle.flush()
                copied = SourceSnapshot.from_stat(os.fstat(target_handle.fileno()))
                after_copy = SourceSnapshot.from_stat(os.fstat(source_handle.fileno()))

            pathname_after = SourceSnapshot.from_stat(canonical.stat())
            if after_copy != opened or pathname_after != opened:
                raise SourceChangedError(f"Source changed while copying: {canonical}")
            if copied.size != opened.size:
                raise SourceChangedError(
                    f"Incomplete cache copy for {canonical}: "
                    f"{copied.size}/{opened.size} bytes"
                )

            os.replace(temporary, destination)
            now = self._clock()
            try:
                self._index.upsert(
                    CacheEntry(
                        key=key,
                        namespace=source.name,
                        source_path=str(canonical),
                        source_dev=opened.device,
                        source_ino=opened.inode,
                        source_size=opened.size,
                        source_mtime_ns=opened.mtime_ns,
                        cache_suffix=suffix,
                        cached_size=opened.size,
                        created_ns=now,
                        last_access_ns=now,
                    )
                )
            except BaseException:
                try:
                    destination.unlink(missing_ok=True)
                except OSError as exc:
                    _LOG.warning(
                        "Could not remove unindexed cache file %s: %s",
                        destination,
                        exc,
                    )
                raise
            _LOG.info(
                "Copied %s in %.2f seconds", canonical, time.monotonic() - started
            )
            return destination
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                _LOG.warning(
                    "Could not remove temporary cache file %s: %s", temporary, exc
                )

    def _ensure_capacity(self, incoming_size: int, excluded_key: str | None) -> None:
        if incoming_size > self.config.max_size_bytes:
            raise CacheCapacityError(
                f"File is larger than the cache limit: {incoming_size} > "
                f"{self.config.max_size_bytes} bytes"
            )

        while True:
            indexed = self._index.total_size()
            actual_free = self._disk_usage(self.config.root).free
            required_for_limit = max(
                0, indexed + incoming_size - self.config.max_size_bytes
            )
            required_for_free = max(
                0, incoming_size + self.config.min_free_bytes - actual_free
            )
            required = max(required_for_limit, required_for_free)
            if required == 0:
                return

            selected: list[tuple[CacheEntry, InterProcessLock]] = []
            selected_bytes = 0
            for entry in self._index.lru_entries(excluded_key):
                lock = self._object_lock(entry.key)
                if not lock.acquire(blocking=False):
                    continue
                selected.append((entry, lock))
                selected_bytes += entry.cached_size
                if selected_bytes >= required:
                    break

            if selected_bytes < required:
                for _, lock in reversed(selected):
                    lock.release()
                raise CacheCapacityError(
                    f"Cannot free {required} bytes; "
                    "cache entries are busy or insufficient"
                )

            try:
                for entry, _ in selected:
                    self._remove_entry(entry)
                    _LOG.info(
                        "EVICT %s (%d bytes)", entry.source_path, entry.cached_size
                    )
            finally:
                for _, lock in reversed(selected):
                    lock.release()

    def _entry_is_valid(
        self, entry: CacheEntry, source: SourceSnapshot, suffix: str
    ) -> bool:
        if entry.cache_suffix != suffix:
            return False
        if (
            entry.source_dev,
            entry.source_ino,
            entry.source_size,
            entry.source_mtime_ns,
        ) != source:
            return False
        path = self._object_path(entry.key, entry.cache_suffix)
        try:
            self._validate_object_directory(path.parent)
            cached_stat = path.lstat()
        except (CacheUnavailableError, OSError):
            return False
        return (
            stat.S_ISREG(cached_stat.st_mode)
            and cached_stat.st_size == entry.cached_size
        )

    def _remove_entry(self, entry: CacheEntry) -> None:
        path = self._object_path(entry.key, entry.cache_suffix)
        try:
            self._validate_object_directory(path.parent)
        except FileNotFoundError:
            self._index.delete(entry.key)
            return
        path.unlink(missing_ok=True)
        self._index.delete(entry.key)

    def _reconcile(self) -> None:
        for child in self._temporary.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink(missing_ok=True)

        expected_paths: set[Path] = set()
        for entry in self._index.entries():
            try:
                path = self._object_path(entry.key, entry.cache_suffix)
            except CacheUnavailableError as exc:
                _LOG.warning("Dropping unsafe cache-index entry %r: %s", entry.key, exc)
                self._index.delete(entry.key)
                continue
            object_lock = self._object_lock(entry.key)
            if not object_lock.acquire(blocking=False):
                expected_paths.add(path)
                continue
            try:
                try:
                    source_path = Path(entry.source_path).resolve(strict=True)
                    source = self._match_source(source_path)
                    source_snapshot = SourceSnapshot.from_stat(source_path.stat())
                except (OSError, RuntimeError):
                    source = None
                    source_snapshot = None
                if (
                    source is None
                    or source.name != entry.namespace
                    or source_snapshot is None
                    or not self._entry_is_valid(
                        entry, source_snapshot, entry.cache_suffix
                    )
                ):
                    self._remove_entry(entry)
                else:
                    expected_paths.add(path)
            finally:
                object_lock.release()

        for path in self._objects.rglob("*"):
            if (path.is_file() or path.is_symlink()) and path not in expected_paths:
                path.unlink(missing_ok=True)
        for directory in sorted(
            (path for path in self._objects.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass

    def _match_source(self, path: Path) -> SourceConfig | None:
        for source, root in self._source_roots:
            try:
                path.relative_to(root)
                return source
            except ValueError:
                continue
        return None

    def _resolve_sources(self) -> tuple[tuple[SourceConfig, Path], ...]:
        resolved: list[tuple[SourceConfig, Path]] = []
        for source in self.config.sources:
            try:
                root = source.root.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise CacheUnavailableError(
                    f"Cannot resolve configured source "
                    f"{source.name}={source.root}: {exc}"
                ) from exc
            if not root.is_dir():
                raise CacheUnavailableError(
                    f"Configured source is not a directory: {source.name}={root}"
                )
            resolved.append((source, root))
        return tuple(
            sorted(resolved, key=lambda item: len(item[1].parts), reverse=True)
        )

    def _object_path(self, key: str, suffix: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise CacheUnavailableError(f"Invalid cache key in index: {key!r}")
        if suffix and not _SAFE_SUFFIX.fullmatch(suffix):
            raise CacheUnavailableError(f"Invalid cache suffix in index: {suffix!r}")
        return self._objects / key[:2] / f"{key}{suffix}"

    def _maintenance_lock(self) -> InterProcessLock:
        return InterProcessLock(self._locks / "maintenance.lock")

    def _object_lock(self, key: str) -> InterProcessLock:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise CacheUnavailableError(f"Invalid cache key in index: {key!r}")
        return InterProcessLock(self._locks / f"{key}.lock")

    def _validate_object_directory(self, path: Path) -> None:
        if self._device is None:
            raise CacheUnavailableError("Cache device is not initialized")
        self._validate_owned_directory(path, self._device)

    def _validate_state_file(
        self, path: Path, expected_device: int, *, required: bool
    ) -> None:
        try:
            value = path.lstat()
        except FileNotFoundError:
            if required:
                raise CacheUnavailableError(
                    f"Cache state file is missing: {path}"
                ) from None
            return
        if path.is_symlink() or not stat.S_ISREG(value.st_mode):
            raise CacheUnavailableError(f"Unsafe cache state file: {path}")
        if value.st_dev != expected_device:
            raise CacheUnavailableError(
                f"Cache state file is on the wrong device: {path}"
            )

    def _validate_storage_before_create(self) -> tuple[Path, int]:
        mountpoint = self.config.required_mountpoint.resolve(strict=True)
        if not mountpoint.is_dir() or not self._mount_probe(mountpoint):
            raise CacheUnavailableError(
                f"Required cache mount is not mounted: {mountpoint}"
            )
        if self.config.root.exists() and self.config.root.is_symlink():
            raise CacheUnavailableError(
                f"Cache root must not be a symlink: {self.config.root}"
            )
        root = self.config.root.resolve(strict=False)
        try:
            root.relative_to(mountpoint)
        except ValueError:
            raise CacheUnavailableError(
                f"Cache root is outside required mount: {root}"
            ) from None

        existing = root
        while not existing.exists():
            if existing == existing.parent:
                raise CacheUnavailableError(
                    f"No existing parent for cache root: {root}"
                )
            existing = existing.parent
        mount_device = mountpoint.stat().st_dev
        if existing.stat().st_dev != mount_device:
            raise CacheUnavailableError(
                f"Cache root parent is not on required mount: {existing}"
            )
        return mountpoint, mount_device

    def _validate_owned_directory(self, path: Path, expected_device: int) -> None:
        value = path.lstat()
        if not stat.S_ISDIR(value.st_mode) or path.is_symlink():
            raise CacheUnavailableError(f"Unsafe cache directory: {path}")
        if value.st_dev != expected_device:
            raise CacheUnavailableError(
                f"Cache directory is on the wrong device: {path}"
            )
        if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
            raise CacheUnavailableError(f"Cache directory is not accessible: {path}")

    def _validate_active_storage(self) -> None:
        if self._device is None:
            raise CacheUnavailableError("Cache device is not initialized")
        mountpoint = self.config.required_mountpoint
        if not self._mount_probe(mountpoint):
            raise CacheUnavailableError(
                f"Required cache mount disappeared: {mountpoint}"
            )
        if mountpoint.stat().st_dev != self._device:
            raise CacheUnavailableError(f"Required cache mount changed: {mountpoint}")
        self._validate_owned_directory(self.config.root, self._device)
        for directory in (self._objects, self._temporary, self._locks):
            self._validate_owned_directory(directory, self._device)
        self._validate_state_file(self._index.path, self._device, required=True)

    def _require_started(self) -> None:
        if not self._started:
            raise CacheUnavailableError("Cache manager has not been started")

    def _bypass(self, original: str) -> CacheLease:
        return CacheLease(
            self, CacheResolution(original, original, None, "bypass"), None
        )


def _cache_key(namespace: str, canonical: Path, suffix: str) -> str:
    identity = f"{namespace}\0{canonical}\0{suffix}".encode(
        "utf-8", errors="surrogatepass"
    )
    return hashlib.sha256(identity).hexdigest()


def _cache_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if _SAFE_SUFFIX.fullmatch(suffix) else ""
