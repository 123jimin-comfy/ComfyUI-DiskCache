"""Portable named locks for cache mutations."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import BinaryIO
import errno
import os
import stat
import threading
import time


class InterProcessLock:
    """A thread and process exclusive lock backed by a persistent lock file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = None
        self._thread_lock = _thread_lock_for(path)
        self._thread_lock_held = False

    def acquire(self, *, blocking: bool = True) -> bool:
        if self._handle is not None:
            raise RuntimeError(f"Lock is already held: {self.path}")
        if not self._thread_lock.acquire(blocking=blocking):
            return False
        self._thread_lock_held = True

        descriptor: int | None = None
        handle: BinaryIO | None = None
        try:
            flags = os.O_RDWR | os.O_CREAT
            flags |= getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode):
                raise OSError(f"Lock path is not a regular file: {self.path}")
            handle = os.fdopen(descriptor, "r+b", buffering=0)
            descriptor = None
            if handle.seek(0, 2) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if not _lock_file(handle, blocking=blocking):
                try:
                    handle.close()
                finally:
                    self._thread_lock.release()
                    self._thread_lock_held = False
                return False
            self._handle = handle
            return True
        except Exception:
            try:
                if handle is not None:
                    handle.close()
                elif descriptor is not None:
                    os.close(descriptor)
            finally:
                if self._thread_lock_held:
                    self._thread_lock.release()
                    self._thread_lock_held = False
            raise

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            raise RuntimeError(f"Lock is not held: {self.path}")
        try:
            handle.seek(0)
            _unlock_file(handle)
        finally:
            try:
                handle.close()
            finally:
                self._thread_lock.release()
                self._thread_lock_held = False

    def __enter__(self) -> "InterProcessLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self.release()


_thread_locks: dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()


def _thread_lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve(strict=False))
    with _thread_locks_guard:
        return _thread_locks.setdefault(key, threading.Lock())


def _lock_file(handle: BinaryIO, *, blocking: bool) -> bool:
    if _WINDOWS:
        import msvcrt

        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if not blocking:
                    return False
                time.sleep(0.05)
    else:
        import fcntl

        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
            return True
        except BlockingIOError:
            return False


def _unlock_file(handle: BinaryIO) -> None:
    if _WINDOWS:
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


_WINDOWS = os.name == "nt"
