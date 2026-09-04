from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Callable
import unittest

from disk_cache.config import CacheConfig, SourceConfig
from disk_cache.manager import CacheManager


class CacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name)
        self.mountpoint = self.base / "mount"
        self.source_root = self.base / "source"
        self.cache_root = self.mountpoint / "cache"
        self.mountpoint.mkdir()
        self.source_root.mkdir()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def config(
        self,
        *,
        max_size_bytes: int = 1024 * 1024,
        min_free_bytes: int = 0,
        min_file_size_bytes: int = 0,
        fail_open: bool = True,
    ) -> CacheConfig:
        return CacheConfig(
            enabled=True,
            root=self.cache_root,
            required_mountpoint=self.mountpoint,
            max_size_bytes=max_size_bytes,
            min_free_bytes=min_free_bytes,
            min_file_size_bytes=min_file_size_bytes,
            eviction="lru",
            validation="stat",
            touch_on_hit=True,
            fail_open=fail_open,
            miss_policy="copy_then_load",
            compatibility_policy="strict",
            sources=(SourceConfig("models", self.source_root),),
        )

    def manager(
        self,
        *,
        config: CacheConfig | None = None,
        clock: Callable[[], int] | None = None,
        copy_file=None,
    ) -> CacheManager:
        kwargs = {
            "mount_probe": lambda path: path.resolve() == self.mountpoint.resolve(),
        }
        if clock is not None:
            kwargs["clock"] = clock
        if copy_file is not None:
            kwargs["copy_file"] = copy_file
        manager = CacheManager(config or self.config(), **kwargs)
        manager.start()
        return manager
