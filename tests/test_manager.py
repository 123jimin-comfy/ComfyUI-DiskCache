from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import threading
from unittest import mock

from disk_cache.errors import CacheUnavailableError
from disk_cache.index import CacheEntry
from disk_cache.locking import InterProcessLock
from disk_cache.manager import CacheManager

from tests.helpers import CacheTestCase


class ManagerTests(CacheTestCase):
    def test_miss_then_hit_preserves_suffix_and_content(self) -> None:
        source = self.source_root / "model.safetensors"
        source.write_bytes(b"model-data")
        manager = self.manager()

        with manager.acquire(source) as lease:
            self.assertEqual(lease.outcome, "miss")
            self.assertTrue(lease.path.endswith(".safetensors"))
            self.assertEqual(Path(lease.path).read_bytes(), b"model-data")
            lease.mark_success()

        with manager.acquire(source) as lease:
            self.assertEqual(lease.outcome, "hit")
            self.assertEqual(Path(lease.path).read_bytes(), b"model-data")
            lease.mark_success()

        self.assertEqual(manager.stats().entries, 1)

    def test_copy_log_formats_byte_counts(self) -> None:
        source = self.source_root / "model.bin"
        source.write_bytes(b"x" * 1_234_567)
        manager = self.manager(config=self.config(max_size_bytes=2_000_000))

        with self.assertLogs("comfyui_disk_cache", level="INFO") as logs:
            with manager.acquire(source) as lease:
                lease.mark_success()

        self.assertTrue(
            any("copying 1,234,567 bytes" in message for message in logs.output)
        )

    def test_source_replacement_invalidates_cached_object(self) -> None:
        source = self.source_root / "model.safetensors"
        source.write_bytes(b"old")
        manager = self.manager()
        with manager.acquire(source) as lease:
            old_cached_path = lease.path
            lease.mark_success()

        old_mtime = source.stat().st_mtime_ns
        source.write_bytes(b"new-content")
        os.utime(source, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))

        with manager.acquire(source) as lease:
            self.assertEqual(lease.outcome, "miss")
            self.assertEqual(Path(lease.path).read_bytes(), b"new-content")
            self.assertEqual(lease.path, old_cached_path)
            lease.mark_success()

    def test_lru_evicts_oldest_entry_by_bytes(self) -> None:
        first = self.source_root / "first.bin"
        second = self.source_root / "second.bin"
        first.write_bytes(b"1111")
        second.write_bytes(b"2222")
        moments = iter((10, 11, 20, 21, 30, 31))
        manager = self.manager(
            config=self.config(max_size_bytes=6), clock=lambda: next(moments)
        )

        with manager.acquire(first) as lease:
            first_cached = Path(lease.path)
            lease.mark_success()
        with manager.acquire(second) as lease:
            second_cached = Path(lease.path)
            lease.mark_success()

        self.assertFalse(first_cached.exists())
        self.assertTrue(second_cached.exists())
        self.assertEqual(manager.stats().entries, 1)

    def test_small_file_is_bypassed_with_exact_path(self) -> None:
        source = self.source_root / "tiny.bin"
        source.write_bytes(b"x")
        manager = self.manager(config=self.config(min_file_size_bytes=2))

        with manager.acquire(str(source)) as lease:
            self.assertEqual(lease.outcome, "bypass")
            self.assertEqual(lease.path, str(source))
            lease.mark_success()

        self.assertEqual(manager.stats().entries, 0)

    def test_file_outside_source_is_bypassed(self) -> None:
        source = self.base / "outside.bin"
        source.write_bytes(b"outside")
        manager = self.manager()

        with manager.acquire(str(source)) as lease:
            self.assertEqual(lease.outcome, "bypass")
            self.assertEqual(lease.path, str(source))

    def test_missing_mount_does_not_create_cache_root(self) -> None:
        manager = CacheManager(self.config(), mount_probe=lambda path: False)

        with self.assertRaises(CacheUnavailableError):
            manager.start()

        self.assertFalse(self.cache_root.exists())

    def test_interrupted_copy_removes_temporary_file(self) -> None:
        source = self.source_root / "model.bin"
        source.write_bytes(b"abcdef")

        def failing_copy(source_handle, target_handle, length):
            target_handle.write(b"partial")
            raise OSError("injected copy failure")

        manager = self.manager(copy_file=failing_copy)

        with self.assertRaisesRegex(OSError, "injected"):
            manager.acquire(source)

        self.assertEqual(list((self.cache_root / "tmp").iterdir()), [])
        self.assertEqual(manager.stats().entries, 0)

    def test_constructor_has_no_filesystem_side_effects(self) -> None:
        CacheManager(self.config(), mount_probe=lambda path: True)

        self.assertFalse(self.cache_root.exists())

    def test_nested_cache_root_is_created(self) -> None:
        nested = self.mountpoint / "one" / "two" / "cache"
        manager = self.manager(config=replace(self.config(), root=nested))

        self.assertEqual(manager.stats().entries, 0)
        self.assertTrue(nested.is_dir())

    def test_mount_disappearance_stops_cache_access(self) -> None:
        mounted = True
        manager = CacheManager(self.config(), mount_probe=lambda path: mounted)
        manager.start()
        source = self.source_root / "model.bin"
        source.write_bytes(b"model")
        mounted = False

        with self.assertRaisesRegex(CacheUnavailableError, "mount disappeared"):
            manager.acquire(source)

    def test_concurrent_same_key_population_copies_once(self) -> None:
        source = self.source_root / "model.safetensors"
        source.write_bytes(b"model-data")
        copy_started = threading.Event()
        release_copy = threading.Event()
        copy_count = 0
        copy_count_lock = threading.Lock()

        def slow_copy(source_handle, target_handle, length):
            nonlocal copy_count
            with copy_count_lock:
                copy_count += 1
            copy_started.set()
            if not release_copy.wait(5):
                raise TimeoutError("test did not release copy")
            shutil.copyfileobj(source_handle, target_handle, length)

        manager = self.manager(copy_file=slow_copy)
        outcomes: list[str] = []
        failures: list[BaseException] = []

        def load() -> None:
            try:
                with manager.acquire(source) as lease:
                    outcomes.append(lease.outcome)
                    lease.mark_success()
            except BaseException as exc:
                failures.append(exc)

        first = threading.Thread(target=load)
        second = threading.Thread(target=load)
        first.start()
        self.assertTrue(copy_started.wait(5))
        second.start()
        release_copy.set()
        first.join(5)
        second.join(5)

        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(sorted(outcomes), ["hit", "miss"])
        self.assertEqual(copy_count, 1)

    def test_failed_fast_lookup_releases_object_lock(self) -> None:
        source = self.source_root / "model.bin"
        source.write_bytes(b"model")
        manager = self.manager()
        with manager.acquire(source) as lease:
            lease.mark_success()

        with mock.patch.object(manager._index, "get", side_effect=OSError("index")):
            with self.assertRaisesRegex(OSError, "index"):
                manager.acquire(source)

        lock_path = next(
            path
            for path in (self.cache_root / "locks").glob("*.lock")
            if len(path.stem) == 64
        )
        probe = InterProcessLock(lock_path)
        self.assertTrue(probe.acquire(blocking=False))
        probe.release()

    def test_failed_index_write_removes_unindexed_object(self) -> None:
        source = self.source_root / "model.bin"
        source.write_bytes(b"model")
        manager = self.manager()

        with mock.patch.object(manager._index, "upsert", side_effect=OSError("index")):
            with self.assertRaisesRegex(OSError, "index"):
                manager.acquire(source)

        self.assertEqual(manager._index.count(), 0)
        self.assertEqual(
            [
                path
                for path in (self.cache_root / "objects").rglob("*")
                if path.is_file()
            ],
            [],
        )

    def test_unlink_failure_keeps_index_accounting(self) -> None:
        source = self.source_root / "model.bin"
        source.write_bytes(b"model")
        manager = self.manager()
        with manager.acquire(source) as lease:
            lease.mark_success()
        entry = manager._index.entries()[0]

        with mock.patch.object(Path, "unlink", side_effect=PermissionError("busy")):
            with self.assertRaisesRegex(PermissionError, "busy"):
                manager._remove_entry(entry)

        self.assertEqual(manager._index.count(), 1)

    def test_clear_does_not_start_when_any_object_is_busy(self) -> None:
        first = self.source_root / "first.bin"
        second = self.source_root / "second.bin"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        manager = self.manager()
        cached_paths: list[Path] = []
        for source in (first, second):
            with manager.acquire(source) as lease:
                cached_paths.append(Path(lease.path))
                lease.mark_success()

        with manager.acquire(second):
            with self.assertRaisesRegex(CacheUnavailableError, "busy cache object"):
                manager.clear()
            self.assertEqual(manager._index.count(), 2)
            self.assertTrue(all(path.exists() for path in cached_paths))

    def test_reconcile_preserves_object_held_by_another_manager(self) -> None:
        source = self.source_root / "model.bin"
        source.write_bytes(b"old")
        first_manager = self.manager()
        with first_manager.acquire(source) as lease:
            lease.mark_success()

        with first_manager.acquire(source) as busy:
            source.write_bytes(b"replacement")
            second_manager = self.manager()
            self.assertTrue(Path(busy.path).exists())
            self.assertEqual(second_manager._index.count(), 1)

        with second_manager.acquire(source) as lease:
            self.assertEqual(lease.outcome, "miss")
            self.assertEqual(Path(lease.path).read_bytes(), b"replacement")

    def test_reconcile_discards_malformed_index_key(self) -> None:
        source = self.source_root / "model.bin"
        extra = self.source_root / "extra.bin"
        source.write_bytes(b"model")
        extra.write_bytes(b"extra")
        first_manager = self.manager()
        with first_manager.acquire(source) as lease:
            lease.mark_success()
        existing = first_manager._index.entries()[0]
        first_manager._index.upsert(
            CacheEntry(
                key="../unsafe",
                namespace=existing.namespace,
                source_path=str(extra),
                source_dev=extra.stat().st_dev,
                source_ino=extra.stat().st_ino,
                source_size=extra.stat().st_size,
                source_mtime_ns=extra.stat().st_mtime_ns,
                cache_suffix=".bin",
                cached_size=extra.stat().st_size,
                created_ns=existing.created_ns,
                last_access_ns=existing.last_access_ns,
            )
        )

        with self.assertLogs("comfyui_disk_cache", level="WARNING"):
            second_manager = self.manager()

        self.assertEqual(second_manager._index.count(), 1)

    def test_symlinked_object_shard_is_rejected(self) -> None:
        source = self.source_root / "model.bin"
        source.write_bytes(b"model")
        manager = self.manager()
        with manager.acquire(source) as lease:
            shard = Path(lease.path).parent
            lease.mark_success()
        manager.clear()
        shard.rmdir()
        outside = self.base / "outside-cache"
        outside.mkdir()
        try:
            shard.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")

        with self.assertRaisesRegex(CacheUnavailableError, "Unsafe cache directory"):
            manager.acquire(source)
        self.assertEqual(list(outside.iterdir()), [])
