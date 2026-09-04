from __future__ import annotations

from pathlib import Path

from disk_cache.config import load_config
from disk_cache.errors import ConfigurationError

from tests.helpers import CacheTestCase


class ConfigTests(CacheTestCase):
    def write_config(self, body: str) -> Path:
        path = self.base / "config.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def valid_body(self) -> str:
        return f"""
[cache]
root = "{self.cache_root.as_posix()}"
required_mountpoint = "{self.mountpoint.as_posix()}"
max_size_gib = 1.5
min_free_gib = 0
min_file_size_mib = 64
fail_open = true
compatibility_policy = "strict"

[[sources]]
name = "models"
root = "{self.source_root.as_posix()}"
"""

    def test_loads_strict_config(self) -> None:
        config = load_config(self.write_config(self.valid_body()))

        self.assertEqual(config.max_size_bytes, int(1.5 * 1024**3))
        self.assertEqual(config.min_file_size_bytes, 64 * 1024**2)
        self.assertEqual(config.sources[0].name, "models")

    def test_guarded_compatibility_is_the_default(self) -> None:
        body = self.valid_body().replace('compatibility_policy = "strict"\n', "")

        config = load_config(self.write_config(body))

        self.assertEqual(config.compatibility_policy, "guarded")

    def test_rejects_unknown_key(self) -> None:
        body = self.valid_body().replace(
            "fail_open = true", "fail_open = true\nmystery = true"
        )

        with self.assertRaisesRegex(ConfigurationError, "Unknown.*mystery"):
            load_config(self.write_config(body))

    def test_rejects_relative_cache_path(self) -> None:
        body = self.valid_body().replace(
            self.cache_root.as_posix(), "relative/cache"
        )

        with self.assertRaisesRegex(ConfigurationError, "absolute path"):
            load_config(self.write_config(body))

    def test_rejects_cache_outside_mount(self) -> None:
        body = self.valid_body().replace(
            self.cache_root.as_posix(), (self.base / "elsewhere").as_posix()
        )

        with self.assertRaisesRegex(ConfigurationError, "must be a child"):
            load_config(self.write_config(body))

    def test_rejects_overlapping_source_and_cache(self) -> None:
        overlapping = self.cache_root / "models"
        body = self.valid_body().replace(
            self.source_root.as_posix(), overlapping.as_posix()
        )

        with self.assertRaisesRegex(ConfigurationError, "must not overlap"):
            load_config(self.write_config(body))
