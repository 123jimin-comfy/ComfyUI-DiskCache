"""Local administration CLI for the model-file cache."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from .config import discover_config_path, load_config
from .errors import DiskCacheError
from .manager import CacheManager, CacheStats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comfyui-disk-cache")
    parser.add_argument("--config", type=Path, help="Path to model-disk-cache.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show indexed and filesystem usage")
    subparsers.add_parser("prune", help="Apply the configured LRU limits")
    clear = subparsers.add_parser("clear", help="Remove all cached model files")
    clear.add_argument("--yes", action="store_true", help="Confirm cache deletion")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config_path = args.config or discover_config_path()
    try:
        config = load_config(config_path)
        manager = CacheManager(config)
        manager.start()
        if args.command == "status":
            _print_stats(manager.stats())
        elif args.command == "prune":
            _print_stats(manager.prune())
        elif args.command == "clear":
            if not args.yes:
                print("Refusing to clear the cache without --yes", file=sys.stderr)
                return 2
            manager.clear()
            _print_stats(manager.stats())
        return 0
    except (DiskCacheError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _print_stats(stats: CacheStats) -> None:
    print(f"entries:       {stats.entries}")
    print(f"indexed size:  {_human_size(stats.indexed_bytes)}")
    print(f"filesystem free: {_human_size(stats.free_bytes)}")
    print(f"cache limit:   {_human_size(stats.max_size_bytes)}")
    print(f"free reserve:  {_human_size(stats.min_free_bytes)}")


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
