"""ComfyUI V3 extension lifecycle."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from comfy_api.latest import ComfyExtension  # type: ignore[import-not-found]

from .config import CacheConfig, discover_config_path, load_config
from .errors import CompatibilityError, ConfigurationError
from .integration import PatchController, detect_comfyui_version, prepare_patch_plan
from .manager import CacheManager


_LOG = logging.getLogger("comfyui_disk_cache")


class DiskCacheExtension(ComfyExtension):
    """Install the cache adapter only after all compatibility checks pass."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path
        self._config: CacheConfig | None = None
        self._manager: CacheManager | None = None
        self._patch: PatchController | None = None

    async def on_load(self) -> None:
        try:
            config_path = self._config_path or discover_config_path()
            config = load_config(config_path)
        except ConfigurationError as exc:
            _LOG.warning("Disk cache disabled: %s", exc)
            return
        except Exception as exc:
            _LOG.error(
                "Disk cache configuration discovery failed; no changes made: %s",
                exc,
                exc_info=True,
            )
            return
        if not config.enabled:
            _LOG.info("Disk cache disabled by %s", config_path)
            return

        try:
            comfy_utils = importlib.import_module("comfy.utils")
            comfyui_version = detect_comfyui_version()
            plan = prepare_patch_plan(
                comfy_utils,
                comfyui_version=comfyui_version,
                compatibility_policy=config.compatibility_policy,
            )
        except (CompatibilityError, ImportError, AttributeError) as exc:
            _LOG.error(
                "Disk cache compatibility check failed; no changes made: %s", exc
            )
            return
        except Exception as exc:
            _LOG.error(
                "Unexpected disk cache compatibility failure; no changes made: %s",
                exc,
                exc_info=True,
            )
            return

        if not plan.implementation_is_tested:
            _LOG.warning(
                "ComfyUI %s loader implementation is not in the tested matrix; "
                "structural checks passed",
                plan.comfyui_version,
            )

        manager = CacheManager(config)
        patch: PatchController | None = None
        try:
            patch = plan.install(manager)
            manager.start()
        except Exception as exc:
            if patch is not None:
                patch.uninstall()
            message = f"Disk cache initialization failed; loader restored: {exc}"
            if config.fail_open:
                _LOG.error(message, exc_info=True)
                return
            raise RuntimeError(message) from exc

        self._config = config
        self._manager = manager
        self._patch = patch
        _LOG.info(
            "Disk cache enabled for ComfyUI %s with %d loader target(s)",
            plan.comfyui_version,
            len(plan.targets),
        )

    async def get_node_list(self) -> list[type]:
        return []
