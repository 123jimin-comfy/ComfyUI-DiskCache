from __future__ import annotations

import functools
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from disk_cache.errors import CompatibilityError
from disk_cache.integration import (
    PatchPlan,
    PatchTarget,
    detect_comfyui_version,
    prepare_patch_plan,
)


def make_comfy_utils(
    signature: str = "ckpt, safe_load=False, device=None, return_metadata=False",
):
    module = ModuleType("comfy.utils")
    module.__file__ = __file__
    source = (
        f"def load_torch_file({signature}):\n"
        "    return ckpt, safe_load, device, return_metadata\n"
    )
    exec(compile(source, __file__, "exec"), module.__dict__)
    return module


class FakeLease:
    def __init__(self, path: str) -> None:
        self.path = path
        self.marked = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def mark_success(self) -> None:
        self.marked = True


class FakeManager:
    def __init__(self, *, fail_open: bool = True, failure: Exception | None = None):
        self.config = SimpleNamespace(fail_open=fail_open)
        self.failure = failure
        self.paths: list[str] = []
        self.lease: FakeLease | None = None

    def acquire(self, path: str) -> FakeLease:
        self.paths.append(path)
        if self.failure is not None:
            raise self.failure
        self.lease = FakeLease(f"cache/{path}")
        return self.lease


class IntegrationTests(unittest.TestCase):
    def test_missing_version_module_is_reported_as_unknown(self) -> None:
        with mock.patch(
            "disk_cache.integration.importlib.import_module",
            side_effect=ModuleNotFoundError("comfyui_version"),
        ):
            self.assertEqual(detect_comfyui_version(), "unknown")

    def test_empty_version_is_reported_as_unknown(self) -> None:
        module = ModuleType("comfyui_version")
        module.__version__ = ""
        with mock.patch(
            "disk_cache.integration.importlib.import_module", return_value=module
        ):
            self.assertEqual(detect_comfyui_version(), "unknown")

    def test_broken_version_module_is_reported_as_unknown(self) -> None:
        with mock.patch(
            "disk_cache.integration.importlib.import_module",
            side_effect=RuntimeError("broken version module"),
        ):
            self.assertEqual(detect_comfyui_version(), "unknown")

    def test_guarded_patch_wraps_core_and_loaded_alias(self) -> None:
        module = make_comfy_utils()
        alias = ModuleType("comfy.clip_vision")
        alias.load_torch_file = module.load_torch_file
        original = module.load_torch_file
        plan = prepare_patch_plan(
            module,
            comfyui_version="future",
            compatibility_policy="guarded",
            loaded_modules={"comfy.clip_vision": alias},
        )
        manager = FakeManager()

        controller = plan.install(manager)  # type: ignore[arg-type]
        try:
            self.assertIs(module.load_torch_file, alias.load_torch_file)
            result = module.load_torch_file("model.safetensors", True, "cpu", True)
            self.assertEqual(result, ("cache/model.safetensors", True, "cpu", True))
            self.assertEqual(manager.paths, ["model.safetensors"])
            self.assertTrue(manager.lease.marked)
        finally:
            controller.uninstall()

        self.assertIs(module.load_torch_file, original)
        self.assertIs(alias.load_torch_file, original)

    def test_loaded_third_party_exact_alias_is_patched(self) -> None:
        module = make_comfy_utils()
        custom_node = ModuleType("custom_node")
        custom_node.load_torch_file = module.load_torch_file
        plan = prepare_patch_plan(
            module,
            comfyui_version="future",
            compatibility_policy="guarded",
            loaded_modules={"custom_node": custom_node},
        )

        controller = plan.install(FakeManager())  # type: ignore[arg-type]
        try:
            self.assertIs(custom_node.load_torch_file, module.load_torch_file)
        finally:
            controller.uninstall()

    def test_changed_signature_is_rejected_without_mutation(self) -> None:
        module = make_comfy_utils("path")
        original = module.load_torch_file

        with self.assertRaisesRegex(CompatibilityError, "signature"):
            prepare_patch_plan(
                module,
                comfyui_version="0.34.0",
                compatibility_policy="guarded",
                loaded_modules={},
            )

        self.assertIs(module.load_torch_file, original)

    def test_strict_mode_rejects_untested_version(self) -> None:
        module = make_comfy_utils()

        with self.assertRaisesRegex(CompatibilityError, "untested"):
            prepare_patch_plan(
                module,
                comfyui_version="99.0.0",
                compatibility_policy="strict",
                loaded_modules={},
            )

    def test_strict_mode_rejects_changed_tested_implementation(self) -> None:
        module = make_comfy_utils()

        with self.assertRaisesRegex(CompatibilityError, "inspect|implementation"):
            prepare_patch_plan(
                module,
                comfyui_version="0.34.0",
                compatibility_policy="strict",
                loaded_modules={},
            )

    def test_unknown_wrapper_is_rejected(self) -> None:
        module = make_comfy_utils()
        original = module.load_torch_file

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            return original(*args, **kwargs)

        module.load_torch_file = wrapper

        with self.assertRaisesRegex(CompatibilityError, "unknown wrapper"):
            prepare_patch_plan(
                module,
                comfyui_version="0.34.0",
                compatibility_policy="guarded",
                loaded_modules={},
            )

    def test_wrong_code_origin_is_rejected(self) -> None:
        module = make_comfy_utils()
        module.__file__ = f"{__file__}.different"

        with self.assertRaisesRegex(CompatibilityError, "does not originate"):
            prepare_patch_plan(
                module,
                comfyui_version="0.34.0",
                compatibility_policy="guarded",
                loaded_modules={},
            )

    def test_unknown_alias_patch_is_rejected(self) -> None:
        module = make_comfy_utils()
        alias = ModuleType("comfy.clip_vision")
        alias.load_torch_file = lambda path: path

        with self.assertRaisesRegex(CompatibilityError, "no longer aliases"):
            prepare_patch_plan(
                module,
                comfyui_version="0.34.0",
                compatibility_policy="guarded",
                loaded_modules={"comfy.clip_vision": alias},
            )

    def test_patch_install_rolls_back_on_late_assignment_failure(self) -> None:
        module = make_comfy_utils()
        original = module.load_torch_file

        class RejectingModule:
            __name__ = "rejecting"

            def __init__(self):
                object.__setattr__(self, "load_torch_file", original)

            def __setattr__(self, name, value):
                if name == "load_torch_file" and value is not original:
                    raise RuntimeError("injected assignment failure")
                object.__setattr__(self, name, value)

        rejecting = RejectingModule()
        plan = PatchPlan(
            original_loader=original,
            targets=(
                PatchTarget(module, "load_torch_file", original),
                PatchTarget(rejecting, "load_torch_file", original),
            ),
            comfyui_version="0.34.0",
            implementation_is_tested=True,
        )

        with self.assertRaisesRegex(RuntimeError, "injected"):
            plan.install(FakeManager())  # type: ignore[arg-type]

        self.assertIs(module.load_torch_file, original)

    def test_cache_failure_falls_back_to_exact_source_argument(self) -> None:
        module = make_comfy_utils()
        plan = prepare_patch_plan(
            module,
            comfyui_version="0.34.0",
            compatibility_policy="guarded",
            loaded_modules={},
        )
        manager = FakeManager(fail_open=True, failure=OSError("cache failed"))
        controller = plan.install(manager)  # type: ignore[arg-type]
        try:
            with self.assertLogs("comfyui_disk_cache", level="WARNING"):
                result = module.load_torch_file("relative/model.bin")
        finally:
            controller.uninstall()

        self.assertEqual(result[0], "relative/model.bin")

    def test_cache_failure_is_raised_when_fail_open_is_disabled(self) -> None:
        module = make_comfy_utils()
        plan = prepare_patch_plan(
            module,
            comfyui_version="0.34.0",
            compatibility_policy="guarded",
            loaded_modules={},
        )
        manager = FakeManager(fail_open=False, failure=OSError("cache failed"))
        controller = plan.install(manager)  # type: ignore[arg-type]
        try:
            with self.assertRaisesRegex(OSError, "cache failed"):
                module.load_torch_file("model.bin")
        finally:
            controller.uninstall()

    def test_loader_exception_is_not_swallowed_or_retried(self) -> None:
        module = make_comfy_utils()
        calls = []

        def failing_loader(ckpt, safe_load=False, device=None, return_metadata=False):
            calls.append(ckpt)
            raise ValueError("invalid model")

        failing_loader.__module__ = "comfy.utils"
        failing_loader.__qualname__ = "load_torch_file"
        failing_loader.__name__ = "load_torch_file"
        plan = PatchPlan(
            original_loader=failing_loader,
            targets=(PatchTarget(module, "load_torch_file", failing_loader),),
            comfyui_version="0.34.0",
            implementation_is_tested=True,
        )
        module.load_torch_file = failing_loader
        controller = plan.install(FakeManager())  # type: ignore[arg-type]
        try:
            with self.assertRaisesRegex(ValueError, "invalid model"):
                module.load_torch_file("model.bin")
        finally:
            controller.uninstall()

        self.assertEqual(calls, ["cache/model.bin"])


if __name__ == "__main__":
    unittest.main()
