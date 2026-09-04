"""Guarded integration with ComfyUI's model-file loader."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import functools
import hashlib
import importlib
import inspect
import logging
from pathlib import Path
import sys
import textwrap
from types import ModuleType
from typing import Any, Callable, Mapping

from .errors import CompatibilityError
from .manager import CacheManager


_LOG = logging.getLogger("comfyui_disk_cache")
_OWNER_ATTRIBUTE = "__comfyui_disk_cache_owner__"
_OWNER = "comfyui-disk-cache:v1"
_ALIAS_MODULES = ("comfy.clip_vision", "comfy.bg_removal_model")
_TESTED_LOADER_FINGERPRINTS = {
    "0.34.0": "f4b7230f7d7bed15e2cd881ca380cbaf6f22acc8fa9a3a885ce1ea21bb8cc21e"
}
TESTED_COMFYUI_VERSIONS = frozenset(_TESTED_LOADER_FINGERPRINTS)


def detect_comfyui_version() -> str:
    """Return ComfyUI's optional version string, or a safe unknown marker."""

    try:
        value = importlib.import_module("comfyui_version").__version__
    except Exception:
        return "unknown"
    return value if isinstance(value, str) and value else "unknown"


@dataclass(frozen=True, slots=True)
class PatchTarget:
    module: Any
    attribute: str
    original: Callable[..., Any]

    @property
    def label(self) -> str:
        return f"{self.module.__name__}.{self.attribute}"


@dataclass(frozen=True, slots=True)
class PatchPlan:
    """A fully validated, side-effect-free set of patch targets."""

    original_loader: Callable[..., Any]
    targets: tuple[PatchTarget, ...]
    comfyui_version: str
    implementation_is_tested: bool

    def install(self, manager: CacheManager) -> "PatchController":
        wrapper = _make_wrapper(self.original_loader, manager)
        applied: list[PatchTarget] = []
        try:
            for target in self.targets:
                if vars(target.module).get(target.attribute) is not target.original:
                    raise CompatibilityError(
                        f"Patch target changed after validation: {target.label}"
                    )
                setattr(target.module, target.attribute, wrapper)
                applied.append(target)
        except Exception:
            for target in reversed(applied):
                if vars(target.module).get(target.attribute) is wrapper:
                    setattr(target.module, target.attribute, target.original)
            raise
        return PatchController(tuple(applied), wrapper)


class PatchController:
    """Own installed replacements and restore only functions still owned by us."""

    def __init__(
        self, targets: tuple[PatchTarget, ...], wrapper: Callable[..., Any]
    ) -> None:
        self.targets = targets
        self.wrapper = wrapper

    def uninstall(self) -> None:
        for target in reversed(self.targets):
            if vars(target.module).get(target.attribute) is self.wrapper:
                setattr(target.module, target.attribute, target.original)


def prepare_patch_plan(
    comfy_utils: ModuleType,
    *,
    comfyui_version: str,
    compatibility_policy: str,
    loaded_modules: Mapping[str, ModuleType | None] | None = None,
) -> PatchPlan:
    """Validate the complete ComfyUI contract without changing any state."""

    if comfy_utils.__name__ != "comfy.utils":
        raise CompatibilityError(
            f"Expected module comfy.utils, got {comfy_utils.__name__!r}"
        )

    original = getattr(comfy_utils, "load_torch_file", None)
    if not inspect.isfunction(original):
        raise CompatibilityError("comfy.utils.load_torch_file is not a Python function")
    if getattr(original, _OWNER_ATTRIBUTE, None) is not None:
        owner = getattr(original, _OWNER_ATTRIBUTE)
        raise CompatibilityError(
            f"comfy.utils.load_torch_file is already patched by {owner}"
        )
    if hasattr(original, "__wrapped__"):
        raise CompatibilityError("comfy.utils.load_torch_file has an unknown wrapper")
    if original.__module__ != "comfy.utils":
        raise CompatibilityError(
            f"Unexpected loader module: {original.__module__!r}"
        )
    if (
        original.__name__ != "load_torch_file"
        or original.__qualname__ != "load_torch_file"
    ):
        raise CompatibilityError("Unexpected load_torch_file function identity")

    signature = inspect.signature(original, follow_wrapped=False)
    expected_signature = inspect.Signature(
        parameters=(
            inspect.Parameter("ckpt", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                "safe_load", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=False
            ),
            inspect.Parameter(
                "device", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            ),
            inspect.Parameter(
                "return_metadata",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=False,
            ),
        )
    )
    if signature != expected_signature:
        raise CompatibilityError(
            "Unsupported comfy.utils.load_torch_file signature: "
            f"expected {expected_signature}, got {signature}"
        )

    module_file = getattr(comfy_utils, "__file__", None)
    code_file = getattr(original, "__code__", None)
    if module_file is None or code_file is None:
        raise CompatibilityError("Cannot verify load_torch_file source identity")
    if Path(module_file).resolve() != Path(code_file.co_filename).resolve():
        raise CompatibilityError(
            "load_torch_file code does not originate from comfy.utils"
        )

    if compatibility_policy not in {"guarded", "strict"}:
        raise CompatibilityError(
            f"Unknown compatibility policy: {compatibility_policy!r}"
        )

    expected_fingerprint = _TESTED_LOADER_FINGERPRINTS.get(comfyui_version)
    if compatibility_policy == "strict" and expected_fingerprint is None:
        tested = ", ".join(sorted(TESTED_COMFYUI_VERSIONS))
        raise CompatibilityError(
            f"ComfyUI {comfyui_version!r} is untested; strict mode accepts: {tested}"
        )
    try:
        actual_fingerprint = _loader_fingerprint(original)
    except CompatibilityError:
        if compatibility_policy == "strict":
            raise
        actual_fingerprint = None
    implementation_is_tested = (
        expected_fingerprint is not None
        and actual_fingerprint == expected_fingerprint
    )
    if compatibility_policy == "strict" and not implementation_is_tested:
        raise CompatibilityError(
            f"ComfyUI {comfyui_version} load_torch_file implementation is untested"
        )

    modules = loaded_modules if loaded_modules is not None else sys.modules
    targets = [PatchTarget(comfy_utils, "load_torch_file", original)]
    target_ids = {id(comfy_utils)}
    for module_name in _ALIAS_MODULES:
        module = modules.get(module_name)
        if module is None:
            continue
        if not isinstance(module, ModuleType):
            raise CompatibilityError(
                f"Unexpected loaded-module value for {module_name}"
            )
        alias = vars(module).get("load_torch_file")
        if alias is not original:
            raise CompatibilityError(
                f"{module_name}.load_torch_file no longer aliases the core loader"
            )

    for _, module in sorted(modules.items()):
        if not isinstance(module, ModuleType) or id(module) in target_ids:
            continue
        if vars(module).get("load_torch_file") is original:
            targets.append(PatchTarget(module, "load_torch_file", original))
            target_ids.add(id(module))

    return PatchPlan(
        original_loader=original,
        targets=tuple(targets),
        comfyui_version=comfyui_version,
        implementation_is_tested=implementation_is_tested,
    )


def _loader_fingerprint(loader: Callable[..., Any]) -> str:
    try:
        source = textwrap.dedent(inspect.getsource(loader))
        tree = ast.parse(source)
    except (OSError, TypeError, IndentationError, SyntaxError) as exc:
        raise CompatibilityError(
            "Cannot inspect comfy.utils.load_torch_file implementation"
        ) from exc
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_torch_file"
    ]
    if len(definitions) != 1:
        raise CompatibilityError(
            "Cannot identify comfy.utils.load_torch_file implementation"
        )
    normalized = ast.dump(
        definitions[0], annotate_fields=True, include_attributes=False
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _make_wrapper(
    original: Callable[..., Any], manager: CacheManager
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(ckpt: str, *args: Any, **kwargs: Any) -> Any:
        try:
            lease = manager.acquire(ckpt)
        except Exception as exc:
            if not manager.config.fail_open:
                raise
            _LOG.warning(
                "Cache unavailable for %s; loading the authoritative file: %s",
                ckpt,
                exc,
            )
            return original(ckpt, *args, **kwargs)

        with lease:
            result = original(lease.path, *args, **kwargs)
            lease.mark_success()
            return result

    setattr(wrapped, _OWNER_ATTRIBUTE, _OWNER)
    return wrapped
