# ComfyUI Disk Cache

ComfyUI Disk Cache transparently copies large model files from authoritative
storage to a configured local cache filesystem immediately before ComfyUI loads
them. Authoritative files remain unchanged; workflows and model names do not.

The extension is deliberately narrow. It wraps the long-lived
`comfy.utils.load_torch_file` boundary after verifying its identity and exact
signature. If that contract changes, the extension reports the incompatibility
and makes no changes to ComfyUI or the cache filesystem. Optional strict mode
also verifies a normalized implementation fingerprint.

## Supported loading paths

The adapter covers standard ComfyUI checkpoint, diffusion-model, text-encoder,
VAE, ControlNet, and LoRA loaders. It also transactionally patches every loaded
module that holds an exact direct alias to the verified function. This includes
the two known core aliases in `comfy.clip_vision` and
`comfy.bg_removal_model`, plus third-party nodes loaded earlier.

It intentionally does not globally wrap `open`, `torch.load`, or Safetensors.
Consequently, GGUF, Nunchaku, textual-inversion embeddings, metadata scans, and
arbitrary third-party direct file access are not cached. Metadata scans should
not populate multi-gigabyte cache entries merely by inspecting them.

## Requirements

- ComfyUI with the V3 `ComfyExtension` API
- Python 3.11 or newer
- A mounted, writable cache filesystem

There are no additional Python dependencies. Guarded compatibility mode works
independently of a particular ComfyUI version number when the verified loader
contract is present. An exact strict-mode fingerprint is currently included for
ComfyUI `0.34.0`.

## Installation

Clone the repository directly into ComfyUI's custom-node directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/123jimin-comfy/ComfyUI-DiskCache.git
```

Copy the example configuration to persistent storage and edit it if needed:

```bash
cp ComfyUI-DiskCache/config.example.toml \
  /path/to/ComfyUI/user/model-disk-cache.toml
```

The extension first reads the path in `COMFYUI_MODEL_DISK_CACHE_CONFIG`. Without
that variable it reads `model-disk-cache.toml` from ComfyUI's user directory.
The environment variable may point to any readable configuration file:

```bash
export COMFYUI_MODEL_DISK_CACHE_CONFIG=/absolute/path/to/model-disk-cache.toml
```

Ensure the configured `required_mountpoint` is mounted before starting ComfyUI,
using whatever service manager or launch process owns that environment. The
extension verifies the mount and device before writing cache data. If
configuration, storage, or compatibility checks fail, it reports the error and
leaves ComfyUI's loader unchanged.

Do not list the generated cache tree in `extra_model_paths.yaml`. ComfyUI should
enumerate only authoritative persistent models; this extension changes the path
passed to the actual loader.

Restart ComfyUI through its normal launch mechanism and inspect its usual log.
The relevant events are `MISS`, `READY`, `HIT`, and `EVICT`. Each includes the
authoritative path; completed cache operations also include the cache path.

## Configuration

```toml
# Replace these illustrative paths and capacity values for the target system.
[cache]
enabled = true
root = "/path/to/cache-mount/comfyui-disk-cache"
required_mountpoint = "/path/to/cache-mount"
max_size_gib = 100
min_free_gib = 10
min_file_size_mib = 64
eviction = "lru"
validation = "stat"
touch_on_hit = true
fail_open = true
miss_policy = "copy_then_load"
compatibility_policy = "guarded"

[[sources]]
name = "comfy-models"
root = "/path/to/ComfyUI/models"
```

Unknown keys and unsupported values are rejected so typos cannot silently
change behavior. All paths are configuration-driven and must be absolute.

- `root` is the extension-owned cache directory. It must be a child of
  `required_mountpoint`, not the mountpoint itself.
- `required_mountpoint` must identify a currently mounted filesystem. This
  prevents a missing removable or ephemeral volume from redirecting writes to
  an underlying directory.
- `max_size_gib` bounds indexed cache objects.
- `min_free_gib` reserves actual filesystem space. Actual free space is checked
  before every copy because an unlinked file can continue consuming blocks while
  a process still has it open or memory-mapped.
- `min_file_size_mib` bypasses files too small to benefit.
- `touch_on_hit` explicitly updates the cached file timestamp despite `noatime`.
- `fail_open` falls back to the exact persistent path on cache infrastructure
  errors. Exceptions from the real model loader are never swallowed or retried.
- `compatibility_policy = "guarded"` is the default. It accepts any ComfyUI
  version only when every structural loader check passes and logs a warning for
  an implementation outside the tested matrix. `"strict"` additionally
  requires a known version and exact reviewed implementation fingerprint.

Only `lru`, `stat`, and `copy_then_load` are currently accepted for their
respective settings. Unsupported future-looking values fail loudly instead of
pretending to work.

Multiple authoritative trees can be declared with additional `[[sources]]`
tables. Symlinks escaping a configured source root are not cached.

## Cold misses and eviction

A cold miss copies the complete source file and then loads the cached copy. This
can make the first load substantially slower for very large files. The benefit
begins on subsequent loads; benchmark cold `A`, cold `B`, then `A` again.

Population uses a bounded buffer, a temporary file, source identity checks, and
an atomic rename. A killed process cannot turn a partial temporary file into a
hit. Startup reconciliation removes partial and orphaned objects.

LRU state is stored in SQLite. Cache mutation is serialized across processes,
and each object remains locked until the underlying ComfyUI loader returns.
Existing Linux mmaps remain valid if an older object is later evicted.

## Administration

Run the CLI from the repository directory:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-DiskCache
python -m disk_cache.cli \
  --config /absolute/path/to/model-disk-cache.toml status
```

Available commands are `status`, `prune`, and `clear --yes`. Stop ComfyUI before
using `clear` if deterministic removal of every entry is required; busy objects
are protected by locks.

## Tests

The unit suite needs no ComfyUI installation:

```bash
python -m unittest discover -s tests -v
```

It covers strict configuration, mount safety and disappearance, concurrent
loads, busy-object protection, cache hits and invalidation, atomic-copy cleanup,
LRU eviction, suffix preservation, compatibility guards, transactional patching,
fail-open behavior, and exception preservation. An actual ComfyUI/GPU
end-to-end test remains a deployment verification step.
