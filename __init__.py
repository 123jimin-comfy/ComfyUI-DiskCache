"""Entry point loaded by ComfyUI's custom-node discovery."""


async def comfy_entrypoint():
    from .disk_cache.extension import DiskCacheExtension

    return DiskCacheExtension()


__all__ = ["comfy_entrypoint"]
