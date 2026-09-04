"""Domain exceptions raised by the disk-cache extension."""


class DiskCacheError(Exception):
    """Base class for expected disk-cache failures."""


class ConfigurationError(DiskCacheError):
    """The extension configuration is invalid."""


class CompatibilityError(DiskCacheError):
    """The running ComfyUI internals do not match the guarded contract."""


class CacheUnavailableError(DiskCacheError):
    """The configured cache filesystem is unavailable or unsafe to use."""


class CacheCapacityError(DiskCacheError):
    """The cache cannot make enough room for an incoming file."""


class SourceChangedError(DiskCacheError):
    """A source file changed while it was being copied."""


class IndexVersionError(DiskCacheError):
    """The cache index uses an unsupported schema version."""
