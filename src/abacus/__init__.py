from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abacus")
except PackageNotFoundError:
    __version__ = "unknown"
