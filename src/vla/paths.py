"""Package resources and explicitly configured writable workspace paths."""
import os
from pathlib import Path
from importlib.resources import files


def workspace() -> Path:
    return Path(os.environ.get("VLA_ROOT", Path.cwd())).expanduser().resolve()


def resource(name: str) -> Path:
    # Standard wheel installations unpack resources next to the package.
    return Path(str(files("vla").joinpath(name)))


def python_root() -> Path:
    return resource("").parent
