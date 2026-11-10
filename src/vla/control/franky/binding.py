"""Fail-closed, hardware-free checks for the isolated Linux Franky process."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from pathlib import Path
import re

REQUIRED_LIBFRANKA_VERSION = "0.18.0"


def _loaded_libfranka_paths() -> set[Path]:
    # Inspect the loaded objects, not ldd's prediction or a package version
    # string. In particular auditwheel wheels carry a hashed library name.
    result = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        name = fields[5]
        deleted = name.endswith(" (deleted)")
        path = Path(name.removesuffix(" (deleted)"))
        if not re.fullmatch(r"libfranka(?:-[0-9a-f]+)?\.so(?:\.\d+)*", path.name):
            continue
        if deleted:
            raise RuntimeError("Loaded libfranka was deleted; restart with an intact worker environment")
        result.add(path.resolve(strict=True))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_binding() -> dict[str, str]:
    """Verify the API and actual loaded libfranka before constructing Robot."""
    try:
        franky = importlib.import_module("franky")
        native = importlib.import_module("franky._franky")
    except Exception as error:
        raise RuntimeError(f"Franky binding unavailable: {error}") from error
    required = ("Robot", "RealtimeConfig", "CartesianImpedanceTracker", "Affine")
    missing = [name for name in required if not hasattr(franky, name)]
    if missing:
        raise RuntimeError("Franky binding lacks required API: " + ", ".join(missing))
    for method in ("set_target", "set_gains", "stop"):
        if not callable(getattr(franky.CartesianImpedanceTracker, method, None)):
            raise RuntimeError(f"Franky tracker lacks required API: {method}")
    try:
        libraries = _loaded_libfranka_paths()
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Cannot verify loaded libfranka: {error}") from error
    if len(libraries) != 1:
        raise RuntimeError(
            f"Expected exactly one loaded libfranka {REQUIRED_LIBFRANKA_VERSION}; "
            f"found {len(libraries)}: {sorted(map(str, libraries))}")
    library, = libraries
    if not library.name.endswith(f".so.{REQUIRED_LIBFRANKA_VERSION}"):
        raise RuntimeError(
            f"Worker requires libfranka {REQUIRED_LIBFRANKA_VERSION}; loaded {library}")
    # Import alone does not exercise pybind11's NumPy/Eigen conversion. Older
    # bindings can import successfully yet crash on the first TCP target.
    # Exercise this before Robot creation (also in the --check-only process).
    import numpy as np
    try:
        affine = franky.Affine(np.eye(4))
        if not (np.allclose(affine.translation, [0., 0., 0.]) and
                np.allclose(affine.quaternion, [0., 0., 0., 1.])):
            raise ValueError("Affine identity roundtrip failed")
    except Exception as error:
        raise RuntimeError(f"Franky NumPy/Affine binding incompatible: {error}") from error
    try:
        version = importlib.metadata.version("franky-control")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(franky, "__version__", "unknown"))
    # Track the native extension and Python tracker separately: forks can
    # expose different APIs under the same franky-control package version.
    tracker = importlib.import_module(franky.CartesianImpedanceTracker.__module__)
    files = {"binding": Path(native.__file__).resolve(strict=True),
             "tracker": Path(tracker.__file__).resolve(strict=True),
             "libfranka_library": library}
    result = {"franky": version, "libfranka": REQUIRED_LIBFRANKA_VERSION,
              "numpy": np.__version__, "affine_check": "passed"}
    for key, path in files.items():
        result[key] = str(path)
        result[f"{key}_sha256"] = _sha256(path)
    return result
