"""Version/API validation must never instantiate a robot or open IPC."""
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla.control.franky import binding, worker_main


@pytest.fixture
def fake_binding(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("binding preflight attempted to open hardware")

    class Tracker:
        __module__ = "franky.tracker"
        set_target = set_gains = stop = forbidden

    franky = ModuleType("franky")
    franky.Robot = forbidden
    franky.RealtimeConfig = object()
    franky.Affine = lambda matrix: SimpleNamespace(translation=[0., 0., 0.], quaternion=[0., 0., 0., 1.])
    franky.CartesianVelocityMotion = lambda *a: object()
    franky.CartesianVelocityStopMotion = lambda: object()
    franky.Twist = lambda *a: object()
    franky.RobotVelocity = lambda *a: object()
    franky.Duration = lambda *a: object()
    native = ModuleType("franky._franky")
    tracker = ModuleType("franky.tracker")
    for module, filename in [(native, "_franky.so"), (tracker, "tracker.py")]:
        path = tmp_path / filename
        path.write_bytes(filename.encode())
        module.__file__ = str(path)
    for module in (franky, native, tracker):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(binding.importlib.metadata, "version", lambda _: "2.0.0+libfranka.0.21.2")
    return franky


def test_verified_library_and_implementation_hashes(fake_binding, monkeypatch, tmp_path):
    library = tmp_path / "libfranka-f9564df3.so.0.21.2"
    library.write_bytes(b"mock libfranka 0.21.2")
    monkeypatch.setattr(binding, "_loaded_libfranka_paths", lambda: {library})
    result = binding.check_binding()
    assert result["libfranka"] == "0.21.2"
    assert result["libfranka_library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert result["binding"].endswith("_franky.so")
    assert len(result["binding_sha256"]) == 64
    assert result["affine_check"] == "passed"


def test_affine_conversion_failure_rejected(fake_binding, monkeypatch, tmp_path):
    library = tmp_path / "libfranka.so.0.21.2"
    monkeypatch.setattr(binding, "_loaded_libfranka_paths", lambda: {library})
    fake_binding.Affine = lambda matrix: SimpleNamespace(translation=[1., 0., 0.], quaternion=[0., 0., 0., 1.])
    with pytest.raises(RuntimeError, match="NumPy/Affine binding incompatible"):
        binding.check_binding()


@pytest.mark.parametrize("names", [[], ["libfranka.so.0.18.0"], ["libfranka.so.0.20.4"],
                                    ["libfranka.so"], ["libfranka.so.0.19"],
                                    ["libfranka.so.0.21.2", "libfranka.so.0.20.4"]])
def test_unknown_wrong_or_mixed_versions_rejected(fake_binding, monkeypatch, tmp_path, names):
    monkeypatch.setattr(binding, "_loaded_libfranka_paths", lambda: {tmp_path / n for n in names})
    with pytest.raises(RuntimeError, match="libfranka 0.21.2"):
        binding.check_binding()


def test_missing_velocity_api_rejected(fake_binding):
    del fake_binding.CartesianVelocityMotion
    with pytest.raises(RuntimeError, match='lacks required API: CartesianVelocityMotion'):
        binding.check_binding()


def test_wrong_package_release_rejected(fake_binding, monkeypatch, tmp_path):
    library = tmp_path/'libfranka.so.0.21.2'
    library.write_bytes(b'library')
    monkeypatch.setattr(binding, '_loaded_libfranka_paths', lambda: {library})
    monkeypatch.setattr(binding.importlib.metadata,'version',lambda _: '1.1.4')
    with pytest.raises(RuntimeError,match='official Franky'):
        binding.check_binding()


def test_maps_resolves_symlink_and_deduplicates_segments(monkeypatch, tmp_path):
    library = tmp_path / "libfranka.so.0.21.2"
    library.touch()
    alias = tmp_path / "libfranka.so.0.18"
    alias.symlink_to(library)
    maps = "\n".join(f"100-200 r-xp 0000 08:01 123 {p}" for p in
                     [library, alias, library, tmp_path / "libfranka-common.so"])
    monkeypatch.setattr(binding.Path, "read_text", lambda _: maps)
    assert binding._loaded_libfranka_paths() == {library}


def test_deleted_loaded_library_rejected(monkeypatch):
    monkeypatch.setattr(binding.Path, "read_text", lambda _:
                        "100-200 r-xp 0000 08:01 123 /tmp/libfranka.so.0.21.2 (deleted)")
    with pytest.raises(RuntimeError, match="deleted"):
        binding._loaded_libfranka_paths()


def test_check_only_needs_no_config_and_opens_no_worker(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["worker_main", "--check-only"])
    monkeypatch.setattr(worker_main, "check_binding", lambda: {"libfranka": "0.21.2"})
    monkeypatch.setattr(worker_main, "FrankyWorker", lambda *a, **k: pytest.fail("worker opened"))
    worker_main.main()
    assert json.loads(capsys.readouterr().out) == {"libfranka": "0.21.2"}


def test_normal_start_checks_binding_before_reading_config(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["worker_main", "--config", str(tmp_path / "absent"),
                                     "--socket-path", str(tmp_path / "worker.sock")])
    def fail():
        raise RuntimeError("libfranka mismatch")
    monkeypatch.setattr(worker_main, "check_binding", fail)
    with pytest.raises(RuntimeError, match="libfranka mismatch"):
        worker_main.main()
    assert not (tmp_path / "worker.sock").exists()
