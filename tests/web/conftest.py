from dataclasses import replace
import json
import pytest

@pytest.fixture
def ros_ready(monkeypatch):
    """Configuration-only ROS installation; never source or start these placeholders."""
    from vla.core import franky_runtime
    monkeypatch.setattr(franky_runtime, "check_worker", lambda root, configured=None: (root/"fake-python", {"libfranka": "0.21.2"}))
    def configure(root, config):
        from vla.web.hardware import preset
        workspace = root/'runtime/ros/workspaces/test'
        target = workspace/'install/local_setup.bash'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# offline fixture\n')
        pointer = root/'runtime/ros/current'
        if not pointer.exists():
            pointer.symlink_to(workspace, target_is_directory=True)
        hardware = root/'hardware.json'
        hardware.write_text(json.dumps(preset(config.arms)))
        return replace(config, ros_config=str(hardware))
    return configure
