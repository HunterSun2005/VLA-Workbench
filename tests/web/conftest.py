from dataclasses import replace
import json
import pytest

@pytest.fixture
def ros_ready():
    """Configuration-only ROS installation; never source or start these placeholders."""
    def configure(root, config):
        from vla.web.hardware import preset
        target = root/'ros_ws/install/local_setup.bash'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# offline fixture\n')
        hardware = root/'hardware.json'
        hardware.write_text(json.dumps(preset(config.arms)))
        return replace(config, ros_config=str(hardware))
    return configure
