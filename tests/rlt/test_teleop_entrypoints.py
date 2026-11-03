from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from vla.datacollection.start_teleop.check_controller_state import (
    controller_is_active,
    format_controllers,
)
from vla.datacollection.start_teleop.prepare_runtime_configs import prepare_runtime_configs


ROOT = Path(__file__).parents[2]
TELEOP_DIR = ROOT / "src/vla/datacollection/start_teleop"


def _backend_source() -> str:
    paths = [TELEOP_DIR / "start_teleop.sh", *sorted((TELEOP_DIR / "lib").glob("*.sh"))]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    gello = tmp_path / "gello.yaml"
    arms = tmp_path / "arms.yaml"
    grippers = tmp_path / "grippers.yaml"
    _write_yaml(
        gello,
        {
            "LEFT": {"namespace": "left", "com_port": "gello-left"},
            "RIGHT": {"namespace": "right", "com_port": "gello-right"},
        },
    )
    _write_yaml(
        arms,
        {
            "LEFT": {"robot_ip": "172.16.0.2"},
            "RIGHT": {"robot_ip": "172.16.1.2"},
        },
    )
    _write_yaml(
        grippers,
        {
            "LEFT": {"com_port": "/dev/ttyUSB-left"},
            "RIGHT": {"com_port": "/dev/ttyUSB-right"},
        },
    )
    return gello, arms, grippers


def _prepare(tmp_path: Path, mode: str) -> dict[str, Path]:
    gello_source, arm_source, gripper_source = _sources(tmp_path)
    outputs = {
        "gello": tmp_path / "runtime_gello.yaml",
        "arms": tmp_path / "runtime_arms.yaml",
        "grippers": tmp_path / "runtime_grippers.yaml",
        "speed": tmp_path / "runtime_speed.yaml",
    }
    prepare_runtime_configs(
        mode=mode,
        gello_source=gello_source,
        arm_source=arm_source,
        gripper_source=gripper_source,
        gello_output=outputs["gello"],
        arm_output=outputs["arms"],
        gripper_output=outputs["grippers"],
        gripper_speed_output=outputs["speed"],
        left_gello="override-left-gello",
        right_gello="override-right-gello",
        left_ip="10.0.0.1",
        right_ip="10.0.0.2",
        left_gripper="/dev/left-gripper",
        right_gripper="/dev/right-gripper",
    )
    return outputs


def test_single_left_runtime_configs_drop_all_right_entries(tmp_path: Path) -> None:
    outputs = _prepare(tmp_path, "single_left")
    for key in ("gello", "arms", "grippers"):
        assert list(yaml.safe_load(outputs[key].read_text())) == ["LEFT"]
    assert yaml.safe_load(outputs["gello"].read_text())["LEFT"]["com_port"] == (
        "override-left-gello"
    )
    assert yaml.safe_load(outputs["arms"].read_text())["LEFT"]["robot_ip"] == "10.0.0.1"
    assert yaml.safe_load(outputs["grippers"].read_text())["LEFT"]["com_port"] == (
        "/dev/left-gripper"
    )


def test_duo_runtime_configs_keep_both_sides(tmp_path: Path) -> None:
    outputs = _prepare(tmp_path, "duo")
    for key in ("gello", "arms", "grippers"):
        assert list(yaml.safe_load(outputs[key].read_text())) == ["LEFT", "RIGHT"]
    speed = yaml.safe_load(outputs["speed"].read_text())
    assert speed["/**"]["ros__parameters"]["joints"] == [
        "robotiq_85_left_knuckle_joint"
    ]


def test_rollout_runtime_configs_do_not_require_or_write_gello(tmp_path: Path) -> None:
    _, arm_source, gripper_source = _sources(tmp_path)
    gello_output = tmp_path / "must_not_exist.yaml"
    prepare_runtime_configs(
        mode="duo",
        control_owner="rollout",
        gello_source=None,
        arm_source=arm_source,
        gripper_source=gripper_source,
        gello_output=gello_output,
        arm_output=tmp_path / "runtime_arms.yaml",
        gripper_output=tmp_path / "runtime_grippers.yaml",
        gripper_speed_output=tmp_path / "runtime_speed.yaml",
    )
    assert not gello_output.exists()


def test_explicit_teleop_entrypoints_are_executable_and_syntax_valid() -> None:
    scripts = [
        TELEOP_DIR / "start_teleop.sh",
        TELEOP_DIR / "start_teleop_duo.sh",
        TELEOP_DIR / "start_teleop_single_left.sh",
        TELEOP_DIR / "start_rollout_duo.sh",
        TELEOP_DIR / "start_rollout_single_left.sh",
    ]
    for script in scripts:
        assert os.access(script, os.X_OK)
    for path in TELEOP_DIR.rglob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    assert "TELEOP_MODE=duo" in scripts[1].read_text()
    assert "TELEOP_MODE=single_left" in scripts[2].read_text()
    assert "CONTROL_OWNER=rollout" in scripts[3].read_text()
    assert "CONTROL_OWNER=rollout" in scripts[4].read_text()


def test_rollout_hardware_has_one_command_owner_and_per_arm_gamepads() -> None:
    backend = _backend_source()
    arm_launch = (TELEOP_DIR / "fr3_duo_teleop.launch.py").read_text(encoding="utf-8")
    gripper_launch = (TELEOP_DIR / "robotiq_duo_teleop.launch.py").read_text(
        encoding="utf-8"
    )
    assert 'if [[ "$CONTROL_OWNER" == "teleop" ]]; then' in backend
    assert 'add_task "${side}_rollout_gamepad"' in backend
    assert "-r __node:=${side}_rollout_joy" in backend
    assert "-r joy:=/${side}/gamepad/joy" in backend
    assert "requires different LEFT_JOY_DEVICE_ID" in backend
    assert "control_owner:=$CONTROL_OWNER" in backend
    assert 'dst="/right/teleop/joint_states"' in arm_launch
    assert 'dst="/right/teleop/gripper_action_width_percent"' in gripper_launch


def test_single_franky_rollout_entrypoint_only_starts_joy_bridge() -> None:
    entrypoint = (TELEOP_DIR / "start_rollout_single_left.sh").read_text(
        encoding="utf-8"
    )
    assert "franka_hardware" not in entrypoint
    assert "robotiq_driver" not in entrypoint
    assert "joy joy_node" in entrypoint
    assert "franka_robotiq_rollout.joy_bridge" in entrypoint
    source_setup = 'source "$ROS_SETUP"'
    assert entrypoint.index("set +u") < entrypoint.index(source_setup)
    assert entrypoint.index(source_setup) < entrypoint.index("set -u", entrypoint.index(source_setup))


def test_teleop_uses_orderly_keyboard_exit_without_forcing_hardware_inactive() -> None:
    backend = _backend_source()
    assert "c|$'\\e'" in backend
    assert "Interrupt requested; shutting down in order." in backend
    assert "set_hardware_component_state" not in backend


def test_teleop_exports_robotpkg_runtime_libraries_and_checks_franka_dependencies() -> None:
    backend = _backend_source()
    assert 'OPENROBOTS_LIB_DIR="${OPENROBOTS_LIB_DIR:-/opt/openrobots/lib}"' in backend
    assert "export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}" in backend
    assert "ldd $hardware_library_q" in backend
    assert "Franka hardware library has unresolved dependencies" in backend


def test_ros2_control_switch_uses_humble_float_timeout_default() -> None:
    backend = _backend_source()
    switch_commands = [
        line for line in backend.splitlines() if "ros2 control switch_controllers" in line
    ]
    assert switch_commands
    assert all("--switch-timeout" not in line for line in switch_commands)


def test_fine_gamepad_mapping_and_unbounded_cartesian_target() -> None:
    backend = _backend_source()
    fine_teleop = (TELEOP_DIR / "fine_teleop.py").read_text(encoding="utf-8")
    for mapping in (
        '"a_button": 0',
        '"b_button": 1',
        '"x_button": 2',
        '"y_button": 3',
    ):
        assert mapping in fine_teleop
    assert "max_xy_displacement" not in fine_teleop
    assert "max_z_displacement" not in fine_teleop
    assert 'FINE_JOINT7_SIGN="${FINE_JOINT7_SIGN:--1.0}"' in backend
    assert '"joint7_button_sign": -1.0' in fine_teleop


def test_keyboard_mode_switch_avoids_per_key_ros_cli_startup() -> None:
    backend = _backend_source()
    assert 'mode_signal="USR1"' in backend
    assert 'local mode_signal="USR2"' in backend
    assert 'KEYBOARD_POLL_SEC="${KEYBOARD_POLL_SEC:-0.02}"' in backend
    assert 'RETURN_JOINT_SPEED="${RETURN_JOINT_SPEED:-0.20}"' in backend
    assert "ros2 service call /left/left_fine_teleop/set_fine_mode" not in backend


def test_gripper_controller_state_uses_structured_name_and_state() -> None:
    class Controller:
        def __init__(self, name: str, state: str) -> None:
            self.name = name
            self.state = state
            self.type = "forward_command_controller/ForwardCommandController"

    controllers = [
        Controller("unrelated", "active"),
        Controller("/left/gripper/left_gripper_speed_controller", "active"),
    ]
    assert controller_is_active(controllers, "left_gripper_speed_controller")
    assert not controller_is_active(
        [Controller("left_gripper_speed_controller", "inactive")],
        "left_gripper_speed_controller",
    )
    assert "left_gripper_speed_controller" in format_controllers(controllers)


def test_rollout_starts_speed_controller_for_every_active_gripper() -> None:
    backend = _backend_source()
    assert 'controller="${side}_gripper_speed_controller"' in backend
    assert 'start_gripper_speed_controller "$side"' in backend
    assert "start_gripper_speed_controller left" in backend


@pytest.mark.parametrize("mode", ["single_left", "duo"])
@pytest.mark.parametrize("owner", ["teleop", "rollout"])
@pytest.mark.parametrize("check_only", ["0", "1"])
def test_launcher_orchestration_without_hardware(tmp_path, mode, owner, check_only):
    # Execute the real main/shutdown flow, substituting only hardware, process,
    # and filesystem operations. A foreign cwd also checks helper path resolution.
    script = r'''
source "$1"
event() { printf 'EVENT %s\n' "$*"; }
ensure_single_instance() { :; }
validate_environment() { :; }
prepare_runtime_configs() {
  GELLO_RUNTIME_CONFIG=gello.yaml
  ARM_RUNTIME_CONFIG=arms.yaml
  GRIPPER_RUNTIME_CONFIG=grippers.yaml
}
bash() { :; }
start_task() { TASK_PIDS[$1]=fake; event "start ${TASK_NAMES[$1]}"; }
task_is_alive() { [[ "$1" == fake ]]; }
wait_for_action() { event "action $1"; }
wait_for_service() { event "service $1"; }
start_gripper_speed_controller() { event "speed $1"; }
apply_collision_behavior() { event "collision $1"; }
monitor_with_keyboard() { event monitor-teleop; }
monitor_rollout_hardware() { event monitor-rollout; }
prepare_arm_shutdown() { event deactivate-arms; }
stop_task() { event "stop ${TASK_NAMES[$1]}"; TASK_PIDS[$1]=''; }
cleanup_runtime_files() { event cleanup; }
main
shutdown_all
shutdown_all
trap - EXIT INT TERM
'''
    env = {**os.environ, "TELEOP_MODE": mode, "CONTROL_OWNER": owner,
           "CHECK_ONLY": check_only, "START_CAMERAS": "1",
           "SET_FULL_COLLISION_BEHAVIOR": "1"}
    result = subprocess.run(
        ["bash", "-c", script, "_", str(TELEOP_DIR / "start_teleop.sh")],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    events = [line.removeprefix("EVENT ") for line in result.stdout.splitlines()
              if line.startswith("EVENT ")]
    if check_only == "1":
        assert events == ["cleanup"]
        return
    sides = ["left", "right"] if mode == "duo" else ["left"]
    inputs = ([f"gello_{mode}_fine", "xbox_joy"] if owner == "teleop"
              else [f"{side}_rollout_gamepad" for side in sides])
    tasks = [*inputs, "camera_base", *[f"camera_{side}_wrist" for side in sides],
             f"robotiq_{mode}", f"fr3_{mode}_controllers"]
    expected = [f"start {task}" for task in tasks[:-1]]
    expected += [f"action /{side}/gripper/robotiq_gripper_controller/gripper_cmd"
                 for side in sides]
    expected += [f"speed {side}" for side in (["left"] if owner == "teleop" else sides)]
    expected += [f"start {tasks[-1]}", *[f"collision {side}" for side in sides]]
    if owner == "teleop":
        tasks.append("left_fine_teleop")
        expected += ["start left_fine_teleop", "service /left/left_fine_teleop/set_fine_mode"]
    expected.append(f"monitor-{owner}")
    for task in reversed(tasks):
        if task == f"fr3_{mode}_controllers":
            expected.append("deactivate-arms")
        expected.append(f"stop {task}")
    assert events == [*expected, "cleanup"]


@pytest.mark.parametrize("owner,managed,check_only,accepted", [
    ("rollout", True, "0", True),
    ("rollout", False, "0", False),
    ("teleop", True, "0", False),
    ("teleop", False, "0", False),
    ("rollout", False, "1", True),
    ("teleop", False, "1", True),
])
def test_launcher_terminal_validation_for_managed_web(tmp_path, owner, managed, check_only, accepted):
    # Exercise real validation with /dev/null stdin, as used by Children.start.
    # Only ROS/library lookups are replaced; no ROS nodes or devices are opened.
    script = r'''
source "$1"
require_file() { :; }
bash() { :; }
OPENROBOTS_LIB_DIR="$PWD"
validate_environment
'''
    env = {**os.environ, "CONTROL_OWNER": owner, "TELEOP_MODE": "duo",
           "CHECK_ONLY": check_only, "LEFT_JOY_DEVICE_ID": "0", "RIGHT_JOY_DEVICE_ID": "1",
           "VLA_RUN_MODE": "managed" if managed and owner == "rollout" else "terminal",
           "VLA_WEB_READY_FILE": str(tmp_path / "hardware.ready") if managed else ""}
    result = subprocess.run(["bash", "-c", script, "_", str(TELEOP_DIR / "start_teleop.sh")],
                            cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=5)
    assert (result.returncode == 0) == accepted, result.stderr
    if not accepted:
        assert "An interactive terminal is required" in result.stderr


@pytest.mark.parametrize("failure", [False, True])
def test_managed_monitor_checks_children_and_exits_without_stdin(tmp_path, failure):
    script = r'''
source "$1"
trap on_exit EXIT
trap 'exit 143' TERM
read() { die "managed monitor must not read stdin"; }
assert_tasks_alive() {
  printf 'EVENT health-check\n'
  if [[ "$FAIL_CHILD" == 1 ]]; then die "mock child exited"; fi
}
sleep() { printf 'EVENT sleep\n'; kill -TERM "$$"; }
cleanup_runtime_files() { printf 'EVENT cleanup\n'; }
monitor_rollout_hardware
'''
    env = {**os.environ, "CONTROL_OWNER": "rollout", "TELEOP_MODE": "duo",
           "VLA_RUN_MODE": "managed",
           "VLA_WEB_READY_FILE": str(tmp_path / "hardware.ready"), "FAIL_CHILD": str(int(failure))}
    result = subprocess.run(["bash", "-c", script, "_", str(TELEOP_DIR / "start_teleop.sh")],
                            cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == (1 if failure else 143), result.stderr
    events = [line for line in result.stdout.splitlines() if line.startswith("EVENT ")]
    assert events == (["EVENT health-check", "EVENT cleanup"] if failure else
                      ["EVENT health-check", "EVENT sleep", "EVENT cleanup"])
    assert "must not read stdin" not in result.stderr
