"""External deployment contract v2. NumPy/stdlib only; no training contract conversion.

The LingBot service carries the same specification independently. Cross-repository
conformance tests protect this boundary; neither process imports the other repo.
"""
from __future__ import annotations

import hashlib
import json
import math
import socket
from urllib.parse import urlsplit

import numpy as np

STATE_FIELDS = ["tcp_position_x", "tcp_position_y", "tcp_position_z",
                "tcp_rot6d_col0_x", "tcp_rot6d_col0_y", "tcp_rot6d_col0_z",
                "tcp_rot6d_col1_x", "tcp_rot6d_col1_y", "tcp_rot6d_col1_z", "gripper_position"]
ACTION_FIELDS = ["desired_" + name for name in STATE_FIELDS[:-1]] + ["gripper"]
CAMERAS = ["observation.images.base", "observation.images.left_wrist"]
UNITS = {"translation": "metres", "rotation": "matrix_columns_0_1",
         "state_gripper": "metres", "action_gripper": "open_fraction_0_1"}
ACTION_SPEC = {"space": "tcp", "arms": ["left"], "representation": "absolute", "version": 1,
               "anchor_rule": "chunk_start_raw_measured_named_fields",
               "rotation": "columns_0_1_base_relative_left_multiply", "storage": "absolute"}
ARTIFACTS = ("weights", "training_config", "processor", "robot_mapping", "norm_stats")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def endpoint_address(endpoint):
    try:
        value = urlsplit(endpoint)
        if (value.scheme != "tcp" or not value.hostname or not value.port or value.username
                or value.password or value.path or value.query or value.fragment):
            raise ValueError()
        return value.hostname, value.port
    except (ValueError, TypeError) as error:
        raise ValueError("policy endpoint must be tcp://HOST:PORT") from error


def connect(endpoint, timeout_s):
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 300:
        raise ValueError("policy timeout must be in (0, 300] seconds")
    return socket.create_connection(endpoint_address(endpoint), timeout=timeout_s)


def validate_deployment(value, *, require_artifacts=True):
    expected = {"version": 2, "kind": "external_deployment", "model_backend": "lingbot_v2",
                "action_spec": ACTION_SPEC,
                "state_fields": STATE_FIELDS, "action_fields": ACTION_FIELDS, "units": UNITS,
                "camera_keys": CAMERAS, "horizon": 50, "fps": 30, "frame_stride": 1,
                "base_frame": "left_fr3_link0"}
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("incompatible external LingBot deployment contract")
    mode = value.get("execution_mode")
    if mode not in ("serial", "rtc"):
        raise ValueError("invalid LingBot execution mode")
    if mode == "rtc":
        rtc = value.get("rtc")
        if (not isinstance(rtc, dict) or set(rtc) != {"prefix_conditioning", "max_prefix_steps"}
                or rtc["prefix_conditioning"] != "clean_tokens_time_zero_training_mask"
                or type(rtc["max_prefix_steps"]) is not int or not 0 < rtc["max_prefix_steps"] < 50):
            raise ValueError("invalid LingBot RTC prefix contract")
    elif "rtc" in value:
        raise ValueError("serial deployment cannot declare RTC")
    shapes = value.get("camera_shapes_hwc", {})
    if set(shapes) != set(CAMERAS) or any(
        not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3
        or any(type(n) is not int or not 0 < n <= 4096 for n in shape) for shape in shapes.values()
    ):
        raise ValueError("invalid external RGB camera shapes")
    tools = value.get("tool_definition")
    if not isinstance(tools, dict) or set(tools) != {"left"}:
        raise ValueError("explicit left TCP tool definition is required")
    tool = tools["left"]
    if not isinstance(tool, dict) or tool.get("base") != value["base_frame"] or tool.get("tip") != "left_fr3_link8":
        raise ValueError("incorrect TCP base/tip frames")
    position = np.asarray(tool.get("position"), dtype=float)
    rotation = np.asarray(tool.get("rotation"), dtype=float)
    if (position.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(position).all() or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("invalid TCP transform")
    if require_artifacts:
        artifacts = value.get("artifacts", {})
        if set(artifacts) != set(ARTIFACTS) or any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in artifacts.values()
        ):
            raise ValueError("external deployment requires all artifact SHA256 fingerprints")
    return value


def health_contract(deployment, tasks):
    validate_deployment(deployment)
    if not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("nonempty unique task prompts are required")
    identity = digest(deployment)
    mode = deployment["execution_mode"]
    rtc = mode == "rtc"
    return {"deployment_contract": deployment, "model_backend": "lingbot_v2", "execution_mode": mode,
            "ready": True, "capabilities": {"serial": mode == "serial", "rtc": rtc, "actor": False, "online": False},
            "model_version": identity, "checkpoint_fingerprint": identity,
            "state_dim": 10, "action_dim": 10, "state_names": STATE_FIELDS, "action_names": ACTION_FIELDS,
            "action_spec": ACTION_SPEC, "camera_keys": CAMERAS,
            "camera_shapes_hwc": deployment["camera_shapes_hwc"], "robot_type": "franka_robotiq",
            "fps": 30, "frame_stride": 1, "plan_horizons": {"vla": 50},
            "task": tasks[0], "tasks": list(tasks), "control_modes": ["vla"],
            "control_mode_protocol": 1, "phase_mode": "vla", "rtc_enabled": rtc, "online_enabled": False,
            "wire_action_representation": "absolute", "actions_stored_as": "absolute",
            "use_relative_actions": False}


def validate_health(header, *, tasks=None, image_size=None, execution_mode=None,
                    inference_slots=14, guard_slots=1):
    deployment = validate_deployment(header.get("deployment_contract"))
    expected = health_contract(deployment, header.get("tasks"))
    if any(header.get(k) != v for k, v in expected.items()):
        raise ValueError("external service health disagrees with its deployment contract")
    if tasks is not None and list(tasks) != header["tasks"]:
        raise ValueError("external service task prompts do not match this session")
    if image_size is not None and any(shape != [*image_size, 3] for shape in header["camera_shapes_hwc"].values()):
        raise ValueError("external service camera dimensions do not match this session")
    if execution_mode is not None:
        if header["execution_mode"] != execution_mode:
            raise ValueError("LingBot execution mode mismatch; no automatic fallback")
        if execution_mode == "rtc":
            validate_rtc_budget(inference_slots, guard_slots, deployment["rtc"]["max_prefix_steps"])
    return header


def validate_rtc_budget(inference_slots, guard_slots, max_prefix_steps=49):
    if any(type(v) is not int or v < 1 for v in (inference_slots, guard_slots)):
        raise ValueError("LingBot RTC budgets must be positive integer slots")
    d = inference_slots + guard_slots
    if d > max_prefix_steps or 50 < 2*d:
        raise ValueError("LingBot RTC requires d <= rtc_max_delay and 50 >= 2d")
    return d


def require_tool(deployment, actual):
    validate_deployment(deployment)
    expected = deployment["tool_definition"]
    if set(actual) != set(expected):
        raise ValueError("external deployment TCP arms mismatch")
    for arm, tool in expected.items():
        other = actual[arm]
        if (any(tool[k] != other.get(k) for k in ("base", "tip"))
                or any(np.asarray(tool[k]).shape != np.asarray(other.get(k)).shape
                       or not np.allclose(tool[k], other[k], atol=1e-6, rtol=0)
                       for k in ("position", "rotation"))):
            raise ValueError("external deployment TCP transform mismatch")
