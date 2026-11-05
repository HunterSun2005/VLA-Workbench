"""ROS gripper recording/drive separation; safe to import without ROS or Torch."""

ROS_GRIPPER_RECORDING_VERSION = 2


def gello_recording_metadata(topics: dict[str, str]) -> dict:
    return {
        "gripper_recording_version": ROS_GRIPPER_RECORDING_VERSION,
        "gripper_command_topics": dict(topics),
        "action_gripper_source": "gello_reference_or_fine_submitted_goal",
        "action_gripper_unit": "open_fraction",
        "state_gripper_unit": "metres",
    }


def gripper_drive_topic(command_topic: str) -> str:
    """Keep recording targets off the ActionClient's GELLO/policy drive input."""
    suffix = "/gripper_action_width_percent"
    if command_topic.endswith(suffix):
        return command_topic[:-len(suffix)] + "/gripper_drive_width_percent"
    return command_topic.rstrip("/") + "/drive"


def gripper_consumer_requirement(command_topic: str) -> tuple[str, int]:
    """Recording heartbeat is separate; legacy raw GELLO has a recorder too."""
    if command_topic.endswith("/gello/gripper_width_percent"):
        return command_topic, 2
    return gripper_drive_topic(command_topic), 1
