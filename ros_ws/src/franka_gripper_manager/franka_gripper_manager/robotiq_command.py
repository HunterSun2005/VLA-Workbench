def width_percent_to_joint_position(
    width_percent: float, open_position: float, closed_position: float
) -> float:
    """Map GELLO width (0 closed, 1 open) to a Robotiq joint position."""
    width_percent = min(1.0, max(0.0, width_percent))
    return closed_position + width_percent * (open_position - closed_position)


def snap_width_percent_to_endpoints(width_percent: float, threshold: float) -> float:
    """Clamp width to 0-1 and snap values near either endpoint."""
    width_percent = min(1.0, max(0.0, width_percent))
    if width_percent <= threshold:
        return 0.0
    if width_percent >= 1.0 - threshold:
        return 1.0
    return width_percent
