"""Managed command wire format shared by production collection and ROS validation."""
from sensor_msgs.msg import JointState


def joint_command(names, joints, generation, command_id, sequence, stamp):
    message = JointState()
    message.header.stamp = stamp
    message.header.frame_id = f'vla:{generation}:{command_id}:{sequence}'
    message.name = list(names)
    message.position = list(joints)
    return message


def controller_identity(message):
    marker, generation, command_id, fault = message.header.frame_id.split(':')
    if marker != 'vla' or fault not in ('0', '1'):
        raise ValueError('invalid controller diagnostic identity')
    return int(generation), int(command_id), fault == '1'
