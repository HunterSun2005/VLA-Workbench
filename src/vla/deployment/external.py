"""Connection and preflight for independently managed inference services."""
from vla_rollout_protocol import recv_packet, send_packet
from vla_rollout_protocol.external import connect, validate_health


def inspect_service(endpoint, timeout_s=5.0, *, tasks=None, image_size=None, execution_mode=None,
                    inference_slots=14, guard_slots=1):
    with connect(endpoint, timeout_s) as connection:
        send_packet(connection, {'kind': 'health', 'request_id': -1})
        header, arrays = recv_packet(connection)
    if arrays or header.get('kind') != 'health' or header.get('request_id') != -1:
        raise ValueError('invalid external service health response')
    return validate_health(header, tasks=tasks, image_size=image_size, execution_mode=execution_mode,
                           inference_slots=inference_slots, guard_slots=guard_slots)
