"""Serial TCP model stub for the fixed-state ROS harness. Never loads hardware/models."""
import json
from pathlib import Path
import socketserver
import threading
import time

import numpy as np

from vla.actions import ActionSpec
from vla.control.execution import PlanResponse
from vla.control.serial_execution import SerialExecutionContract, SerialExecutor
from vla.datacollection.franka_robotiq_rollout.policy_client import ExternalPolicyClient
from vla_rollout_protocol import recv_packet, send_packet
from vla_rollout_protocol.external import health_contract


class SerialValidation:
    def __init__(self, epoch, anchor, *, rtc=False):
        self.rtc = rtc
        fixture = json.loads((Path(__file__).resolve().parents[1]/'tests/fixtures/lingbot_wire_v2.json').read_text())
        if rtc:
            fixture['deployment']['execution_mode'] = 'rtc'
            fixture['deployment']['rtc'] = {'prefix_conditioning': 'clean_tokens_time_zero_training_mask',
                                            'max_prefix_steps': 15}
        health = health_contract(fixture['deployment'], fixture['tasks'])
        self.health = health
        class Handler(socketserver.BaseRequestHandler):
            def handle(handler):
                handler.request.settimeout(5)
                header, arrays = recv_packet(handler.request)
                if header['kind'] == 'health':
                    send_packet(handler.request, {**health, 'kind': 'health', 'request_id': header['request_id']})
                elif header['kind'] in ('invalidate', 'progress'):
                    send_packet(handler.request, {'kind': header['kind'], 'request_id': header.get('request_id', -1)})
                else:
                    from vla.control.execution import PlanRequest
                    request = PlanRequest.from_packet(header, arrays)
                    time.sleep(.06)  # Model-worker latency only; the ROS tick never sleeps here.
                    actions = np.tile(request.anchor, (50, 1))
                    actions[:, 0] += .001*np.sin(np.arange(50)/50*2*np.pi)
                    actions[:request.start_idx] = request.prefix
                    result = PlanResponse(request, actions, ('vla',)*50, (0,)*50, actions)
                    send_packet(handler.request, *result.packet())
        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
        self.server = Server(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = ExternalPolicyClient('tcp://%s:%s' % self.server.server_address, 5)
        self.client.set_control_mode('vla')
        self.executor = SerialExecutor(SerialExecutionContract(ActionSpec(), 'mock-control', 'mock-urdf', 'mock-tool'),
                                       session_id='mock-serial', model_version=health['model_version'], epoch_ns=epoch)
        if rtc:
            from vla.control.lingbot_execution import LingBotRTCContract, LingBotRTCExecutor
            self.executor = LingBotRTCExecutor(
                LingBotRTCContract(ActionSpec(), {'vla': 50}, 'mock-control', 'mock-urdf', 'mock-tool'),
                session_id='mock-lingbot-rtc', model_version=health['model_version'], epoch_ns=epoch)
        self.executor.enable(anchor, episode_id='mock-episode')
        self.hold_slots = self.action_slots = self.requests = self.completed_chunks = 0
        self.continuations = self.executed_while_inflight = 0
        self.last_plan = None
        self.images = {key: np.zeros(shape, dtype=np.uint8) for key, shape in health['camera_shapes_hwc'].items()}

    def step(self, slot, now_ns, anchor):
        self.client.raise_if_failed()
        candidate = self.client.poll_latest()
        if candidate is not None:
            if not self.executor.install(candidate, now_ns, **({} if self.rtc else {'start_slot': slot})):
                raise RuntimeError('serial mock candidate rejected')
        request = self.executor.request(slot=slot, anchor=anchor, observation_id=slot,
                                        observed_at_ns=now_ns, policy='vla')
        if request is not None:
            state = anchor.copy()
            state[-1] *= .085  # Measured gripper state is metres; anchor is a fraction.
            self.client.submit(plan=request, state=state, images=self.images,
                               task=self.health['task'], phase='vla')
            self.requests += 1
        record = self.executor.advance(slot, now_ns)
        if record is None:
            raise RuntimeError('unexpected serial mock exhaustion/fault')
        if record.source == 'system_hold':
            self.hold_slots += 1
        else:
            self.action_slots += 1
            if self.executor.inflight is not None:
                self.executed_while_inflight += 1
            if self.rtc and record.plan_version != self.last_plan:
                self.continuations += int(self.last_plan is not None)
                self.last_plan = record.plan_version
            if self.rtc:
                assert record.slot_id >= record.action_index  # Original fixed slot grid.
        self.executor.events.clear()
        return record

    def acknowledge(self, record):
        if self.rtc:
            return
        before = self.executor._awaiting_final_ack
        self.executor.acknowledge(record)
        if before and not self.executor._awaiting_final_ack:
            self.completed_chunks += 1

    def report(self):
        return {name: getattr(self, name) for name in ('hold_slots', 'action_slots', 'requests', 'completed_chunks', 'continuations', 'executed_while_inflight')}

    def close(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
