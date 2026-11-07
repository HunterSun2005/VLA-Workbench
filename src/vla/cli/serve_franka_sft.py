"""Serve a PI05 SFT checkpoint to the Franka collector; no AC or training."""
from __future__ import annotations
import argparse, hashlib, pathlib, signal, socket
import numpy as np
from vla_rollout_protocol import recv_packet, send_packet

TASK = "Place the white lid firmly onto the jig."
STATE_NAMES = ["tcp_position_x","tcp_position_y","tcp_position_z","tcp_rot6d_col0_x","tcp_rot6d_col0_y","tcp_rot6d_col0_z","tcp_rot6d_col1_x","tcp_rot6d_col1_y","tcp_rot6d_col1_z","gripper_position"]
ACTION_NAMES = ["desired_tcp_position_x","desired_tcp_position_y","desired_tcp_position_z","desired_tcp_rot6d_col0_x","desired_tcp_rot6d_col0_y","desired_tcp_rot6d_col0_z","desired_tcp_rot6d_col1_x","desired_tcp_rot6d_col1_y","desired_tcp_rot6d_col1_z","gripper"]

def digest(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with (path/"model.safetensors").open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

class Server:
    def __init__(self,args):
        from vla.adapters.lerobot import register
        register()
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.torch=torch; self.prepare=prepare_observation_for_inference
        self.path=pathlib.Path(args.checkpoint).expanduser().resolve(); self.device=torch.device(args.device)
        cfg=PreTrainedConfig.from_pretrained(self.path,local_files_only=True); cfg.device=args.device
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        cfg.rtc_config = RTCConfig(enabled=True)
        from vla.actions.metadata import checkpoint_contract
        from vla.actions import ActionSpec
        self.action_contract = checkpoint_contract(self.path)
        self.spec = ActionSpec.from_dict(self.action_contract['action_spec'])
        self.frame_stride = self.action_contract['statistics']['contract']['sampling']['frame_stride']
        cfg.action_contract = self.action_contract
        self.policy=PI05Policy.from_pretrained(self.path,config=cfg).to(self.device).eval()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        self.pre,self.post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(self.path),preprocessor_overrides={"device_processor":{"device":args.device}, "tokenizer_processor": {"tokenizer_name": args.tokenizer_path, "tokenizer": tokenizer}},postprocessor_overrides={})
        self.fingerprint=digest(self.path); self.task=args.task
        self.tasks = tuple(getattr(args, "tasks", None) or [self.task])
        self.action_dim = self.spec.dim
        dual = self.spec.arms == ('left', 'right')
        if dual != (getattr(args, 'arms', 'single_left') == 'dual'):
            raise ValueError('checkpoint arms do not match selected arms')
        self.robot_type = 'franka_robotiq_dual' if dual else 'franka_robotiq'
        self.camera_keys = [key for key in cfg.input_features if key.startswith('observation.images.')]
        self.state_names = self.action_contract['state_fields']
        self.state_dim = len(self.state_names)
        if tuple(cfg.output_features['action'].shape) != (self.action_dim,) or tuple(cfg.input_features['observation.state'].shape) != (self.state_dim,):
            raise ValueError('checkpoint feature dimensions disagree with action contract')
        self.contract = {
            'checkpoint_fingerprint': self.fingerprint, 'state_dim': self.state_dim, 'action_dim': self.action_dim,
            'camera_keys': self.camera_keys, 'robot_type': self.robot_type, 'task': self.task, 'tasks': list(self.tasks),
            'fps': 30, 'state_names': self.state_names, 'action_names': list(self.spec.action_fields),
            'action_spec': self.spec.to_dict(), 'wire_action_representation': 'absolute',
            'actions_stored_as': 'absolute', 'use_relative_actions': self.spec.representation == 'chunk_relative',
            'camera_shapes_hwc': {key: [cfg.input_features[key].shape[1], cfg.input_features[key].shape[2], 3]
                                  for key in self.camera_keys},
            'control_modes': ['vla'], 'control_mode_protocol': 1, 'phase_mode': 'vla',
            'rtc_enabled': True, 'rtc_execution_horizon': cfg.rtc_config.execution_horizon,
            'rtc_queue_threshold': min(28, cfg.chunk_size-1), 'online_enabled': False}
        from vla.control.plan_service import PlanGate
        self.plan_gate = PlanGate()
        self.contract.update(plan_horizons={'vla': cfg.chunk_size}, frame_stride=self.frame_stride,
                             action_contract=self.action_contract)

    def handle(self, h, a):
        from vla.control.plan_service import control_message
        from vla.adapters.lerobot.plan_inference import infer_plan
        kind = h.get('kind')
        if kind == 'health':
            return {'kind': 'health', 'request_id': h.get('request_id', -1), **self.contract}, {}
        result = control_message(self, h)
        if result is not None:
            return result
        if kind != 'plan':
            raise ValueError('only protocol-v2 candidate plans are supported')
        return infer_plan(self, h, a, sft=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', choices=['single_left', 'dual'], default='single_left')
    p.add_argument('--checkpoint', required=True, type=pathlib.Path)
    p.add_argument('--socket-path', required=True, type=pathlib.Path)
    p.add_argument('--task', default=TASK)
    p.add_argument('--device', default='cuda')
    p.add_argument('--tokenizer-path', required=True)
    args = p.parse_args()
    from vla.web.policy_server import serve
    serve(Server(args), args.socket_path)


if __name__ == '__main__':
    main()
