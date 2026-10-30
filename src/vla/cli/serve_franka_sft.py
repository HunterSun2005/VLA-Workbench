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
        import lerobot.policies.pi05.configuration_pi05  # register PI05
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
        cfg.rtc_config = None
        self.policy=PI05Policy.from_pretrained(self.path,config=cfg).to(self.device).eval()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        self.pre,self.post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(self.path),preprocessor_overrides={"device_processor":{"device":args.device}, "tokenizer_processor": {"tokenizer_name": args.tokenizer_path, "tokenizer": tokenizer}},postprocessor_overrides={})
        self.fingerprint=digest(self.path); self.task=args.task
        self.contract={"checkpoint_fingerprint":self.fingerprint,"state_dim":10,"action_dim":10,"camera_keys":["observation.images.base","observation.images.left_wrist"],"camera_shapes_hwc":{"observation.images.base":[224,224,3],"observation.images.left_wrist":[224,224,3]},"robot_type":"franka_robotiq","task":self.task,"tasks":[self.task],"fps":30,"state_names":STATE_NAMES,"action_names":ACTION_NAMES,"actions_stored_as":"absolute","use_relative_actions":False,"phase_mode":"vla","rtc_enabled":True,"rtc_profile_path":"local-pi05","rtc_execution_horizon":10,"rtc_queue_threshold":28,"online_enabled":False}
        self.contract.update(rtc_enabled=False, inference_mode="sft_action_queue")
        dual = getattr(args, "arms", "single_left") == "dual"
        self.action_dim = 20 if dual else 10
        self.robot_type = "franka_robotiq_dual" if dual else "franka_robotiq"
        self.camera_keys = ["observation.images." + role for role in
                            (["base", "left_wrist", "right_wrist"] if dual else ["base", "left_wrist"])]
        if tuple(cfg.output_features["action"].shape) != (self.action_dim,):
            raise ValueError("SFT action dimension does not match selected arms")
        if tuple(cfg.input_features["observation.state"].shape) != (self.action_dim,):
            raise ValueError("SFT state dimension does not match selected arms")
        if {key for key in cfg.input_features if key.startswith("observation.images.")} != set(self.camera_keys):
            raise ValueError("SFT camera contract does not match selected arms")
        self.contract.update(state_dim=self.action_dim, action_dim=self.action_dim, robot_type=self.robot_type,
            camera_keys=self.camera_keys,
            camera_shapes_hwc={key: [cfg.input_features[key].shape[1], cfg.input_features[key].shape[2], 3]
                               for key in self.camera_keys},
            state_names=[f"{arm}_{name}" for arm in ("left", "right") for name in STATE_NAMES] if dual else STATE_NAMES,
            action_names=[f"{arm}_{name}" for arm in ("left", "right") for name in ACTION_NAMES] if dual else ACTION_NAMES,
            control_modes=["vla"], control_mode_protocol=1)
        self.mode_version = -1

    def handle(self,h,a):
        kind=h.get("kind"); rid=int(h.get("request_id",-1))
        if kind=="health": return {"kind":"health","request_id":rid,**self.contract},{}
        if kind!="infer": raise ValueError(f"unsupported request kind: {kind!r}")
        if h.get("task",self.task)!=self.task: raise ValueError("task mismatch")
        if h.get("phase", "vla") != "vla": raise ValueError("SFT server only supports vla phase")
        managed = h.get("control_mode")
        if managed is not None:
            version = int(h.get("mode_version", -1))
            if managed != "vla" or version < 0 or version < self.mode_version:
                raise ValueError("invalid or stale SFT control mode")
            if version != self.mode_version:
                self.policy.reset()
            self.mode_version = version
        if h.get("reset_episode", False): self.policy.reset()
        obs={"observation.state":np.asarray(a["observation.state"],dtype=np.float32),
             **{key: np.asarray(a[key]) for key in self.camera_keys}}
        if obs["observation.state"].shape != (self.action_dim,) or not np.isfinite(obs["observation.state"]).all():
            raise ValueError("invalid SFT state")
        for key in self.camera_keys:
            if list(obs[key].shape) != self.contract["camera_shapes_hwc"][key] or obs[key].dtype != np.uint8:
                raise ValueError("invalid SFT image shape/type")
        with self.torch.inference_mode():
            x=self.prepare(obs,self.device,self.task,self.robot_type); x=self.pre(x); y=self.post(self.policy.select_action(x))
        y=np.asarray(y.detach().cpu() if hasattr(y,"detach") else y,dtype=np.float32); y=y.reshape(-1,self.action_dim)[0]
        return {"kind":"action","request_id":rid,"checkpoint_fingerprint":self.fingerprint,"requested_phase":"vla","source_type":0.0,"phase":0.0,
                **({"control_mode":"vla","mode_version":self.mode_version} if managed else {})},{"action":y}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True,type=pathlib.Path); p.add_argument("--socket-path",required=True,type=pathlib.Path); p.add_argument("--task",default=TASK); p.add_argument("--device",default="cuda"); p.add_argument("--tokenizer-path", required=True); a=p.parse_args(); s=Server(a); stop=False
    def sig(*_):
        nonlocal stop; stop=True
    signal.signal(signal.SIGINT,sig); signal.signal(signal.SIGTERM,sig); a.socket_path.parent.mkdir(parents=True,exist_ok=True)
    if a.socket_path.exists(): a.socket_path.unlink()
    srv=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); srv.bind(str(a.socket_path)); srv.listen(1); srv.settimeout(.5); print(f"PI05 SFT server ready: {a.socket_path}",flush=True)
    try:
        while not stop:
            try: c,_=srv.accept()
            except socket.timeout: continue
            with c:
                try:
                    h,arr=recv_packet(c); rh,ra=s.handle(h,arr); send_packet(c,rh,ra)
                except Exception as e: send_packet(c,{"kind":"error","request_id":int(h.get("request_id",-1)),"error_type":type(e).__name__,"error":str(e)})
    finally:
        srv.close(); a.socket_path.unlink(missing_ok=True)
if __name__=="__main__": main()
