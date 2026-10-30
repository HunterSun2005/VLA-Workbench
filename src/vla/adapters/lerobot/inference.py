"""Single-step inference using LeRobot's public processing interfaces."""
from contextlib import nullcontext
import torch
from lerobot.policies.utils import prepare_observation_for_inference


def predict_action(observation, policy, device, preprocessor, postprocessor, use_amp, task=None, robot_type=None):
    # Keep the caller's ndarray dictionary intact and preserve the batch dimension.
    device = torch.device(device)
    precision = torch.autocast(device_type="cuda") if use_amp and device.type == "cuda" else nullcontext()
    with torch.inference_mode(), precision:
        batch = prepare_observation_for_inference(dict(observation), device, task, robot_type)
        return postprocessor(policy.select_action(preprocessor(batch)))
