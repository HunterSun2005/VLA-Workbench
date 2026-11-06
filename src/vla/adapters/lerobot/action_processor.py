"""Stateless LeRobot preprocessing; inference inversion requires an explicit context."""
from dataclasses import dataclass
import numpy as np
import torch
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from vla.actions import ActionSpec, DeltaActions, AbsoluteActions


def transform_tensor(operation, values, anchor):
    is_tensor = isinstance(values, torch.Tensor)
    raw = values.detach().cpu().numpy() if is_tensor else np.asarray(values)
    origin = anchor.detach().cpu().numpy() if isinstance(anchor, torch.Tensor) else np.asarray(anchor)
    result = operation(raw, origin)
    return torch.as_tensor(result, dtype=values.dtype, device=values.device) if is_tensor else result


@ProcessorStepRegistry.register(name='vla_delta_actions')
@dataclass
class DeltaActionsProcessor(ProcessorStep):
    action_spec: dict

    def __call__(self, transition):
        result = transition.copy()
        action = result.get(TransitionKey.ACTION)
        if action is None:
            return result
        spec = ActionSpec.from_dict(self.action_spec)
        observation = result.get(TransitionKey.OBSERVATION, {})
        anchor = observation.get('observation.anchor')
        if anchor is None:
            raise ValueError('raw observation.anchor is required independently of selected model state')
        result[TransitionKey.ACTION] = transform_tensor(DeltaActions(spec), action, anchor)
        return result

    def transform_features(self, features):
        return features

    def get_config(self):
        return {'action_spec': self.action_spec}


class AbsoluteActionsProcessor:
    """An immutable request-bound output adapter, never a last-observation cache."""
    def __init__(self, postprocessor, context):
        self.postprocessor, self.context = postprocessor, context

    def __call__(self, normalized):
        represented = self.postprocessor(normalized)
        return transform_tensor(AbsoluteActions(self.context.spec), represented, self.context.anchor)


def adapt_processors(pre, post, spec):
    from lerobot.processor import RelativeActionsProcessorStep, AbsoluteActionsProcessorStep
    pre.steps = [step for step in pre.steps if not isinstance(step, (RelativeActionsProcessorStep, DeltaActionsProcessor))]
    post.steps = [step for step in post.steps if not isinstance(step, AbsoluteActionsProcessorStep)]
    # Insert after batching, before normalization; the anchor is never normalized.
    index = next(i for i, step in enumerate(pre.steps) if type(step).__name__ == 'NormalizerProcessorStep')
    pre.steps.insert(index, DeltaActionsProcessor(spec.to_dict()))
    return pre, post
