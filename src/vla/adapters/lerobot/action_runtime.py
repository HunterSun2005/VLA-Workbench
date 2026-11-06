"""Explicit chunk context for normalized policy/reference/actor actions."""
import time
import numpy as np
import torch
from vla.actions import ActionSpec, ChunkContext, QuantileNormalizer
from vla.actions.chunks import ChunkCodec


class ActionRuntime:
    def __init__(self, contract):
        self.spec = ActionSpec.from_dict(contract['action_spec'])
        self.normalizer = QuantileNormalizer(contract['statistics'])

    def codec(self, batch, generation=0, request_id=0):
        anchor = batch.get('observation.anchor')
        if anchor is None:
            raise ValueError('inference requires raw observation.anchor from the request')
        if isinstance(anchor, torch.Tensor):
            anchor = anchor.detach().cpu().numpy()
        anchor = np.asarray(anchor)
        if anchor.shape == (1, self.spec.dim):
            anchor = anchor[0]
        return ChunkCodec(ChunkContext(self.spec, anchor, time.monotonic_ns(), generation, request_id), self.normalizer)

    def decode(self, values, batch):
        array = values.detach().cpu().numpy()
        represented = self.normalizer.unnormalize(array)
        represented[..., self.spec.width-1::self.spec.width] = np.clip(
            represented[..., self.spec.width-1::self.spec.width], 0, 1)
        from vla.actions import AbsoluteActions
        codec = self.codec(batch)
        absolute = AbsoluteActions(self.spec)(represented, codec.context.anchor)
        return torch.as_tensor(absolute, device=values.device, dtype=values.dtype)

    def encode(self, values, batch):
        result = self.codec(batch).encode(values.detach().cpu().numpy())
        return torch.as_tensor(result, device=values.device, dtype=values.dtype)
