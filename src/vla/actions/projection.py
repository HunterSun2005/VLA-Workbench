"""Deployment projection and its differentiable training equivalent."""
import numpy as np
from .spec import ActionSpec
from .statistics import QuantileNormalizer
from .transforms import AbsoluteActions, DeltaActions, matrix_to_rot6d, rot6d_to_matrix


class ActionProjection:
    def __init__(self, contract):
        self.spec = ActionSpec.from_dict(contract['action_spec'])
        self.normalizer = QuantileNormalizer(contract['statistics'])
        projection = contract['execution_projection']
        self.lower = np.asarray(projection['lower'], dtype=np.float64)
        self.upper = np.asarray(projection['upper'], dtype=np.float64)
        count = len(self.spec.arms)*(7 if self.spec.space == 'joint' else 3)
        if (self.lower.shape != (count,) or self.upper.shape != (count,)
                or not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper)):
            raise ValueError('invalid effective execution projection bounds')

    def numpy(self, normalized, anchor):
        represented = self.normalizer.unnormalize(np.clip(normalized, -1, 1))
        spec = self.spec
        represented[..., spec.width-1::spec.width] = np.clip(represented[..., spec.width-1::spec.width], 0, 1)
        absolute = AbsoluteActions(spec)(represented, anchor)
        width = 7 if spec.space == 'joint' else 3
        for i, offset in enumerate(range(0, spec.dim, spec.width)):
            absolute[..., offset:offset+width] = np.clip(absolute[..., offset:offset+width],
                                                       self.lower[i*width:(i+1)*width], self.upper[i*width:(i+1)*width])
            if spec.space == 'tcp':
                absolute[..., offset+3:offset+9] = matrix_to_rot6d(rot6d_to_matrix(absolute[..., offset+3:offset+9]))
        return self.normalizer.normalize(DeltaActions(spec)(absolute, anchor))

    def torch(self, normalized, anchor):
        import torch
        spec = self.spec
        value = normalized.reshape(normalized.shape[0], -1, spec.dim)
        anchor = anchor.to(value).unsqueeze(-2)
        low = torch.as_tensor(self.normalizer.low, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(self.normalizer.scale, device=value.device, dtype=value.dtype)
        represented = (value.clamp(-1, 1)+1)*scale/2+low
        lower = torch.as_tensor(self.lower, device=value.device, dtype=value.dtype)
        upper = torch.as_tensor(self.upper, device=value.device, dtype=value.dtype)

        def rotation(r):
            x, y = r[..., :3], r[..., 3:]
            xnorm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
            x = x/xnorm.clamp_min(1e-8)
            y = y-(x*y).sum(-1, keepdim=True)*x
            ynorm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
            if bool(((xnorm < 1e-8) | (ynorm < 1e-8)).any()):
                raise ValueError('degenerate rot6d projection')
            y = y/ynorm
            return torch.stack((x, y, torch.linalg.cross(x, y, dim=-1)), -1)

        blocks = []
        width = 7 if spec.space == 'joint' else 3
        for i, off in enumerate(range(0, spec.dim, spec.width)):
            position = represented[..., off:off+width]
            origin = anchor[..., off:off+width]
            if spec.representation == 'chunk_relative':
                position = position+origin
            position = torch.maximum(torch.minimum(position, upper[i*width:(i+1)*width]), lower[i*width:(i+1)*width])
            if spec.representation == 'chunk_relative':
                position = position-origin
            if spec.space == 'tcp':
                r = rotation(represented[..., off+3:off+9])
                if spec.representation == 'chunk_relative':
                    base = rotation(anchor[..., off+3:off+9])
                    r = r@base
                    # Coordinator canonicalizes the absolute rotation before re-encoding.
                    r = rotation(torch.cat((r[..., :, 0], r[..., :, 1]), -1))@base.transpose(-1, -2)
                rot6d = torch.cat((r[..., :, 0], r[..., :, 1]), -1)
                position = torch.cat((position, rot6d), -1)
            blocks.append(torch.cat((position, represented[..., off+spec.width-1:off+spec.width].clamp(0, 1)), -1))
        result = 2*(torch.cat(blocks, -1)-low)/scale-1
        return result.reshape_as(normalized)
