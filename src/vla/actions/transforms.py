"""Chunk anchored actions; rotations compose in the robot base frame."""
import numpy as np

from .spec import validate_array


def matrix_to_rot6d(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('rotation must be a finite 3x3 matrix')
    if not np.allclose(matrix.swapaxes(-2, -1) @ matrix, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(matrix), 1, atol=1e-5):
        raise ValueError('rotation matrix must belong to SO(3)')
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def rot6d_to_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError('rotation must be finite rot6d')
    x, y = values[..., :3], values[..., 3:]
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('degenerate rot6d first column')
    x = x / norm
    y = y - np.sum(x * y, axis=-1, keepdims=True) * x
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('degenerate rot6d columns')
    y = y / norm
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def _transform(actions, anchor, spec, inverse):
    actions, anchor = validate_array(actions, spec), validate_array(anchor, spec)
    # A batched [B,D] anchor applies to every [B,K,D] future action.
    if actions.ndim == anchor.ndim + 1:
        anchor = np.expand_dims(anchor, -2)
    if np.broadcast_shapes(actions.shape, anchor.shape) != actions.shape:
        raise ValueError('anchor does not broadcast to action chunk')
    result = actions.copy()
    for offset in range(0, spec.dim, spec.width):
        end = offset + (7 if spec.space == 'joint' else 3)
        if spec.space == 'tcp':
            rotation = rot6d_to_matrix(actions[..., offset+3:offset+9])
            origin = rot6d_to_matrix(anchor[..., offset+3:offset+9])
        if spec.representation == 'absolute':
            continue
        result[..., offset:end] += (1 if inverse else -1) * anchor[..., offset:end]
        if spec.space == 'tcp':
            result[..., offset+3:offset+9] = matrix_to_rot6d(
                rotation @ (origin if inverse else origin.swapaxes(-2, -1)))
    return result


class DeltaActions:
    def __init__(self, spec):
        self.spec = spec

    def __call__(self, actions, anchor):
        return _transform(actions, anchor, self.spec, False)


class AbsoluteActions:
    def __init__(self, spec):
        self.spec = spec

    def __call__(self, actions, anchor):
        return _transform(actions, anchor, self.spec, True)
