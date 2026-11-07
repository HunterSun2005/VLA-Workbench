"""The common causal window rule for readers, action statistics and Replay."""
import numpy as np


def window_ends(valid, *, segments=None, slots=None, episodes=None, observed=None):
    valid = np.asarray(valid, dtype=bool)
    count = len(valid)
    if valid.shape != (count,):
        raise ValueError('window validity must be one dimensional')
    boundaries = np.zeros(count, bool)
    boundaries[-1:] = True
    for values, consecutive in ((segments, False), (slots, True), (episodes, False)):
        if values is None:
            continue
        values = np.asarray(values).reshape(-1)
        if len(values) != count:
            raise ValueError('window metadata length mismatch')
        boundaries[:-1] |= np.diff(values) != (1 if consecutive else 0)
    if observed is not None:
        observed = np.asarray(observed).reshape(-1)
        if len(observed) != count:
            raise ValueError('observation validity length mismatch')
        valid = valid & observed.astype(bool)
    ends = np.empty(count, dtype=np.int64)
    end = count
    for i in range(count-1, -1, -1):
        if not valid[i]:
            end = i
        elif boundaries[i]:
            end = i+1
        ends[i] = end
    return ends


def window_indices(ends, start, horizon, frame_stride=1):
    if min(horizon, frame_stride) < 1 or not 0 <= start < len(ends):
        raise ValueError('invalid action window')
    return np.arange(start, min(int(ends[start]), start+horizon*frame_stride), frame_stride, dtype=np.int64)
