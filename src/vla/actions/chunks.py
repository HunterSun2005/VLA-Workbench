"""Request-local conversion and absolute execution queues for RTC and RLT."""
from collections import deque
from threading import RLock
import numpy as np
from .spec import ChunkContext, immutable_array
from .transforms import AbsoluteActions, DeltaActions


class ChunkCodec:
    def __init__(self, context: ChunkContext, normalizer):
        self.context, self.normalizer = context, normalizer

    def encode(self, absolute):
        return self.normalizer.normalize(DeltaActions(self.context.spec)(absolute, self.context.anchor))

    def decode(self, model_output):
        return AbsoluteActions(self.context.spec)(self.normalizer.unnormalize(model_output), self.context.anchor)

    def prefix(self, absolute_queue):
        """Re-encode overlap with this request's anchor, including normalization."""
        return self.encode(absolute_queue)

    def replay(self, reference_absolute, executed_absolute, *, next_context, next_absolute):
        self.context.spec.require_compatible(next_context.spec)
        return {'reference': self.encode(reference_absolute), 'executed': self.encode(executed_absolute),
                'next': ChunkCodec(next_context, self.normalizer).encode(next_absolute),
                'absolute_reference': np.asarray(reference_absolute).copy(),
                'absolute_executed': np.asarray(executed_absolute).copy(),
                'anchor': self.context.anchor.copy(), 'next_anchor': next_context.anchor.copy(),
                'action_spec': self.context.spec.to_dict(), 'version': 1}


class AbsoluteChunkQueue:
    """Only absolute targets survive a response; a generation change revokes requests."""
    def __init__(self):
        self.generation = 0
        self._latest_request = -1
        self._queue = deque()
        self._lock = RLock()

    def invalidate(self):
        with self._lock:
            self.generation += 1
            self._latest_request = -1
            self._queue.clear()
            return self.generation

    def install(self, codec, output, *, consumed=0):
        with self._lock:
            context = codec.context
            if context.generation != self.generation or context.request_id <= self._latest_request:
                return False
            absolute = codec.decode(output)
            if absolute.ndim != 2 or not 0 <= consumed <= len(absolute):
                raise ValueError('invalid chunk or RTC consumed prefix')
            self._latest_request = context.request_id
            self._queue = deque(immutable_array(row) for row in absolute[consumed:])
            return True

    def snapshot(self):
        with self._lock:
            return np.array(list(self._queue), copy=True)

    def pop(self):
        with self._lock:
            return self._queue.popleft() if self._queue else None
