"""Request-local conversion of committed absolute targets for RTC."""
from .spec import ChunkContext
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
