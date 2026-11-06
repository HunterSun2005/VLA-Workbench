import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION
from .configuration_vla_pi05 import VLAPI05Config


class VLAPI05Policy(PI05Policy):
    config_class = VLAPI05Config
    name = 'vla_pi05'

    def forward(self, batch, reduction='mean'):
        images, image_masks = self._preprocess_images(batch)
        actions = self.prepare_action(batch)
        losses = self.model.forward(images, image_masks, batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK], actions,
            self.model.sample_noise(actions.shape, actions.device),
            self.model.sample_time(actions.shape[0], actions.device))
        losses = losses[..., :self.config.output_features[ACTION].shape[0]]
        valid = ~batch.get('action_is_pad', torch.zeros(losses.shape[:2], dtype=torch.bool, device=losses.device)).bool()
        if not valid.any() or not valid.any(dim=1).all():
            raise ValueError('training chunk has no valid target')
        masked = losses * valid.unsqueeze(-1)
        per_sample = masked.sum((1, 2)) / (valid.sum(1) * losses.shape[-1])
        metrics = {'loss': per_sample.mean().item(),
                   'loss_per_dim': (masked.sum((0, 1)) / valid.sum()).detach().cpu().tolist()}
        if reduction not in ('mean', 'none'):
            raise ValueError('unsupported loss reduction')
        return (per_sample if reduction == 'none' else per_sample.mean()), metrics
