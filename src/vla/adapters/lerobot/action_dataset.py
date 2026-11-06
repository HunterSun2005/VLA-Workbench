"""Validity masks shared by contract-aware SFT and RLToken data loading."""
import numpy as np
import torch
from torch.utils.data import Dataset


class ActionDataset(Dataset):
    def __init__(self, dataset, sampling):
        self.dataset = dataset
        self.sampling = sampling
        table = dataset.hf_dataset
        count = len(table)
        self.valid = (np.asarray(table['complementary_info.action_valid']).reshape(-1).astype(bool)
                      if 'complementary_info.action_valid' in table.column_names else np.ones(count, bool))
        self.episodes = np.asarray(table['episode_index']).reshape(-1)
        self.segments = (np.asarray(table['complementary_info.segment_id']).reshape(-1)
                         if 'complementary_info.segment_id' in table.column_names else np.zeros(count))
        # Match the statistics sampler: starts reset at every episode/segment boundary.
        boundaries = np.r_[0, np.flatnonzero((self.episodes[1:] != self.episodes[:-1])
                                            | (self.segments[1:] != self.segments[:-1])) + 1, count]
        starts = np.zeros(count, dtype=bool)
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            starts[begin:end:sampling.get('start_stride', 1)] = True
        self.indices = np.flatnonzero(self.valid & starts)
        if not len(self.indices):
            raise ValueError('training dataset has no valid action anchors')

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, key):
        return getattr(object.__getattribute__(self, 'dataset'), key)

    def __getitem__(self, index):
        start = int(self.indices[index])
        item = dict(self.dataset[start])
        indices = start + np.arange(self.sampling['chunk_length']) * self.sampling['frame_stride']
        bounded = np.minimum(indices, len(self.valid)-1)
        valid = ((indices < len(self.valid)) & self.valid[bounded]
                 & (self.episodes[bounded] == self.episodes[start])
                 & (self.segments[bounded] == self.segments[start]))
        invalid = torch.as_tensor(~valid)
        item['action_is_pad'] = item.get('action_is_pad', invalid) | invalid
        return item
