"""Validity masks shared by contract-aware SFT and RLToken data loading."""
import numpy as np
import torch
from vla.actions.windows import window_ends, window_indices
from torch.utils.data import Dataset


class ActionDataset(Dataset):
    def __init__(self, dataset, sampling):
        self.dataset = dataset
        self.sampling = sampling
        table = dataset.hf_dataset
        count = len(table)
        required = {'complementary_info.action_valid', 'complementary_info.supervision_valid',
                    'complementary_info.observation_valid', 'complementary_info.slot_id',
                    'complementary_info.segment_id', 'episode_index'}
        if not required <= set(table.column_names):
            raise ValueError('schema-5 execution metadata is required for training')
        self.valid = np.asarray(table['complementary_info.action_valid']).reshape(-1).astype(bool)
        self.valid &= np.asarray(table['complementary_info.supervision_valid']).reshape(-1).astype(bool)
        self.episodes = np.asarray(table['episode_index']).reshape(-1)
        self.segments = np.asarray(table['complementary_info.segment_id']).reshape(-1)
        self.ends = window_ends(self.valid, segments=self.segments, episodes=self.episodes,
                               slots=np.asarray(table['complementary_info.slot_id']).reshape(-1),
                               observed=np.asarray(table['complementary_info.observation_valid']).reshape(-1))
        self.valid &= self.ends > np.arange(count)
        # Match the statistics sampler: starts reset at every episode/segment boundary.
        boundaries = np.r_[0, np.flatnonzero((self.episodes[1:] != self.episodes[:-1])
                                            | (self.segments[1:] != self.segments[:-1])
                                            | (np.diff(np.asarray(table['complementary_info.slot_id']).reshape(-1)) != 1)) + 1, count]
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
        indices = window_indices(self.ends, start, self.sampling['chunk_length'], self.sampling['frame_stride'])
        valid = np.arange(self.sampling['chunk_length']) < len(indices)
        invalid = torch.as_tensor(~valid)
        item['action_is_pad'] = item.get('action_is_pad', invalid) | invalid
        return item
