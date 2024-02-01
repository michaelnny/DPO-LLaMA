# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Iterable, List
import os
import random
import math
import itertools
import pickle
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class FineTuneDataset(Dataset):
    def __init__(self, data_sources: Iterable[str], max_seq_len: int = 2048, max_samples: int = 0) -> None:
        """
        Args:
            data_sources: a list of string path to where to load the dataset.
            max_seq_len: prompt_tokens + completion_tokens length greater than this will be discarded.
            max_samples: keep maximum number of samples, default 0 no limit.
        """

        assert len(data_sources) > 0
        assert max_seq_len >= 100
        assert max_samples >= 0

        self.data_sources = data_sources
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples

        self.data = []

        # Load datasets
        for source in data_sources:
            samples = pickle.load(open(source, 'rb'))
            for sample in samples:
                x, y = sample['prompt_tokens'], sample['completion_tokens']
                seq_length = len(x) + len(y)
                if seq_length <= self.max_seq_len:
                    self.data.append((x, y))

        if self.max_samples > 0 and len(self.data) > self.max_samples:
            random.shuffle(self.data)
            self.data = self.data[: self.max_samples]

        seq_length_stats = []  # track statistics
        for item in self.data:
            x, y = item
            seq_length = len(x) + len(y)
            seq_length_stats.append(seq_length)

        self.total_num_tokens = sum(seq_length_stats)
        self.seq_length_stats = {
            'min': int(np.min(seq_length_stats)),
            'max': int(np.max(seq_length_stats)),
            'mean': int(np.mean(seq_length_stats)),
            'std': int(np.std(seq_length_stats)),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def get_metadata(self):
        return {
            'num_samples': len(self),
            'num_tokens': self.total_num_tokens,
            'sequence_length_stats': self.seq_length_stats,
            'data_sources': self.data_sources,
        }


class PreferenceDataset(Dataset):
    def __init__(self, data_sources: Iterable[str], max_seq_len: int = 2048, max_samples: int = 0) -> None:
        """
        Args:
            data_sources: a list of string path to where to load the dataset.
            max_seq_len: prompt_tokens + completion_tokens length greater than this will be discarded.
            max_samples: maximum number of samples to include, default 0 no limit.
        """
        assert len(data_sources) > 0
        assert max_seq_len >= 100
        assert max_samples >= 0

        self.data_sources = data_sources
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples

        self.data = []

        # Load datasets
        for source in data_sources:
            samples = pickle.load(open(source, 'rb'))
            for item in samples:
                # assert 'chosen_tokens' in item
                # assert 'rejected_tokens' in item
                # assert 'chosen_ref_logprobs' in item
                # assert 'rejected_ref_logprobs' in item
                # assert 'len_prompt' in item
                # assert 'len_chosen_completion' in item
                # assert 'len_rejected_completion' in item

                # exclude those samples with length greater than max sequence length
                seqlen_chosen = item['len_prompt'] + item['len_chosen_completion']
                seqlen_rejected = item['len_prompt'] + item['len_rejected_completion']
                if seqlen_chosen > self.max_seq_len or seqlen_rejected > self.max_seq_len:
                    continue

                self.data.append(item)

        if self.max_samples > 0 and len(self.data) > self.max_samples:
            random.shuffle(self.data)
            self.data = self.data[: self.max_samples]

        # track statistics
        stats = {
            'prompt': [],
            'chosen_completion': [],
            'rejected_completion': [],
        }

        for item in self.data:
            stats['prompt'].append(item['len_prompt'])
            stats['chosen_completion'].append(item['len_chosen_completion'])
            stats['rejected_completion'].append(item['len_rejected_completion'])

        self.seq_length_stats = {}
        for k, v in stats.items():
            self.seq_length_stats[f'{k}_min'] = np.min(v)
            self.seq_length_stats[f'{k}_max'] = np.max(v)
            self.seq_length_stats[f'{k}_mean'] = np.mean(v)
            self.seq_length_stats[f'{k}_std'] = np.std(v)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        sample = {
            'chosen_tokens': torch.tensor(item['chosen_tokens'], dtype=torch.long),
            'rejected_tokens': torch.tensor(item['rejected_tokens'], dtype=torch.long),
            'chosen_ref_logprobs': torch.tensor(item['chosen_ref_logprobs'], dtype=torch.float),
            'rejected_ref_logprobs': torch.tensor(item['rejected_ref_logprobs'], dtype=torch.float),
            'len_prompt': item['len_prompt'],
            'len_chosen_completion': item['len_chosen_completion'],
            'len_rejected_completion': item['len_rejected_completion'],
        }

        return sample

    def get_metadata(self):
        return {
            'num_samples': len(self),
            'seq_length_stats': self.seq_length_stats,
            'data_sources': self.data_sources,
        }
