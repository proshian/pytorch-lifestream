from typing import List, Dict, Tuple
from functools import reduce
from operator import iadd

import torch
import numpy as np  # for typing

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit
from ptls.data_load.padded_batch import PaddedBatch  # for typing


class ColesDataset(FeatureDict, torch.utils.data.Dataset):
    """Dataset for ptls.frames.coles.CoLESModule

    Parameters
    ----------
    data:
        source data with feature dicts
    splitter: 
        object from from `ptls.frames.coles.split_strategy`.
        Used to split original sequence into subsequences which are samples from one client.
    col_time:
        column name with event_time
    """

    def __init__(self,
                 data,
                 splitter: AbsSplit,
                 col_time: str = 'event_time',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a list of n feature dicts (only sequential features)
        sampled from the client with index `idx`.
        """
        feature_arrays = self.data[idx]
        return self.get_splits(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.get_splits(feature_arrays)

    def get_splits(self, feature_arrays):
        """
        Returns a list of n feature dicts (only sequential features). 
        Each dict is sampled from the original `feature_arrays` thus
        we get n samples from one client.
        """
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        return [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in indexes]

    @staticmethod
    def collate_fn(batch: List[List[Dict[str, np.ndarray]]]) -> Tuple[PaddedBatch, torch.Tensor]:
        class_labels = [i for i, class_samples in enumerate(batch) for _ in class_samples]
        # Flatten `List[List[Dict[str, np.ndarray]]]` to `List[Dict[str, np.ndarray]]`.
        batch = reduce(iadd, batch)
        padded_batch = collate_feature_dict(batch)
        return padded_batch, torch.LongTensor(class_labels)


class ColesIterableDataset(ColesDataset, torch.utils.data.IterableDataset):
    pass
