from typing import Optional

from torch.utils.data import Dataset  # for typing
import torch
import pytorch_lightning as pl
from functools import partial


class PtlsDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: Optional[Dataset] = None,
                 train_batch_size: int = 1,
                 train_num_workers: int = 0,
                 train_drop_last: bool = False,
                 valid_data: Optional[Dataset] = None,
                 valid_batch_size: Optional[int] = None,
                 valid_num_workers: Optional[int] = None,
                 valid_drop_last: bool = False,
                 test_data: Optional[Dataset] = None,
                 test_batch_size: Optional[int] = None,
                 test_num_workers: Optional[int] = None,
                 test_drop_last: bool = False,
                 ) -> None:

        super().__init__()
        self.save_hyperparameters(ignore=['train_data', 'valid_data', 'test_data'])
        
        if self.hparams.valid_num_workers is None:
            self.hparams.valid_num_workers = self.hparams.train_num_workers
        if self.hparams.test_num_workers is None:
            self.hparams.test_num_workers = self.hparams.valid_num_workers

        if self.hparams.valid_batch_size is None:
            self.hparams.valid_batch_size = self.hparams.train_batch_size
        if self.hparams.test_batch_size is None:
            self.hparams.test_batch_size = self.hparams.valid_batch_size

        if train_data is not None:
            self.train_dataloader = partial(self.train_dl, train_data)
        if valid_data is not None:
            self.val_dataloader = partial(self.val_dl, valid_data)
        if test_data is not None:
            self.test_dataloader = partial(self.test_dl, test_data)
            
    def train_dl(self, train_data: Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=train_data,
            collate_fn=train_data.collate_fn,
            shuffle=not isinstance(train_data, torch.utils.data.IterableDataset),
            num_workers=self.hparams.train_num_workers,
            batch_size=self.hparams.train_batch_size,
            drop_last=self.hparams.train_drop_last,
        )

    def val_dl(self, valid_data: Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=valid_data,
            collate_fn=valid_data.collate_fn,
            shuffle=False,
            num_workers=self.hparams.valid_num_workers,
            batch_size=self.hparams.valid_batch_size,
            drop_last=self.hparams.valid_drop_last,
        )

    def test_dl(self, test_data: Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=test_data,
            collate_fn=test_data.collate_fn,
            shuffle=False,
            num_workers=self.hparams.test_num_workers,
            batch_size=self.hparams.test_batch_size,
            drop_last=self.hparams.test_drop_last,
        )