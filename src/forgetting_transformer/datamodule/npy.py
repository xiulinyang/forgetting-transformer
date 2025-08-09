# src/forgetting_transformer/datamodule/npy.py
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .common import DataInfo, Batch
from forgetting_transformer.utils import check_divisible

class StatefulLoader:
    """Wrap a PyTorch DataLoader to add state_dict()/load_state_dict() for resuming."""
    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self._iter = None
        self.batch_id = 0

    def __iter__(self):
        self._iter = iter(self._dataloader)
        return self

    def __next__(self):
        batch = next(self._iter)
        self.batch_id += 1
        return batch

    def __getattr__(self, name):
        return getattr(self._dataloader, name)

    def state_dict(self):
        return {"batch_id": self.batch_id}

    def load_state_dict(self, state):
        self.batch_id = state.get("batch_id", 0)

class NPYSliceDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        self.path = Path(path)
        self.seq_len = seq_len
        self.arr = np.load(self.path, mmap_mode="r")  # 不占内存

        if self.arr.ndim == 1:
            total = self.arr.shape[0]
            self.num_samples = total // seq_len
            self.mode = "1d"
        elif self.arr.ndim == 2:
            assert self.arr.shape[1] == seq_len, f"Expected shape [N,{seq_len}], got {self.arr.shape}"
            self.num_samples = self.arr.shape[0]
            self.mode = "2d"
        else:
            raise ValueError(f"Unsupported npy shape: {self.arr.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == "1d":
            start = idx * self.seq_len
            end = start + self.seq_len
            x = self.arr[start:end]
        else:
            x = self.arr[idx]
        x = np.asarray(x, dtype=np.int64)
        # labels = next-token；和训练脚本期望一致
        input_ids = torch.from_numpy(x)
        labels = input_ids.clone()
        return input_ids, labels

def _collate_batch(batch, bos_id: Optional[int], seq_len: int):

    inputs = torch.stack([b[0] for b in batch], dim=0)  # (B, L)
    labels = torch.stack([b[1] for b in batch], dim=0)  # (B, L)

    if bos_id is not None:
        inputs[:, 0] = bos_id
    resets = torch.zeros_like(inputs, dtype=torch.bool)
    resets[:, 0] = True
    return Batch(input_ids=inputs, labels=labels, resets=resets)


class NPYDataModule:
    def __init__(
        self,
        data_path: str,
        rank: int,
        world_size: int,
        train_batch_len: int,
        train_batch_size: int,
        train_num_workers: int,
        eval_tokens: int,
        eval_batch_len: int,
        eval_local_batch_size: int,
        eval_num_workers: int,
        train_seq_len: Optional[int] = None,
        eval_seq_len: Optional[int] = None,
        bos_token_id: Optional[int] = 0,
    ):
        self.data_path = data_path
        self.rank = rank
        self.world_size = world_size

        self.train_batch_len = train_batch_len
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers

        self.eval_tokens = eval_tokens
        self.eval_batch_len = eval_batch_len
        self.eval_local_batch_size = eval_local_batch_size
        self.eval_num_workers = eval_num_workers

        self.train_seq_len = train_seq_len or train_batch_len
        self.eval_seq_len = eval_seq_len or eval_batch_len
        self.bos_token_id = bos_token_id


        assert self.train_seq_len == self.train_batch_len
        assert (self.train_seq_len & (self.train_seq_len - 1)) == 0, "seq_len 必须是 2 的幂"
        assert self.eval_seq_len == self.eval_batch_len

        self._train_ds = None
        self._eval_ds = None
        self._vocab_size = None  # 如果你需要从数据推断，可以加逻辑


    def prepare_data(self):
        pass

    def setup(self, stage=""):
        self._train_ds = NPYSliceDataset(self.data_path, self.train_seq_len)
        self._eval_ds = self._train_ds

    def train_dataloader(self) -> Tuple[StatefulLoader, DataInfo]:
        ds = self._train_ds
        dl = DataLoader(
            ds,
            batch_size=self.train_batch_size,
            shuffle=True,       # 简单起见
            num_workers=self.train_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda b: _collate_batch(b, self.bos_token_id, self.train_seq_len),
        )
        loader = StatefulLoader(dl)

        global_tokens_per_batch = self.train_batch_size * self.train_batch_len
        local_tokens_per_batch = global_tokens_per_batch  # 单机单进程
        info = DataInfo(
            vocab_size=0,  # 置 0，主脚本会在构建模型前注入真实 vocab_size
            global_tokens_per_batch=global_tokens_per_batch,
            local_tokens_per_batch=local_tokens_per_batch,
            batch_len=self.train_batch_len,
            seq_len=self.train_seq_len,
            total_tokens=ds.__len__() * self.train_seq_len,
        )
        return loader, info

    def val_dataloader(self) -> Tuple[StatefulLoader, DataInfo]:
        ds = self._eval_ds
        dl = DataLoader(
            ds,
            batch_size=self.eval_local_batch_size,
            shuffle=False,
            num_workers=self.eval_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda b: _collate_batch(b, self.bos_token_id, self.eval_seq_len),
        )
        loader = StatefulLoader(dl)

        global_tokens_per_batch = self.eval_local_batch_size * self.eval_batch_len
        local_tokens_per_batch = global_tokens_per_batch
        info = DataInfo(
            vocab_size=0,
            global_tokens_per_batch=global_tokens_per_batch,
            local_tokens_per_batch=local_tokens_per_batch,
            batch_len=self.eval_batch_len,
            seq_len=self.eval_seq_len,
            total_tokens=min(self.eval_tokens, ds.__len__() * self.eval_seq_len),
        )
        return loader, info