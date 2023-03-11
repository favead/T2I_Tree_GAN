from typing import Callable, Tuple, List
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
import h5py


class SRFileDataset(Dataset):
    def __init__(self, filename: str, ts: Callable = None) -> None:
        super(SRFolderDataset, self).__init__()
        self.filename = filename
        self.ts = ts
        return None

    def __getitem__(self, indx: int) -> Tuple[Tensor, Tensor]:
        x = torch.randn(100, 1, 1)
        with h5py.File(self.filename, 'r') as f:
            y = f['y'][indx]
            assert y.shape == (16, 16, 3)
            for t in self.ts:
                y = t(y)
            return x, y

    def __len__(self) -> int:
        with h5py.File(self.filename, 'r') as f:
            return len(f['y'])
