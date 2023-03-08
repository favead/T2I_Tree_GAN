from typing import Callable, Tuple, List
import torch
from torch import Tensor
from torch.utils.data import Dataset
import h5py
import numpy as np


class SRFolderDataset(Dataset):
    def __init__(self, files: List[str], read_image: Callable, ts: Callable = None) -> None:
        super(SRFolderDataset, self).__init__()
        self.files = files
        self.ts = ts
        self.read_image = read_image
        return None
        
    def __getitem__(self, indx: int) -> Tuple[Tensor, Tensor]:
        y = self.read_image(self.files[indx])
        x = torch.rand((y.shape))
        for t in self.ts:
            y = t(y)
            x = t(x)
        return x, y
        
    def __len__(self) -> int:
        return len(self.files)
