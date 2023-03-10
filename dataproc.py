from typing import Callable, Tuple, List
import torch
from torch import Tensor
from torch.utils.data import Dataset
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
        assert y.shape == (16, 16, 3)
        x = torch.randn(100, 1, 1)
        for t in self.ts:
            y = t(y)
        return x, y
        
    def __len__(self) -> int:
        return len(self.files)
