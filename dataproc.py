from typing import Callable, Tuple, List
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SRFolderDataset(Dataset):
    def __init__(self, files: List[str], read_image: Callable, ts: List[Callable] = None) -> None:
        super(SRFolderDataset, self).__init__()
        self.files = files
        self.read_image = read_image
        self.ts = ts

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = torch.randn(100, 1, 1)
        y = self.read_image(self.files[index])
        assert y.shape == (16, 16, 3)
        for t in self.ts:
            y = t(y)
        return x, y

    def __len__(self) -> int:
        return len(self.files)
