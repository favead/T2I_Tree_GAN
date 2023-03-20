from typing import Callable, Tuple, List, Dict
import torch
from torch import Tensor
from torch.utils.data import Dataset


class T2IFolderDataset(Dataset):
    def __init__(self, files: List[str], read_data: Callable, get_embedding: Callable,
                 read_image: Callable, ts: List[Callable] = None) -> None:
        super(T2IFolderDataset, self).__init__()
        self.files = files
        self.read_image = read_image
        self.ts = ts

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        y = self.read_image(sample['img_path'])
        for t in self.ts:
            y = t(y)
        return sent_emb, wrong_emb, y

    def __len__(self) -> int:
        return len(self.files)
