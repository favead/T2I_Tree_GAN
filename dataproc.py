from typing import Callable, Tuple, List, Dict
import torch
from torch import Tensor
from torch.utils.data import Dataset


class T2IFolderDataset(Dataset):
    def __init__(self, files: List[str], read_data: Callable, get_embedding: Callable,
                 read_image: Callable, ts: List[Callable] = None) -> None:
        super(T2IFolderDataset, self).__init__()
        self.files = files
        self.read_data = read_data
        self.get_embedding = get_embedding
        self.read_image = read_image
        self.ts = ts

    def get_random_embedding(self, actual_index: int) -> str:
        index = torch.randint(0, len(self.files), (1,)).item()
        while index == actual_index:
            index = torch.randint(0, len(self.files), (1,)).item()
        rnd_emb = self.get_embedding(self.files[index])
        return rnd_emb

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample = self.read_data(self.files[index])
        sent_emb = self.get_embedding(sample['img_path'])
        y = self.read_image(sample['img_path'])
        rnd_emb = self.get_random_embedding(index)
        wrong_emb = 0.5 * rnd_emb + 0.5 * sent_emb
        sent_emb = sent_emb.view(768, 1, 1)
        wrong_emb = wrong_emb.view(768, 1, 1)
        for t in self.ts:
            y = t(y)
        return sent_emb, wrong_emb, y

    def __len__(self) -> int:
        return len(self.files)
