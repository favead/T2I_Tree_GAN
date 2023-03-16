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

    def get_wrong_caption(self, actual_index: int) -> str:
        index = torch.randint(0, len(self.files), (1,)).item()
        if index == actual_index:
            return self.get_wrong_caption(actual_index)
        rnd_caption = self.read_data(self.files[index])['caption']
        return rnd_caption

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample = self.read_data(self.files[index])
        sent_emb = self.get_embedding(sample['caption'])
        y = self.read_image(sample['img_path'])
        wrong_cap = self.get_wrong_caption(index)
        wrong_emb = 0.5 * self.get_embedding(wrong_cap) + 0.5 * sent_emb
        sent_emb = sent_emb.view(768, 1, 1)
        wrong_emb = wrong_emb.view(768, 1, 1)
        for t in self.ts:
            y = t(y)
        return sent_emb, wrong_emb, y

    def __len__(self) -> int:
        return len(self.files)
