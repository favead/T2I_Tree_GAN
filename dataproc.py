from typing import Callable, Tuple, List
from torch import Tensor
from torch.utils.data import Dataset
import h5py
import numpy as np


class SRFolderDataset(Dataset):
    def __init__(self, files: List[str], crop_times: int, scale: int,
                 crop_area: int, read_image: Callable, resize_image: Callable,
                 crop_image: Callable, ts: Callable = None) -> None:
        super(SRFolderDataset, self).__init__()
        self.files = files
        self.ts = ts
        self.crop_times = crop_times
        self.scale = scale
        self.crop_area = crop_area
        self.read_image = read_image
        self.crop_image = crop_image
        self.resize_image = resize_image
        return None
        
    def __getitem__(self, indx: int) -> Tuple[Tensor, Tensor]:
        hr = self.read_image(self.files[indx // self.crop_times])
        hr_crop = self.crop_image(hr, self.crop_area)
        inter = np.random.randint(0, 3)
        lr_crop = self.resize_image(hr_crop, scale=self.scale, is_up=False, typ=inter)
        for t in self.ts:
            hr_crop = t(hr_crop)
            lr_crop = t(lr_crop)
        return lr_crop, hr_crop
        
    def __len__(self) -> int:
        return len(self.files) * self.crop_times

        
class SRTestDataset(Dataset):
    def __init__(self, files: List[Tuple[str, str]], x_T: List[Callable], get_data: Callable) -> None:
        super(SRTestDataset, self).__init__()
        self.files = files
        self.x_T = x_T
        self.get_data = get_data
        
    def __getitem__(self, indx: int) -> Tuple[Tensor, Tensor]:
        x_path, y_path = self.files[indx]
        x, y = map(self.get_data, [x_path, y_path])
        x = self.x_T(x)
        return x, y
    
    def __len__(self) -> int:
        return len(self.files)
