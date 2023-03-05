from typing import Callable, Tuple, List
from torch import Tensor
from torch.utils.data import Dataset
import h5py
import numpy as np
from imageproc import read_image, resize_image, crop_image
from settings import Config


class DatasetCreator:
    def __init__(self, filename: str, crop_times: int, img_files: List[str],
                crop_area: int, scale: int) -> None:
       self.filename = filename
       self.crop_times = crop_times
       self.img_files = img_files
       self.crop_area = crop_area
       self.scale = scale
       return None
    
    def create_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        x_patches, y_patches = [], []
        for y_path in self.files:
            y = read_image(y_path)
            for _ in self.crop_times:
                y_crop = crop_image(y, self.crop_area)
                inter = np.random.randint(0, 3)
                x_crop = resize_image(y_crop, self.scale, is_up=False, typ=inter)
                x_patches.append(x_crop)
                y_patches.append(y_crop)
        return x_patches, y_patches

    def h5py_dataset_template(self) -> None:
        f = h5py.File(self.filename, 'a')
        x_patches, y_patches = self.create_dataset()
        y_patches = np.array(y_patches).astype(np.uint8)
        x_patches = np.array(x_patches).astype(np.uint8)
        f.create_dataset('y', data=y_patches)
        f.create_dataset('x', data=x_patches)
        f.close()
        return None



class SRFolderDataset(Dataset):
    def __init__(self, files: List[str], ts: Callable = None,
                 crop_times: int = Config.crop_times, scale: int = Config.scale,
                 crop_area: int = Config.crop_area) -> None:
        super(SRFolderDataset, self).__init__()
        self.files = files
        self.ts = ts
        self.crop_times = crop_times
        self.scale = scale
        self.crop_area = crop_area
        
    def __getitem__(self, indx: int) -> Tuple[Tensor, Tensor]:
        hr = read_image(self.files[indx // self.crop_times])
        hr_crop = crop_image(hr, self.crop_area)
        inter = np.random.randint(0, 3)
        lr_crop = resize_image(hr_crop, scale=self.scale, is_up=False, typ=inter)
        for t in self.ts:
            hr_crop = t(hr_crop)
            lr_crop = t(lr_crop)
        return lr_crop, hr_crop
        
    def __len__(self) -> int:
        return len(self.files) * self.crop_times


class SRFileDataset(Dataset):
    def __init__(self, h5py_filename: str, ts: Callable = None) -> None:
        super(SRFileDataset, self).__init__()
        self.h5py_filename = h5py_filename
        self.ts = ts
        
    def __getitem__(self, indx: int) -> Tuple[Tensor, Tensor]:
        with h5py.File(self.h5py_filename, 'r') as f:
            x = f['x'][indx]
            y = f['y'][indx]
            x, y = map(self.ts, [x, y]) if self.ts else x, y
            return x, y
        
    def __len__(self) -> int:
        with h5py.File(self.h5py_filename, 'r') as f:
            return len(f['x'])

        
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
