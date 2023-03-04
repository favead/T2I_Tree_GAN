from typing import Callable, Tuple, List
from torch import Tensor
from torch.utils.data import Dataset
import h5py
import numpy as np


class DatasetCreator:
    def __init__(self, filename: str) -> None:
       self.filename = filename
       return None
    
    def create_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def h5py_dataset_template(self) -> None:
        f = h5py.File(self.filename, 'a')
        x_patches, y_patches = self.create_dataset()
        y_patches = np.array(y_patches).astype(np.uint8)
        x_patches = np.array(x_patches).astype(np.uint8)
        f.create_dataset('y', data=y_patches)
        f.create_dataset('x', data=x_patches)
        f.close()
        return None


class SRDataset(Dataset):
    def __init__(self, h5py_filename: str, ts: Callable = None) -> None:
        super(SRDataset, self).__init__()
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
