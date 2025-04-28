import math, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TwoRingGaussians(Dataset):
    """
    50 % of the points are sampled from a narrow inner ring (radius r1),
    50 % from a wider outer ring (radius r2).  Both are corrupted with
    isotropic Gaussian noise (std).
    """
    def __init__(self,
                 n_samples: int = 100_000,
                 r1: float = 2.0,
                 r2: float = 4.0,
                 std: float = 0.05):
        super().__init__()
        n1 = n_samples // 2
        n2 = n_samples - n1

        def ring(n, r):
            theta = 2.0 * math.pi * np.random.rand(n)
            pts = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)
            pts += np.random.randn(n, 2) * std
            return pts

        self.x = torch.from_numpy(np.vstack([ring(n1, r1), ring(n2, r2)])).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        # keep (x, kwargs) signature expected by load_data()
        return self.x[i], {}                 
        
def load_data_points(batch_size=1024,**ds_kwargs):
    # infinite generator that exactly mirrors image_datasets.load_data()
    dl = DataLoader(TwoRingGaussians(**ds_kwargs),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0)
    while True:
        yield from dl
        
