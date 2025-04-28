import torch
import torch.nn as nn
from guided_diffusion.nn import timestep_embedding     # already provided by repo

class MLPDiff(nn.Module):
    """
    ε‑predictor for 2‑D points.  Takes (N,2) and returns (N,2).
    """
    def __init__(self, model_channels: int = 256, n_layers: int = 3):
        super().__init__()
        hidden = []
        last = 2 + model_channels        # concat point & time‑embedding
        for _ in range(n_layers):
            hidden += [nn.Linear(last, model_channels), nn.SiLU()]
            last = model_channels
        self.net = nn.Sequential(*hidden, nn.Linear(last, 2))

        self.model_channels = model_channels

    def forward(self, x, t, **kwargs):
        """
        x : (N,2)          – noisy points
        t : (N,) int64     – timestep indices
        """
        emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        x_in = torch.cat([x, emb], dim=1)
        return self.net(x_in)

