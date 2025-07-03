import torch.nn as nn

class MLMHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model, bias=False)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.lin(x)
        x = self.gelu(x)
        x = self.norm(x)

        return x
