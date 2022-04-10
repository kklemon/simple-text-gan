import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)


class ReZero(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, **kwargs):
        return x + self.alpha * self.module(x, *args, **kwargs)


def ResBlock(dim, kernel_size=5, batch_norm=False):
    make_norm = lambda: nn.BatchNorm1d(dim) if batch_norm else nn.Identity()

    return Residual(nn.Sequential(
        make_norm(),
        nn.ReLU(),
        nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),

        make_norm(),
        nn.ReLU(),
        nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
    ))


def Generator(latent_dim, seq_len, dim, num_blocks=5, vocab_size=256, kernel_size=5):
    blocks = [ResBlock(dim, kernel_size=kernel_size, batch_norm=True) for _ in range(num_blocks)]
    return nn.Sequential(
        nn.Linear(latent_dim, dim * seq_len),
        Rearrange('b (d s) -> b d s', d=dim),
        *blocks,
        nn.Conv1d(dim, vocab_size, kernel_size=1),
        nn.Softmax(dim=1)
    )


def Discriminator(seq_len, dim, num_blocks, vocab_size=256, kernel_size=5):
    blocks = [ResBlock(dim, kernel_size=kernel_size, batch_norm=False) for _ in range(num_blocks)]
    return nn.Sequential(
        nn.Conv1d(vocab_size, dim, kernel_size=1),
        *blocks,
        Rearrange('b d s -> b (d s)'),
        nn.Linear(dim * seq_len, 1),
        nn.Flatten()
    )
