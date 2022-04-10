from pathlib import Path

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, path, seq_len, pad=0):
        self.lines = list(filter(bool, Path(path).read_bytes().split(b'\n')))
        self.seq_len = seq_len
        self.pad = pad

    def __len__(self):
        return len(self.lines)

    @staticmethod
    def decode(x):
        return ''.join(map(chr, x))

    def __getitem__(self, idx):
        item = torch.full((self.seq_len,), self.pad)
        line = self.lines[idx][:self.seq_len]
        item[:len(line)] = torch.tensor(list(line))
        return item
