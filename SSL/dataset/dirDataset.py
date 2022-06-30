#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from typing import Optional, Tuple

import torchaudio

from torch import nn, Tensor
from torch.utils.data.dataset import Dataset


class DirDataset(Dataset):
    def __init__(
        self,
        path: str,
        transform: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.path = path
        self.transform = transform
        self.filenames = os.listdir(self.path)

    def __getitem__(self, idx: int) -> Tuple[str, Tensor]:
        path = os.path.join(self.path, self.filenames[idx])
        data, _ = torchaudio.load(path)  # type: ignore

        if self.transform is not None:
            data = self.transform(data)
            data = data.squeeze()

        return self.filenames[idx], data

    def __len__(self) -> int:
        return len(self.filenames)
