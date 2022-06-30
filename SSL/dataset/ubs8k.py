#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from typing import Dict, Iterable, Tuple

import torchaudio

from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchaudio.transforms import Resample


FOLDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


class URBANSOUND8K(Dataset):
    NB_CLASS = 10
    CLASSES = []

    def __init__(self, root: str, folds: Iterable[int], resample_sr: int = 22050) -> None:
        self.root = root
        self.folds = list(folds)
        self.resample_sr = resample_sr

        self.meta = self._load_metadata()
        self.wav_dir = os.path.join(root, "UrbanSound8K", "audio")

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        filename = self.meta["filename"][idx]
        target = self.meta["target"][idx]
        fold = self.meta["fold"][idx]

        file_path = os.path.join(self.wav_dir, f"fold{fold}", filename)

        waveform, sr = torchaudio.load(file_path)  # type: ignore
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sr)
        waveform = waveform.squeeze()

        return waveform, target

    def __len__(self) -> int:
        return len(self.meta["filename"])

    def _load_metadata(self) -> Dict[str, list]:
        csv_path = os.path.join(
            self.root, "UrbanSound8K", "metadata", "UrbanSound8K.csv"
        )

        with open(csv_path) as f:
            lines = f.read().splitlines()
            lines = lines[1:]  # remove the header

        info = {"filename": [], "fold": [], "target": []}
        for line in lines:
            splitted_line = line.split(",")
            if int(splitted_line[5]) in self.folds:  # l[6] == file folds
                info["filename"].append(splitted_line[0])
                info["fold"].append(int(splitted_line[5]))
                info["target"].append(int(splitted_line[6]))

        return info

    def _resample(self, waveform: Tensor, sr: int) -> Tensor:
        resampler = Resample(sr, self.resample_sr)
        return resampler(waveform)

    def _to_mono(self, waveform: Tensor) -> Tensor:
        if len(waveform.shape) == 2:
            if waveform.shape[0] == 1:
                return waveform
            return waveform.mean(dim=0)
        else:
            raise ValueError(
                f"waveform tensor should be of shape (channels, time). currently is of shape {waveform.shape}"
            )
