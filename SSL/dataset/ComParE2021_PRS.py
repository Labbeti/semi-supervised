#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from typing import Dict, List, Tuple

import torchaudio

from torch import Tensor
from torch.utils.data.dataset import Dataset


class COMPARE2021_PRS(Dataset):
    CLASSES = ["background", "chimpanze", "geunon", "mandrille", "redcap"]

    def __init__(self, root: str, subset: str) -> None:
        assert subset in ["train", "test", "devel"]
        self.root = root
        self.subset = subset

        self.subsets_info = self._load_csv()
        self.wav_dir = os.path.join(self.root, "ComParE2021_PRS", "dist", "wav")

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        audio_name = self.subsets_info["audio_names"][idx]
        target = self.subsets_info["target"][idx]
        file_path = os.path.join(self.wav_dir, audio_name)

        waveform, sr = torchaudio.load(file_path)  # type: ignore

        return waveform, target

    def __len__(self) -> int:
        return len(self.subsets_info["audio_names"])

    def _to_cls_idx(self, target_str: str) -> int:
        if target_str == "?":
            return -1

        return COMPARE2021_PRS.CLASSES.index(target_str)

    def _load_csv(self) -> Dict[str, List]:
        def read_csv(path: str) -> Dict[str, List]:
            with open(path, "r") as file:
                lines = file.read().splitlines()
                lines = lines[1:]

            output = {
                "audio_names": [line.split(",")[0] for line in lines],
                "target": [self._to_cls_idx(line.split(",")[1]) for line in lines],
            }

            return output

        csv_root = os.path.join(self.root, "ComParE2021_PRS", "dist", "lab")

        if self.subset == "train":
            return read_csv(os.path.join(csv_root, "train.csv"))

        elif self.subset == "test":
            return read_csv(os.path.join(csv_root, "test.csv"))

        return read_csv(os.path.join(csv_root, "devel.csv"))
