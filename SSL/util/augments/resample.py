#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from typing import Any, List, Tuple

import torch

from torch import nn, Tensor
from torch.distributions import Uniform
from torchaudio.transforms import Resample as TorchAudioResample


class Resample(nn.Module):
    INTERPOLATIONS = ("nearest", "linear")

    def __init__(
        self,
        rates: Tuple[float, float] = (0.5, 1.5),
        interpolation: str = "nearest",
        dim: int = -1,
        p: float = 1.0,
    ) -> None:
        """Resample an audio waveform signal.

        :param rates: The rate of the stretch. Ex: use 2.0 for multiply the signal length by 2. (default: (0.5, 1.5))
        :param interpolation: Interpolation for resampling. Can be one of ("nearest", "linear").
            (default: "nearest")
        :param dim: The dimension to modify. (default: -1)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        if interpolation not in self.INTERPOLATIONS:
            raise ValueError(
                f'Invalid interpolation mode "{interpolation}". Must be one of {self.INTERPOLATIONS}.'
            )

        super().__init__()
        self.rates = rates
        self.interpolation = interpolation
        self.dim = dim
        self.p = p

    def extra_repr(self) -> str:
        hparams = {
            "rates": self.rates,
            "interpolation": self.interpolation,
            "dim": self.dim,
            "p": self.p,
        }
        return ", ".join(f"{k}={v}" for k, v in hparams.items())

    def forward(self, x):
        if self.p >= 1.0 or random.random() <= self.p:
            return self.process(x)
        else:
            return x

    def process(self, data: Tensor) -> Tensor:
        sampler = Uniform(*self.rates)
        rate = sampler.sample().item()

        if self.interpolation == "nearest":
            data = self.resample_nearest(data, rate)
        elif self.interpolation == "linear":
            cst = 16000
            tchaudio_resample = TorchAudioResample(cst, cst * rate)
            data = tchaudio_resample(data)
        else:
            raise ValueError(
                f"Invalid interpolation mode {self.interpolation=}. Must be one of {self.INTERPOLATIONS}."
            )

        return data

    def resample_nearest(self, data: Tensor, rate: float) -> Tensor:
        length = data.shape[self.dim]
        step = 1.0 / rate
        indexes = torch.arange(0, length, step)
        indexes = indexes.round().long().clamp(max=length - 1)
        slices: List[Any] = [slice(None)] * len(data.shape)
        slices[self.dim] = indexes
        data = data[slices]
        data = data.contiguous()
        return data
