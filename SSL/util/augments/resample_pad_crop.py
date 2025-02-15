#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from typing import Tuple, Union

from torch import nn, Tensor

from .crop import Crop
from .pad import Pad
from .resample import Resample


class ResamplePadCrop(nn.Module):
    def __init__(
        self,
        rates: Tuple[float, float] = (0.9, 1.1),
        target_length: Union[int, str] = "same",
        align: str = "random",
        fill_value: float = 0.0,
        dim: int = -1,
        p: float = 1.0,
    ) -> None:
        """
        Resample, Pad and Crop the signal.

        :param rates: The ratio of the signal used for resize. (default: (0.9, 1.1))
        :param target_length: Optional target length of the signal dimension.
                If 'auto', the output will have the same shape than the input.
                (default: 'auto')
        :param align: Alignment to use for cropping and padding. Can be 'left', 'right', 'center' or 'random'.
                (default: 'random')
        :param fill_value: The fill value when padding the waveform. (default: 0.0)
        :param dim: The dimension to stretch and pad or crop. (default: -1)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__()
        self.rates = rates
        self._target_length = target_length
        self.align = align
        self.fill_value = fill_value
        self.dim = dim
        self.p = p

        target_length = self.target_length if isinstance(self.target_length, int) else 1
        self.stretch = Resample(rates, dim=dim)
        self.pad = Pad(target_length, align, fill_value, dim, mode="constant")
        self.crop = Crop(target_length, align, dim)

    def extra_repr(self) -> str:
        hparams = {
            "rates": self.rates,
            "target_length": self._target_length,
            "align": self.align,
            "fill_value": self.fill_value,
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
        if self.target_length == "same":
            target_length = data.shape[self.dim]
            self.pad.target_length = target_length
            self.crop.target_length = target_length

        data = self.stretch(data)
        data = self.pad(data)
        data = self.crop(data)
        return data

    @property
    def target_length(self) -> Union[int, str]:
        return self._target_length
