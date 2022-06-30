#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch

from torch import nn


class RandomChoice(nn.Module):
    def __init__(
        self,
        *transforms: Callable,
        n_choices: Union[int, Tuple[int, int]] = 1,
        weights: Optional[Sequence[float]] = None,
        p: float = 1.0,
    ) -> None:
        """
        Select randomly k transforms in a list and apply them sequentially.

        An transform can be chosen multiple times if n_choices > 1. (with replacement)

        :param *transforms: The list of transforms from we choose the apply a transform.
        :param n_choices: The number of transforms to choose.
                If tuple, it will be interpreted as a range [min,max[ for sampling the number of choices for each sample.
                (default: 1)
        :param weights: The probabilities to choose the transform. (default: None)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__()
        self.transforms = list(transforms)
        self.n_choices = n_choices
        self.weights = weights
        self.p = p

    def extra_repr(self) -> str:
        return f"n_transforms={len(self.transforms)}, n_choices={self.n_choices}, weights={self.weights}, p={self.p}"

    def forward(self, x):
        if self.p >= 1.0 or random.random() <= self.p:
            return self.transform(x)
        else:
            return x

    def transform(self, x: Any) -> Any:
        if isinstance(self.n_choices, tuple):
            n_choices_min, n_choices_max = self.n_choices
            n_choices = int(torch.randint(n_choices_min, n_choices_max, ()).item())
        else:
            n_choices = self.n_choices

        transforms = random.choices(
            self.transforms,
            weights=self.weights,
            k=n_choices,
        )
        for transform in transforms:
            x = transform(x)
        return x
