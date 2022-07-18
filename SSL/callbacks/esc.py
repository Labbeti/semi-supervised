#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Callable

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def get_lr_lambda(epochs: int) -> Callable[[int], float]:
    epochs = max(epochs, 1)

    def lr_lambda(epoch: int) -> float:
        return (1.0 + math.cos((epoch - 1) * math.pi / epochs)) * 0.5

    return lr_lambda


def supervised(epochs: int, optimizer: Optimizer, **kwargs) -> list:
    lr_scheduler = LambdaLR(optimizer, get_lr_lambda(epochs))
    return [lr_scheduler]


def dct(epochs: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(epochs, optimizer, **kwargs)


def dct_uniloss(epochs: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(epochs, optimizer, **kwargs)


def mean_teacher(epochs: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(epochs, optimizer, **kwargs)


def fixmatch(epochs: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(epochs, optimizer, **kwargs)
