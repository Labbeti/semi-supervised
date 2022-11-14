#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Callable

import numpy as np


def linear_rampup(current_epoch: int, ramp_length: int) -> float:
    if ramp_length == 0:
        return 0.0

    if current_epoch >= ramp_length:
        return 1.0

    return current_epoch / ramp_length


def cosine_rampup(current_epoch: int, ramp_length: int) -> float:
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    if ramp_length == 0:
        return 0.0

    if current_epoch >= ramp_length:
        return 1.0

    return -float(0.5 * (math.cos(math.pi * current_epoch / ramp_length) - 1))


def sigmoid_rampup(current_epoch: int, ramp_length: int) -> float:
    """
    https://arxiv.org/pdf/1803.05984.pdf
    Exponential rampup from https://arxiv.org/abs/1610.02242
    """
    if ramp_length == 0:
        return 0.0

    if current_epoch >= ramp_length:
        return 1.0

    current = np.clip(current_epoch, 0.0, ramp_length)
    phase = 1.0 - current / ramp_length
    return float(math.exp(-5.0 * phase * phase))


def sigmoid_rampdown(current_epoch: int, ramp_length: int) -> float:
    if ramp_length == 0:
        return 1.0

    if current_epoch >= ramp_length:
        return 0.0

    current = np.clip(current_epoch, 0.0, ramp_length)
    phase = 1.0 - (current / ramp_length)
    return 1 - float(math.exp(-5.0 * phase ** 2))


class Warmup:
    def __init__(self, max: float, epochs: int, method: Callable[[int, int], float]) -> None:
        self.max = max
        self.epochs = epochs
        self.method = method
        self.current_epoch = 0
        self.value = method(0, epochs)

    def reset(self) -> None:
        self.current_epoch = 0
        self.value = self.method(0, self.epochs)

    def step(self) -> float:
        if self.current_epoch < self.epochs:
            self.current_epoch += 1
            ramp = self.method(self.current_epoch, self.epochs)
            self.value = self.max * ramp

        return self.value

    def __call__(self) -> float:
        return self.value


def test_warmup() -> None:
    linear_warmup = Warmup(10, 80, linear_rampup)
    exp_warmup = Warmup(10, 80, sigmoid_rampup)
    cosine_warmup = Warmup(10, 80, cosine_rampup)

    linear_values = []
    exp_values = []
    cosine_values = []

    for _ in range(150):
        linear_values.append(linear_warmup.value)
        exp_values.append(exp_warmup.value)
        cosine_values.append(cosine_warmup.value)

        linear_warmup.step()
        exp_warmup.step()
        cosine_warmup.step()

    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.subplot(3, 1, 1)
    plt.plot(linear_values, label="linear")
    plt.plot(exp_values, label="exponential")
    plt.plot(cosine_values, label="sigmoid")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_warmup()
