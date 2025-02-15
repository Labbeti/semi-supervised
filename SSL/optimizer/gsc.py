#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.optim import Optimizer, Adam


def supervised(model, learning_rate: float = 0.001, **kwargs) -> Optimizer:
    return Adam(model.parameters(), lr=learning_rate)


def dct(model1, model2, learning_rate: float = 0.001, **kwargs) -> Optimizer:

    parameters = list(model1.parameters()) + list(model2.parameters())
    return Adam(parameters, lr=learning_rate)


def dct_uniloss(model1, model2, learning_rate: float = 0.001, **kwargs) -> Optimizer:
    return dct(model1, model2, learning_rate, **kwargs)


def mean_teacher(student, learning_rate: float = 0.003, **kwargs) -> Optimizer:
    return Adam(student.parameters(), lr=learning_rate)


def fixmatch(model, learning_rate: float = 0.001, **kwargs) -> Optimizer:
    return Adam(model.parameters(), lr=learning_rate)
