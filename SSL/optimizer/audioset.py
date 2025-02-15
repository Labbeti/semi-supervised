#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.optim import Optimizer, Adam


def supervised(
    model,
    lr: float = 0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0,
    amsgrad=True,
    **kwargs
) -> Optimizer:

    return Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )


def dct(model1, model2, learning_rate: float = 0.001, **kwargs) -> Optimizer:
    parameters = list(model1.parameters()) + list(model2.parameters())
    return Adam(parameters, lr=learning_rate)


def dct_uniloss(model1, model2, learning_rate: float = 0.001, **kwargs) -> Optimizer:
    raise NotImplementedError


def mean_teacher(student, **kwargs) -> Optimizer:
    return supervised(student, **kwargs)


def fixmatch(model, **kwargs):
    return supervised(model, **kwargs)
