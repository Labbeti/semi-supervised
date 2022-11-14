#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

from torch import nn
from torch.utils.data.dataloader import DataLoader

from SSL.dataset.audiosetDataset import get_supervised, get_fixmatch


def supervised(
    dataset_root: str,
    rdcc_nbytes: int = 512 * 1024 ** 2,
    data_shape: tuple = (64, 500,),
    data_key: str = "data",
    train_transform: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    batch_size: int = 64,
    supervised_ratio: float = 1.0,
    unsupervised_ratio: Optional[float] = None,
    balance: bool = True,
    num_workers: int = 10,
    pin_memory: bool = False,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    all_params = locals()
    fn = get_supervised(version="balanced")
    return fn(**all_params)


def dct(**kwargs):
    raise NotImplementedError


def dct_uniloss(**kwargs):
    raise NotImplementedError


def mean_teacher(**kwargs):
    raise NotImplementedError


def fixmatch(
    dataset_root: str,
    rdcc_nbytes: int = 512 * 1024 ** 2,
    data_shape: tuple = (320000,),
    data_key: str = "waveform",
    train_transform: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    batch_size: int = 64,
    supervised_ratio: float = 0.1,
    unsupervised_ratio: Optional[float] = None,
    balance: bool = True,
    num_workers: int = 10,
    pin_memory: bool = False,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    all_params = locals()
    fn = get_fixmatch(version="balanced")
    return fn(**all_params)
