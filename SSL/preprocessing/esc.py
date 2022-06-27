#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import hydra

from omegaconf import DictConfig
from torch import nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from SSL.util.compose import compose_augment


transform_to_spec = nn.Sequential(
    MelSpectrogram(sample_rate=44100, n_fft=2048, hop_length=512, n_mels=64),
    AmplitudeToDB(),
)


def supervised(aug_cfg: Optional[DictConfig] = None) -> Tuple[nn.Module, nn.Module]:
    if aug_cfg is not None:
        train_pool = hydra.utils.instantiate(aug_cfg)
    else:
        train_pool = []
    train_transform = compose_augment(train_pool, transform_to_spec, None, None)
    val_transform = transform_to_spec
    return train_transform, val_transform  # type: ignore


def dct(aug_cfg: Optional[DictConfig] = None) -> Tuple[nn.Module, nn.Module]:
    return supervised(aug_cfg)


def dct_uniloss(aug_cfg: Optional[DictConfig] = None) -> Tuple[nn.Module, nn.Module]:
    return supervised(aug_cfg)


def dct_aug4adv(aug_cfg: Optional[DictConfig] = None) -> Tuple[nn.Module, nn.Module]:
    raise NotImplementedError


def mean_teacher(aug_cfg: Optional[DictConfig] = None) -> Tuple[nn.Module, nn.Module]:
    return supervised(aug_cfg)


def fixmatch(aug_cfg: Optional[DictConfig] = None) -> Tuple[nn.Module, nn.Module]:
    return supervised(aug_cfg)
