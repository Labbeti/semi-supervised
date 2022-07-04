#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import hydra

from omegaconf import ListConfig
from torch import nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from SSL.util.compose import compose_augment
from SSL.util.transforms import PadUpTo, Squeeze


transform_to_spec = nn.Sequential(
    PadUpTo(target_length=16000, mode="constant", value=0),
    MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64),
    AmplitudeToDB(),
    Squeeze(0),
)


def supervised(aug_cfg: Optional[ListConfig] = None) -> Tuple[nn.Module, nn.Module]:
    if aug_cfg is not None:
        train_pool = []
        for aug_cfg_i in aug_cfg:
            type_and_aug = {
                "type": aug_cfg_i["type"],
                "aug": hydra.utils.instantiate(aug_cfg_i["aug"]),
            }
            train_pool.append(type_and_aug)
    else:
        train_pool = []
    train_transform = compose_augment(train_pool, transform_to_spec, None, None)
    val_transform = transform_to_spec
    return train_transform, val_transform  # type: ignore


def dct(aug_cfg: Optional[ListConfig] = None) -> Tuple[nn.Module, nn.Module]:
    return supervised(aug_cfg)


def dct_uniloss(aug_cfg: Optional[ListConfig] = None) -> Tuple[nn.Module, nn.Module]:
    return supervised(aug_cfg)


def dct_aug4adv(aug_cfg: Optional[ListConfig] = None) -> Tuple[nn.Module, nn.Module]:
    raise NotImplementedError


def mean_teacher(aug_cfg: Optional[ListConfig] = None) -> Tuple[nn.Module, nn.Module]:
    return supervised(aug_cfg)


def fixmatch(aug_cfg: Optional[ListConfig] = None) -> Tuple[nn.Module, nn.Module]:
    return supervised(aug_cfg)
