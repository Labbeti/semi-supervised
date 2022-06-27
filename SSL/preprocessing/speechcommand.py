#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import hydra

from omegaconf import ListConfig
from torch.nn import Module, Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from SSL.util.compose import compose_augment
from SSL.util.transforms import PadUpTo, Squeeze


transform_to_spec = Sequential(
    PadUpTo(target_length=16000, mode="constant", value=0),
    MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64),
    AmplitudeToDB(),
    Squeeze(0),
)


def supervised(aug_cfg: Optional[ListConfig] = None) -> Tuple[Module, Module]:
    if aug_cfg is not None:
        train_pool = []
        for aug_i_cfg in aug_cfg:
            type_and_aug = {
                "type": aug_i_cfg["type"],
                "aug": hydra.utils.instantiate(aug_i_cfg["aug"]),
            }
            train_pool.append(type_and_aug)
    else:
        train_pool = []
    train_transform = compose_augment(train_pool, transform_to_spec, None, None)
    val_transform = transform_to_spec
    return train_transform, val_transform  # type: ignore


def dct(aug_cfg: Optional[ListConfig] = None) -> Tuple[Module, Module]:
    return supervised(aug_cfg)


def dct_uniloss(aug_cfg: Optional[ListConfig] = None) -> Tuple[Module, Module]:
    return supervised(aug_cfg)


def dct_aug4adv(aug_cfg: Optional[ListConfig] = None) -> Tuple[Module, Module]:
    raise NotImplementedError


def mean_teacher(aug_cfg: Optional[ListConfig] = None) -> Tuple[Module, Module]:
    return supervised(aug_cfg)


def fixmatch(aug_cfg: Optional[ListConfig] = None) -> Tuple[Module, Module]:
    return supervised(aug_cfg)
