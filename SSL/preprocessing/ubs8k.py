#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import hydra

from omegaconf import DictConfig, ListConfig
from torch import nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from SSL.util.compose import compose_augment
from SSL.util.transforms import PadUpTo


# Define the sequence to transform a waveform into log-mel spectrogram
transform_to_spec = nn.Sequential(
    PadUpTo(target_length=22050 * 4, mode="constant", value=0),
    MelSpectrogram(sample_rate=22050, n_fft=2048, hop_length=512, n_mels=64),
    AmplitudeToDB(),
)


def _get_ubs8k_transforms(
    aug_cfg: Optional[ListConfig] = None,
    pre_trans_cfg: Optional[DictConfig] = None,
    post_trans_cfg: Optional[DictConfig] = None,
) -> Tuple[nn.Module, nn.Module]:

    if aug_cfg is not None:
        train_pool = []
        for aug_cfg_i in aug_cfg:
            type_and_aug = (
                aug_cfg_i["type"],
                hydra.utils.instantiate(aug_cfg_i["aug"]),
            )
            train_pool.append(type_and_aug)
    else:
        train_pool = []

    if pre_trans_cfg is not None:
        pre_trans = hydra.utils.instantiate(pre_trans_cfg)
    else:
        pre_trans = None

    if post_trans_cfg is not None:
        post_trans = hydra.utils.instantiate(post_trans_cfg)
    else:
        post_trans = None

    train_transform = compose_augment(
        train_pool, transform_to_spec, pre_trans, post_trans
    )
    val_transform = transform_to_spec
    return train_transform, val_transform  # type: ignore


dct = _get_ubs8k_transforms
fixmatch = _get_ubs8k_transforms
mean_teacher = _get_ubs8k_transforms
supervised = _get_ubs8k_transforms
