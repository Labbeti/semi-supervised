#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

from torch.utils.data.dataloader import DataLoader

from SSL.dataset_loader.gsc import SpeechCommand10
from SSL.dataset_loader.gsc import mean_teacher as gsc35_mean_teacher
from SSL.util.utils import ZipCycle


def mean_teacher(
    **kwargs,
) -> Tuple[None, ZipCycle, DataLoader, DataLoader]:
    kwargs["dataset_class"] = SpeechCommand10
    return gsc35_mean_teacher(**kwargs)
