#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect

from types import ModuleType
from typing import Callable, List

from torch import nn

import SSL.models.audioset as a
import SSL.models.ComParE2021_PRS as c
import SSL.models.esc as esc
import SSL.models.gsc as sc
import SSL.models.ubs8k as u8
import SSL.models.ubs8k_test as u8_test


dataset_mapper = {
    "ubs8k": [u8, u8_test],
    # "cifar10": [c10],
    "esc10": [esc],
    "esc50": [esc],
    "gsc": [sc],
    # "gtzan": [gtzan],
    "audioset-balanced": [a],
    "audioset-unbalanced": [a],
    "compare2021-prs": [c],
}


def get_model_from_name(
    model_name: str, module_list: List[ModuleType]
) -> Callable[..., nn.Module]:
    all_members = []
    for module in module_list:
        all_members += inspect.getmembers(module)

    for name, obj in all_members:
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__name__ == model_name:
                return obj

    # Error message if model doesn't exist for the dataset
    available_models = [
        name
        for name, obj in all_members
        if inspect.isclass(obj) or inspect.isfunction(obj)
    ]
    msg = f"Model {model_name} doesn't exist for this dataset\n"
    msg += f"Available models are: {available_models}"
    raise ValueError(msg)


def load_model(dataset_name: str, model_name: str) -> Callable[..., nn.Module]:
    dataset_name = dataset_name.lower()

    if dataset_name not in dataset_mapper:
        available_dataset = ", ".join(list(dataset_mapper.keys()))
        raise ValueError(
            f"Dataset {dataset_name} is not available.\n Available datasets: {available_dataset}"
        )

    return get_model_from_name(model_name, dataset_mapper[dataset_name])
