#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math
import os
import random

from typing import Iterable, List, Optional, Tuple, Type, Union

from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from SSL.dataset.esc import ESC10, ESC50, FOLDS
from SSL.dataset_loader.utils import guess_folds
from SSL.util.utils import ZipCycle, ZipDataset


def _split_s_u(train_dataset, s_ratio: float = 1.0, nb_class: int = 10) -> Tuple[List[int], List[int]]:
    if s_ratio == 1.0:
        return list(range(len(train_dataset))), []

    s_idx, u_idx = [], []
    nb_s = int(math.ceil(len(train_dataset) * s_ratio) // nb_class)
    cls_idx = [[] for _ in range(nb_class)]

    # To each file, an index is assigned, then they are split into classes
    for i in range(len(train_dataset)):
        _, y = train_dataset[i]
        cls_idx[y].append(i)

    for i in range(len(cls_idx)):
        random.shuffle(cls_idx[i])
        s_idx += cls_idx[i][:nb_s]
        u_idx += cls_idx[i][nb_s:]

    return s_idx, u_idx


def cache_feature(func):
    def decorator(*args, **kwargs):
        key = ",".join(map(str, args))

        if key not in decorator.cache:
            decorator.cache[key] = func(*args, **kwargs)

        return decorator.cache[key]

    decorator.cache = dict()
    return decorator


class ESC10_NoSR(ESC10):
    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        x, sr, y = super().__getitem__(index)
        return x, y


class ESC50_NoSR(ESC50):
    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        x, sr, y = super().__getitem__(index)
        return x, y


# =============================================================================
#       DEEP CO-TRAINING
# =============================================================================
def dct(
    dataset_root: str,
    supervised_ratio: float = 0.1,
    batch_size: int = 100,
    train_folds: Union[Iterable[int], int, None] = (1, 2, 3, 4),
    val_folds: Union[Iterable[int], int, None] = (5,),
    train_transform_s: Optional[nn.Module] = None,
    train_transform_u: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    download: bool = True,
    dataset_class: Union[Type[ESC10], Type[ESC50]] = ESC10_NoSR,
    num_workers: int = 0,
    pin_memory: bool = False,
    verbose: int = 0,
) -> Tuple[None, ZipCycle, DataLoader, None]:
    train_folds, val_folds = guess_folds(train_folds, val_folds, FOLDS, verbose)

    # Recover extra commun arguments
    loader_args = dict(num_workers=num_workers, pin_memory=pin_memory,)

    dataset_path = os.path.join(dataset_root)

    # validation subset
    val_dataset = dataset_class(
        root=dataset_path, folds=val_folds, download=download, transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )

    # training subset
    train_dataset_s = dataset_class(
        root=dataset_path, folds=train_folds, download=download, transform=train_transform_s
    )
    train_dataset_u = copy.deepcopy(train_dataset_s)
    train_dataset_u.transform = train_transform_u

    s_idx, u_idx = _split_s_u(
        train_dataset_s, supervised_ratio, nb_class=train_dataset_s.nb_class
    )

    # Calc the size of the Supervised and Unsupervised batch
    s_batch_size = int(math.floor(batch_size * supervised_ratio))
    u_batch_size = int(math.ceil(batch_size * (1 - supervised_ratio)))

    sampler_s = SubsetRandomSampler(s_idx)
    sampler_u = SubsetRandomSampler(u_idx)

    train_loader_s1 = DataLoader(
        train_dataset_s, batch_size=s_batch_size, sampler=sampler_s, **loader_args
    )
    train_loader_s2 = DataLoader(
        train_dataset_s, batch_size=s_batch_size, sampler=sampler_s, **loader_args
    )
    train_loader_u = DataLoader(
        train_dataset_u, batch_size=u_batch_size, sampler=sampler_u, **loader_args
    )

    # combine the three loader into one
    train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])

    return None, train_loader, val_loader, None


# =============================================================================
#       DEEP CO-TRAINING
# =============================================================================
def dct_uniloss(
    **kwargs
) -> Tuple[None, ZipCycle, DataLoader, None]:
    return dct(**kwargs)


# =============================================================================
#        SUPERVISED DATASETS
# =============================================================================
def supervised(
    dataset_root,
    supervised_ratio: float = 1.0,
    batch_size: int = 128,
    train_transform: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    train_folds: Union[Iterable[int], int, None] = (1, 2, 3, 4),
    val_folds: Union[Iterable[int], int, None] = (5,),
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    verbose: int = 0,
) -> Tuple[None, DataLoader, DataLoader, None]:
    """
    Load the cifar10 dataset for Deep Co Training system.
    """
    train_folds, val_folds = guess_folds(train_folds, val_folds, FOLDS, verbose)

    # Recover extra commun arguments
    loader_args = dict(num_workers=num_workers, pin_memory=pin_memory)

    dataset_path = os.path.join(dataset_root)

    # validation subset
    val_dataset = ESC10_NoSR(
        root=dataset_path, folds=val_folds, download=download, transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )

    # Training subset
    train_dataset = ESC10_NoSR(
        root=dataset_path, folds=train_folds, download=download, transform=train_transform
    )

    if supervised_ratio == 1.0:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **loader_args
        )

    else:
        s_idx, u_idx = _split_s_u(
            train_dataset, supervised_ratio, nb_class=train_dataset.nb_class
        )

        sampler_s = SubsetRandomSampler(s_idx)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler_s, **loader_args
        )

    return None, train_loader, val_loader, None


# =============================================================================
#        MEAN TEACHER
# =============================================================================
def mean_teacher(
    dataset_root: str,
    supervised_ratio: float = 0.1,
    batch_size: int = 128,
    has_same_trans: bool = True,
    student_transform: Optional[nn.Module] = None,
    teacher_transform: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    train_folds: Union[Iterable[int], int, None] = (1, 2, 3, 4),
    val_folds: Union[Iterable[int], int, None] = (5,),
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    verbose: int = 0,
    dataset_class: Type[ESC50] = ESC10_NoSR,
) -> Tuple[None, ZipCycle, DataLoader, None]:
    """
    Load the cifar10 dataset for Deep Co Training system.
    """
    train_folds, val_folds = guess_folds(train_folds, val_folds, FOLDS, verbose)

    # Recover extra commun arguments
    loader_args = dict(num_workers=num_workers, pin_memory=pin_memory)

    # validation subset
    val_dataset = dataset_class(
        root=dataset_root, folds=val_folds, download=download, transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_args,
    )

    train_student_dataset = dataset_class(
        root=dataset_root, folds=train_folds, download=download, transform=student_transform
    )
    if has_same_trans:
        # Training subset
        train_dataset = train_student_dataset
    else:
        train_teacher_dataset = copy.deepcopy(train_student_dataset)
        train_teacher_dataset.transform = teacher_transform
        train_dataset = ZipDataset(train_student_dataset, train_teacher_dataset)

    s_idx, u_idx = _split_s_u(
        train_student_dataset, supervised_ratio, nb_class=train_student_dataset.nb_class
    )
    s_batch_size = int(math.floor(batch_size * supervised_ratio))
    u_batch_size = int(math.ceil(batch_size * (1 - supervised_ratio)))
    if verbose >= 1:
        print("s_batch_size: ", s_batch_size)
        print("u_batch_size: ", u_batch_size)

    sampler_s = SubsetRandomSampler(s_idx)
    sampler_u = SubsetRandomSampler(u_idx)

    train_s_loader = DataLoader(
        train_dataset, batch_size=s_batch_size, sampler=sampler_s
    )
    train_u_loader = DataLoader(
        train_dataset, batch_size=u_batch_size, sampler=sampler_u
    )

    train_loader = ZipCycle([train_s_loader, train_u_loader])

    return None, train_loader, val_loader, None


# =============================================================================
#        FIXMATCH
# =============================================================================
def fixmatch(**kwargs):
    raise NotImplementedError
