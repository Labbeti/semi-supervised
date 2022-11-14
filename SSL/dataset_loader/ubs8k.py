#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math
import random

from typing import Any, Iterable, List, Optional, Tuple, Union

from torch import nn, Tensor
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from SSL.dataset.ubs8k import URBANSOUND8K, FOLDS
from SSL.dataset_loader.utils import guess_folds
from SSL.util.utils import Cacher, ZipCycle, ZipDataset


class UrbanSound8K(URBANSOUND8K):
    def __init__(
        self,
        root: str,
        folds: Iterable[int],
        transform: Optional[nn.Module] = None,
        cache: bool = False,
    ) -> None:
        super().__init__(root, folds)
        self.transform = transform
        self.cache = cache

        self.cached_getitem = Cacher(self._cacheable_getitem)
        self.cached_transform = Cacher(self._cacheable_transform)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        data, target = self.cached_getitem(idx=idx, caching=True)
        data = self.cached_transform(data, key=idx, caching=self.cache)
        return data, target

    def _cacheable_getitem(self, idx: int) -> Tuple[Tensor, int]:
        return super().__getitem__(idx)

    def _cacheable_transform(self, x, key=None):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x


def split_s_u(dataset: URBANSOUND8K, s_ratio: float) -> Tuple[List[int], List[int]]:
    idx_list = list(range(len(dataset.meta["filename"])))
    s_idx, u_idx = [], []

    # sort the classes
    class_idx = [[] for _ in range(URBANSOUND8K.NB_CLASS)]
    for idx in tqdm(idx_list):
        class_idx[dataset.meta["target"][idx]].append(idx)

    # split each class seperatly to keep distribution
    for i in range(URBANSOUND8K.NB_CLASS):
        random.shuffle(class_idx[i])

        nb_item = len(class_idx[i])
        nb_s = int(math.ceil(nb_item * s_ratio))

        s_idx += class_idx[i][:nb_s]
        u_idx += class_idx[i][nb_s:]

    return s_idx, u_idx


def supervised(
    dataset_root: str,
    supervised_ratio: float = 1.0,
    batch_size: int = 64,
    train_folds: Union[Iterable[int], int, None] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    val_folds: Union[Iterable[int], int, None] = (10,),
    train_transform: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    use_cache: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    verbose: int = 1,
) -> Tuple:
    """
    Load the UrbanSound dataset for supervised systems.
    """
    train_folds, val_folds = guess_folds(train_folds, val_folds, FOLDS, verbose)
    loader_args = {"num_workers": num_workers, "pin_memory": pin_memory}

    # validation subset
    # val_dataset = Dataset(manager, folds=val_folds, cached=True)
    val_dataset = UrbanSound8K(
        dataset_root, val_folds, transform=val_transform, cache=True
    )
    print("nb file validation: ", len(val_dataset))
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )

    # training subset
    train_dataset = UrbanSound8K(
        dataset_root, train_folds, transform=train_transform, cache=use_cache
    )
    print("nb file training: ", len(train_dataset))

    if supervised_ratio == 1.0:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **loader_args
        )

    else:
        s_idx, _u_idx = split_s_u(train_dataset, supervised_ratio)

        # Train loader only use the s_idx
        sampler_s = SubsetRandomSampler(s_idx)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler_s,
        )

    return None, train_loader, val_loader


def mean_teacher(
    dataset_root: str,
    supervised_ratio: float = 0.1,
    batch_size: int = 64,
    train_folds: Union[Iterable[int], int, None] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    val_folds: Union[Iterable[int], int, None] = (10,),
    has_same_trans: bool = True,
    student_transform: Optional[nn.Module] = None,
    teacher_transform: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    use_cache: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    verbose: int = 1,
    download: Any = None,
) -> Tuple:
    train_folds, val_folds = guess_folds(train_folds, val_folds, FOLDS, verbose)
    loader_args = {"num_workers": num_workers, "pin_memory": pin_memory}

    # validation subset
    val_dataset = UrbanSound8K(
        dataset_root, val_folds, transform=val_transform, cache=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )

    train_student_dataset = UrbanSound8K(
        dataset_root, train_folds, transform=student_transform, cache=use_cache
    )
    if has_same_trans:
        # training subset
        train_dataset = train_student_dataset
    else:
        train_teacher_dataset = copy.deepcopy(train_student_dataset)
        train_teacher_dataset.transform = teacher_transform
        train_dataset = ZipDataset(train_student_dataset, train_teacher_dataset)

    # Calc the size of the Supervised and Unsupervised batch
    s_idx, u_idx = split_s_u(train_student_dataset, supervised_ratio)

    s_batch_size = int(math.floor(batch_size * supervised_ratio))
    u_batch_size = int(math.ceil(batch_size * (1.0 - supervised_ratio)))

    sampler_s = SubsetRandomSampler(s_idx)
    sampler_u = SubsetRandomSampler(u_idx)

    train_s_loader = DataLoader(
        train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args,
    )
    train_u_loader = DataLoader(
        train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args,
    )

    train_loader = ZipCycle([train_s_loader, train_u_loader], align="max")

    return None, train_loader, val_loader, None


def dct(
    dataset_root: str,
    batch_size: int = 100,
    download: Any = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    supervised_ratio: float = 0.1,
    train_folds: Union[Iterable[int], int, None] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    train_transform_s: Optional[nn.Module] = None,
    train_transform_u: Optional[nn.Module] = None,
    use_cache: bool = False,
    val_folds: Union[Iterable[int], int, None] = (10,),
    val_transform: Optional[nn.Module] = None,
    verbose: int = 1,
) -> Tuple:
    train_folds, val_folds = guess_folds(train_folds, val_folds, FOLDS, verbose)

    loader_args = {"num_workers": num_workers, "pin_memory": pin_memory}

    # validation subset
    val_dataset = UrbanSound8K(
        dataset_root, val_folds, transform=val_transform, cache=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )

    # training subset
    train_dataset_s = UrbanSound8K(
        dataset_root, train_folds, transform=train_transform_s, cache=use_cache
    )
    train_dataset_u = copy.deepcopy(train_dataset_s)
    train_dataset_u.transform = train_transform_u

    # Calc the size of the Supervised and Unsupervised batch
    s_idx, u_idx = split_s_u(train_dataset_s, supervised_ratio)

    s_batch_size = int(math.floor(batch_size * supervised_ratio))
    u_batch_size = int(math.ceil(batch_size * (1 - supervised_ratio)))

    sampler_s1 = SubsetRandomSampler(s_idx)
    sampler_s2 = SubsetRandomSampler(s_idx)
    sampler_u = SubsetRandomSampler(u_idx)

    train_loader_s1 = DataLoader(
        train_dataset_s, batch_size=s_batch_size, sampler=sampler_s1, **loader_args
    )
    train_loader_s2 = DataLoader(
        train_dataset_s, batch_size=s_batch_size, sampler=sampler_s2, **loader_args
    )
    train_loader_u = DataLoader(
        train_dataset_u, batch_size=u_batch_size, sampler=sampler_u, **loader_args
    )
    train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])

    return None, train_loader, val_loader, None


def dct_uniloss(**kwargs):
    raise NotImplementedError


def fixmatch(**kwargs):
    raise NotImplementedError
