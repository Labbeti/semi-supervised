#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import itertools
import os
import random

from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch

from tqdm import tqdm, trange
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

from SSL.util.utils import cache_to_disk, ZipCycleInfinite


class Audioset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[nn.Module] = None,
        version: str = "unbalanced",
        rdcc_nbytes: int = 512 * 1024 ** 2,
        data_shape: tuple = (320000,),
        data_key: str = "waveform",
    ) -> None:
        """
        A pytorch dataset of Google AUdioset.

        Args:
            root (bool): The directory that contain the HDF files.
            transform: (Module) The transformation to apply on each samples.
            version: (str) The version of the dataset, "[unbalanced | balanced | eval]"
            rdcc_nbytes: (int) The HDF "raw data chunk cache" in bytes. default 512 MB

            data_shape: (tuple) The shape of the data contain in the HDF files
                (320000, ) for raw audio sampled at 32000 KHz
                (64, 500, ) for the pre-compute mel-spectrogram

            data_key: (str) The key under which the data is store in the HDF files.
                "waveform" when using the raw audio
                "data" when using the pre-compute mel-spectrogram
        """
        self.transform = transform
        self.version = version
        self.rdcc_nbytes = rdcc_nbytes
        self.hdf_root = root
        self.data_shape = data_shape

        if self.version not in ["balanced", "unbalanced", "eval"]:
            raise ValueError('version available: "unbalanced", "balanced" and "eval"')

        # HDF dataset name change if you use pre-compute feature
        self.data_key = data_key

        # variable to manage the hdf files
        self.hdf_mapper = dict()
        self.hdf_nb_row = dict()
        self.hdf_nb_chunk = dict()
        self.hdf_chunk_size = None

        # store all targets and all audio_names for faster loading
        self.targets = None
        self.audio_names = None

        self._check_hdf()
        self._prepare_hdfs()
        self._errors()

    def _errors(self):
        """The different condition to use this dataset are.

        - batch size must be a power of 2.
        - HDF chunk size must be a power of 2.
        - HDF chunk size must be identical accross all HDF files.
        """

        def is_power2(n):
            return (n & (n - 1) == 0) and n != 0

        if not is_power2(self.hdf_chunk_size):
            raise RuntimeError(
                "HDF file chunk first dimension must be a pwoer of 2. it is %d"
                % self.hdf_chunk_size
            )

        # TODO chunk size identical in each HDF files (should be done here ?)

    def _check_hdf(self):
        audioname_path = os.path.join(self.hdf_root, "audio_names.npy")
        targets_path = os.path.join(self.hdf_root, "targets.npy")

        if not os.path.isfile(audioname_path):
            print("audioname_path: ", audioname_path)
            raise RuntimeError(
                "Audioname standalone file doesn't exist. Please create it."
            )

        if not os.path.isfile(targets_path):
            raise RuntimeError(
                "Targets standalone file doesn't exist. Please create it."
            )

        # TODO add verification for HDF files
        # total number of row

    def _prepare_hdfs(self):
        """Open and get some statistic from the HDF files.

        The HDF file being divided into chunk of size n, If the last chunk is incomplete
        it will not be taken into consideration.""
        """

        def get_chunk_valid_stat(hdf_file):
            nb_row = len(hdf_file["audio_name"])
            chunk_size = hdf_file["audio_name"].chunks[0]

            nb_valid_chunk = nb_row // chunk_size
            nb_valid_row = nb_valid_chunk * chunk_size

            return nb_valid_row, nb_valid_chunk

        hdf_names = [
            name
            for name in os.listdir(self.hdf_root)
            if ".h5" in name and self.version[:4] in name[:4]
        ]

        for name in hdf_names:
            path = os.path.join(self.hdf_root, name)

            hdf_file = h5py.File(path, "r", rdcc_nbytes=self.rdcc_nbytes, swmr=True)
            nb_row, nb_chunk = get_chunk_valid_stat(hdf_file)

            self.hdf_mapper[name] = hdf_file
            self.hdf_nb_row[name] = nb_row
            self.hdf_nb_chunk[name] = nb_chunk

        self.hdf_chunk_size = self.hdf_mapper[name]["audio_name"].chunks[0]

        targets_path = os.path.join(self.hdf_root, "targets.npy")
        self.targets = np.load(targets_path)[()]

        audio_names_path = os.path.join(self.hdf_root, "audio_names.npy")
        self.audio_names = np.load(audio_names_path)

    def _close_hdfs(self):
        for hdf_file in self.hdf_mapper.values():
            hdf_file.close()

        self.hdf_mapper = dict()
        self.hdf_nb_row = dict()
        self.hdf_nb_chunk = dict()
        self.hdf_chunk_size = None

    def __getitem__(self, sample_idx: int) -> Tuple[Tensor, Tensor]:
        """Recover one filefrom the Audioset dataset.

        Fetching file should be aligned to the HDF dataset's chunk dimension.

        One HDF dataset can be separated into chunk for more efficient IO interaction. Fetching one
        file will lead to loading the chunk the file is in. Therefore, fetching the other file
        containing inside this chunk will be drastically faster.
        Feeding the sample index to the dataset with respect to this optimization is done using the batch sampler bellow
        """
        # 1 - Find in which HDF file and which chunk is the sample
        hdf_file, chunk_idx, _ = self._get_location(sample_idx)

        # 2 - Read the complete chunk (will be store in memory)
        data, targets = self._read_chunk(hdf_file, chunk_idx)

        # 3 - Keep only the wanted file
        sample_chunk_pos = sample_idx % self.hdf_chunk_size
        data, target = data[sample_chunk_pos], targets[sample_chunk_pos]

        # 4 - Apply Transformation
        data = self._apply_transform(data)

        return data, target

    def _get_location(self, sample_idx: int) -> Tuple[h5py.File, int, str]:
        hdf_names = list(self.hdf_mapper.keys())
        cumulative_nb_chunk = np.cumsum(list(self.hdf_nb_chunk.values()))

        # find the chunk which contain the sample (at the dataset level)
        global_chunk_idx = sample_idx // self.hdf_chunk_size

        # Find in which HDF file the chunk is
        for i in range(len(cumulative_nb_chunk)):
            if global_chunk_idx < cumulative_nb_chunk[i]:
                if i == 0:
                    local_chunk_idx = global_chunk_idx
                    hdf_name = hdf_names[i]
                    return self.hdf_mapper[hdf_name], local_chunk_idx, hdf_name

                else:
                    local_chunk_idx = global_chunk_idx - cumulative_nb_chunk[i - 1]
                    hdf_name = hdf_names[i]
                    return self.hdf_mapper[hdf_name], local_chunk_idx, hdf_name

    def _read_chunk(self, hdf_file, chunk_idx) -> Tuple[np.ndarray, np.ndarray]:
        start = chunk_idx * self.hdf_chunk_size
        end = start + self.hdf_chunk_size

        targets = np.zeros(shape=(self.hdf_chunk_size, 527), dtype=bool)
        waveforms = np.zeros(
            shape=(self.hdf_chunk_size, *self.data_shape), dtype=np.int16
        )

        hdf_file["target"].read_direct(targets, slice(start, end), None)
        hdf_file[self.data_key].read_direct(waveforms, slice(start, end), None)

        return waveforms, targets

    def get_data(self, sample_idx):
        """To call if need to read only one sample from the hdf file"""
        hdf_file, chunk_idx, hdf_name = self._get_location(sample_idx)

        hdf_sample_idx = sample_idx % self.hdf_nb_row[hdf_name]
        return hdf_file[self.data_key][hdf_sample_idx]

    def get_target(self, sample_idx):
        """To call if need to read only the labels of one sample"""
        hdf_file, chunk_idx, hdf_name = self._get_location(sample_idx)

        hdf_sample_idx = sample_idx % self.hdf_nb_row[hdf_name]
        return hdf_file["target"][hdf_sample_idx]

    def _apply_transform(self, data):
        if self.transform is None:
            return data

        data = self.transform(data)
        data = data.squeeze()
        return data

    @functools.lru_cache(maxsize=1)
    def __len__(self) -> int:
        nb_total_row = sum(list(self.hdf_nb_row.values()))
        return nb_total_row


class SingleAudioset(Audioset):
    def __getitem__(self, sample_idx: int) -> Tuple[Tensor, Tensor]:
        """Recover one file from the Audioset dataset."""
        # 1 - Find in which HDF file and which chunk is the sample
        hdf_file, chunk_idx, _ = self._get_location(sample_idx)

        # 2 - recove only the required file (don't read the chunk)
        data = self.get_data(sample_idx)
        target = self.get_target(sample_idx)

        # 2 - Apply Transformation
        data = self._apply_transform(data)

        return data, target


class SingleBalancedSampler:
    def __init__(self, dataset: SingleAudioset, index_list: list, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle

        self.index_list = index_list
        self.all_targets = self._get_all_targets()
        self.sorted_sample_indexes = self._sort_per_class()

    def _get_all_targets(self):
        """Pre-fetch all the labels corresponding to the dataset samples.
        It will be used to balance the dataset.
        The list returned is in the same order than self.index_list
        """
        print("Getting all target")
        if self.dataset.targets is not None:
            return self.dataset.targets

        return [self.dataset.get_target(idx) for idx in tqdm(self.index_list)]

    def _sort_per_class(self):
        """Pre-sort all the sample among the 527 different class.
        It will used to pick the correct file to feed the model
        """
        print("Sort the classes")
        class_indexes = [[] for _ in range(527)]

        for sample_idx, target in zip(self.index_list, self.all_targets):
            target_idx = np.where(target == 1)[0]

            for t_idx in target_idx:
                class_indexes[t_idx].append(sample_idx)

        return class_indexes

    def _shuffle(self):
        # Sort the file for each class
        for i in self.sorted_sample_indexes:
            random.shuffle(i)

        # Sort the class order
        random.shuffle(self.sorted_sample_indexes)

    def __len__(self):
        return len(self.index_list)

    def __iter__(self):
        """Round Robin algorithm to fetch file one by one from each class."""
        if self.shuffle:
            self._shuffle()

        global_index = 0
        for cls_idx in itertools.cycle(range(527)):
            selected_class = self.sorted_sample_indexes[cls_idx]

            if len(selected_class) > 0:
                local_idx = global_index % len(selected_class)
                global_index += 1

                yield selected_class[local_idx]


class ChunkAlignSampler(object):
    """Yield mini-batch align with the HDF chunk.

    Args:
        dataset (Audioset) dataset
        batch_size (int): Size of mini-batch.
        shuffle (bool): If ``True`` the sampler will shuffle the entier dataset but keep
            the sample in each minibatches align with the HDF chunks
    """

    def __init__(
        self, dataset: Audioset, batch_size: int, shuffle: bool = False
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = True
        self.shuffle = shuffle

        self.hdf_batches = self._prepare_minibatch()

        self._errors()

    def _errors(self):
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive value but got batch_size={}".format(
                    self.batch_size
                )
            )

        if not isinstance(self.drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value but got drop_last={}".format(
                    self.drop_last
                )
            )

    def _prepare_minibatch(self):
        # Find the number of file in every hdf file
        hdf_nb_sample = list(self.dataset.hdf_nb_row.values())

        # For every file, create a list of idx and drop the last mini-batch if not complete)
        start_idx = 0
        hdf_batches = []

        for nb_sample in hdf_nb_sample:
            indexes = np.arange(start_idx, start_idx + nb_sample)

            extra = len(indexes) % self.batch_size
            if extra != 0:
                indexes = indexes[:-extra]

            hdf_nb_batch = len(indexes) // self.batch_size

            hdf_batches.append(np.split(indexes, hdf_nb_batch))
            start_idx += nb_sample

        return hdf_batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.hdf_batches)

            for i in range(len(self.hdf_batches)):
                random.shuffle(self.hdf_batches)

        for hdf_batches in self.hdf_batches:
            for batch in hdf_batches:
                yield batch

    def __len__(self):
        return sum(map(len, self.hdf_batches))


def batch_balancer(pool_size=100, batch_size: int = 64):
    """Stochastic batch balancing.

    Audioset is a very unbalanced dataset. The ratio between the least represented and
    the more represented class is 0.04 (closer to 1 is better).
    Using a pool of samples, the balancer find the class that is the most represented
    inside the current minibatch and replace a certain number of these samples by some
    present in the pool. The process can be repeat to further balance the batch.
    The random aspect of the batch make it imperfect but its execution is very fast,
    impact on training speed is minimum but impact on score is great.
    Experiment shown that the ratio f a mini-batch could go from 0.07 up to 0.8.

    The balancing is do that why in order to keep the efficiency of chunked HDF files
    """
    valid_batch_size = [16, 32, 64, 128, 256]

    if batch_size == 16:
        repeat = 14
    elif batch_size == 32:
        repeat = 28
    elif batch_size == 64:
        repeat = 57
    elif batch_size == 128:
        repeat = 114
    elif batch_size == 256:
        repeat = 230
    else:
        raise ValueError(
            f"Balancer is configure to work with batch_size equal to {valid_batch_size}"
        )

    print(f'parameters "repeat" set to: {repeat}')

    def balance(data: list):
        def get_first_neg(idx):
            for i in range(len(balance.pool_y)):
                if balance.pool_y[i][idx] == 0:
                    return i
            return None

        x = np.asarray([d[0] for d in data])
        y = np.asarray([d[1] for d in data])

        # If the pool is not full, can't perform balancing
        if len(balance.pool_y) < pool_size:
            balance.pool_x.extend(x)
            balance.pool_y.extend(y)

        # Find the class that is over-represented
        for _ in range(repeat):
            class_idx = np.argmax(np.sum(y, axis=0))
            to_remove = np.where(y[:, class_idx] == 1)[0]

            if len(to_remove != 0):
                to_remove = np.random.choice(to_remove)

                # chose a file from the pool that is not from the class.
                # consume the pool's sample to unsure the file isn't present twice
                pool_idx = get_first_neg(class_idx)
                if pool_idx is not None:
                    x[to_remove] = balance.pool_x.pop(pool_idx)
                    y[to_remove] = balance.pool_y.pop(pool_idx)

        return torch.from_np(x), torch.from_np(y)

    balance.pool_x = []
    balance.pool_y = []

    return balance


# =============================================================================
#
#      SUBSET SPLIT FUNCTIONS
#
# =============================================================================
@functools.lru_cache(maxsize=5)
def get_all_targets(dataset):
    """Load the targets corresponding to the version of the dataset used.

    When using unbalanced set, the targets are loaded from a file for performance reason
    When using balanced set, the targets are fetch directly from the the HDF file 1 by 1
    """
    all_targets = []

    if dataset.version == "unbalanced":
        return dataset.targets

    for i in trange(len(dataset)):
        all_targets.append(dataset.get_target(i))

    return all_targets


def get_class_sum(all_targets, batch_indexes: list):
    all_targets = np.asarray(all_targets)
    all_indexes = np.hstack(batch_indexes)

    return np.sum(all_targets[all_indexes], axis=0)


def display_statistics(stats: list):
    cols = list(stats[0].keys())

    header_form = " {:^12.12} |" * len(cols)
    value_form = " {:<12.4e} |" * len(cols)

    print(header_form.format(*cols))
    print("-" * 13 * (len(cols) + 1))

    for stat in stats:
        values = list(stat.values())
        value_form = ""

        for v in values:
            if isinstance(v, int) and len(str(v)) > 12:
                value_form += " {:>12.4e} |"
            elif isinstance(v, float):
                value_form += " {:>12.4f} |"
            elif isinstance(v, int):
                value_form += " {:>12d} |"
            else:
                pass

        print(value_form.format(*values))


def get_class_statistic(dataset, batch_indexes: list):
    """Using a list of number of occurence per class, compute the basic statistic needed to evaluate the quality of the split"""
    all_targets = get_all_targets(dataset)
    total_sum = np.sum(all_targets, axis=0)
    class_sum = get_class_sum(all_targets, batch_indexes)

    dist = class_sum / total_sum

    mean = np.mean(dist)
    std = np.std(dist)
    mini = np.min(dist)
    maxi = np.max(dist)
    occur = int(sum(class_sum))
    missing = len(class_sum[class_sum == 0])

    only_positive = class_sum[class_sum > 0]
    ratio = min(only_positive.astype(float)) / max(only_positive.astype(float))

    return (
        class_sum,
        dist,
        dict(
            mean=mean,
            std=std,
            mini=mini,
            maxi=maxi,
            ratio=ratio,
            occur=occur,
            missing=missing,
        ),
    )


@cache_to_disk(path=None)
def class_balance_split(
    dataset,
    supervised_ratio: float = 0.1,
    unsupervised_ratio: float = None,
    batch_size: int = 64,
    verbose: bool = True,
    seed: int = 1234,
):
    """Perform supervised / unsupervised split "equally" distributed within each class.
    In order to achieve some balancing, the "chunked" HDf features can't be used. Each files
    will be fetch individually. The speed will be greatly reduce.
    """

    def fill_subset(remaining_samples, expected):
        subset_occur = np.zeros(shape=(527,))
        subset = []

        with tqdm(total=sum(expected)) as progress:
            for class_idx in range(527):
                idx = 0
                while (
                    idx < len(remaining_samples)
                    and subset_occur[class_idx] < expected[class_idx]
                ):
                    if remaining_samples[idx][0][class_idx] == 1:
                        target, target_idx = remaining_samples.pop(idx)
                        subset_occur += target
                        subset.append(target_idx)
                        progress.update(sum(target))

                    idx += 1

        return np.asarray(subset), remaining_samples

    if unsupervised_ratio is None:
        unsupervised_ratio = 1 - supervised_ratio

    assert 0.0 <= supervised_ratio <= 1.0
    assert 0.0 <= unsupervised_ratio <= 1.0
    assert supervised_ratio + unsupervised_ratio <= 1.0

    batch_sampler = ChunkAlignSampler(dataset, batch_size=batch_size, shuffle=False)
    all_batches = [batch_id for batch_id in batch_sampler]

    if supervised_ratio == 1.0:
        return list(range(len(dataset))), []

    # get all dataset targets and compute original class distribution metrics
    if dataset.version == "unbalanced":
        all_targets = dataset.targets
    else:
        all_targets = get_all_targets(dataset)

    all_targets_idx = list(range(len(all_targets)))
    _, _, original_statistics = get_class_statistic(dataset, all_batches)

    # expected occurance and tolerance
    total_occur = np.sum(all_targets, axis=0)
    s_expected_occur = np.ceil(total_occur * supervised_ratio)
    u_expected_occur = np.ceil(total_occur * unsupervised_ratio)
    print("s expected occur: ", sum(s_expected_occur))

    # loop through the dataset and constitute the two subset.
    remaining_sample = list(zip(all_targets, all_targets_idx))
    s_subset, remaining_sample = fill_subset(remaining_sample, s_expected_occur)
    #     s_batches = _split(s_subset)
    s_batches = s_subset

    # For the unsupervised subset, if automatic set, then it is the remaining samples
    if unsupervised_ratio + supervised_ratio == 1.0:
        u_subset = np.asarray([s[1] for s in remaining_sample])

    else:
        u_subset, _ = fill_subset(remaining_sample, u_expected_occur)

    #     u_batches = _split(u_subset)
    u_batches = u_subset

    # Compute the statistics for the supervised and unsupservised splits
    _, _, s_stats = get_class_statistic(dataset, s_batches)
    _, _, u_stats = get_class_statistic(dataset, u_batches)

    if verbose:
        print(display_statistics([original_statistics, s_stats, u_stats]))

    return s_batches, u_batches


class BatchSamplerFromList(Sampler):
    """Sample batch from a list of batches"""

    def __init__(self, batches: List[List]):
        self.batches = batches

    def __iter__(self):
        random.shuffle(self.batches)

        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def get_supervised(version: str = "unbalanced", **kwargs):
    def supervised(
        dataset_root: str,
        rdcc_nbytes: int = 512 * 1024 ** 2,
        data_shape: tuple = (64, 500,),
        data_key: str = "data",
        train_transform: Optional[nn.Module] = None,
        val_transform: Optional[nn.Module] = None,
        batch_size: int = 64,
        supervised_ratio: float = 1.0,
        unsupervised_ratio: float = None,
        balance: bool = True,
        num_workers: int = 10,
        pin_memory: bool = False,
        **kwargs,
    ) -> Tuple[DataLoader, DataLoader]:

        # Dataset parameters
        d_params = dict(
            root=os.path.join(dataset_root, "AudioSet/hdfs/mel_64x500"),
            rdcc_nbytes=rdcc_nbytes,
            data_shape=data_shape,
            data_key=data_key,
        )

        # Dataloader parameters
        l_params = dict(num_workers=num_workers, pin_memory=pin_memory,)

        # validation subset
        val_dataset = SingleAudioset(
            **d_params, version="eval", transform=val_transform
        )
        #         val_indexes = list(range(len(val_dataset)))
        #         val_sampler = SingleBalancedSampler(val_dataset, val_indexes, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, **l_params)

        # Training subset
        train_dataset = SingleAudioset(
            **d_params, version=version, transform=train_transform
        )

        if supervised_ratio == 1.0:
            train_indexes = list(range(len(train_dataset)))
            train_sampler = SingleBalancedSampler(
                train_dataset, train_indexes, shuffle=True
            )
            s_train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler, **l_params
            )

        else:
            if unsupervised_ratio is None:
                unsupervised_ratio = 1 - supervised_ratio

            s_indexes, u_indexes = class_balance_split(
                train_dataset,
                supervised_ratio=supervised_ratio,
                unsupervised_ratio=unsupervised_ratio,
                batch_size=batch_size,
                verbose=True,
            )

            s_batch_sampler = SingleBalancedSampler(
                train_dataset, s_indexes, shuffle=True
            )
            #             u_batch_sampler = SingleBalancedSampler(train_dataset, u_indexes, shuffle=True)

            s_train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=s_batch_sampler,
                **l_params,
            )
        #             u_train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=u_batch_sampler, **l_params)

        #             train_loader = ZipCycle([s_train_loader, u_train_loader])

        return None, s_train_loader, val_loader

    return supervised


def get_mean_teacher(version: str = "unbalanced", **kwargs):
    def mean_teacher(
        dataset_root: str,
        rdcc_nbytes: int = 512 * 1024 ** 2,
        data_shape: tuple = (64, 500,),
        data_key: str = "data",
        train_transform: Optional[nn.Module] = None,
        val_transform: Optional[nn.Module] = None,
        batch_size: int = 64,
        supervised_ratio: float = 1.0,
        unsupervised_ratio: float = None,
        balance: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False,
        **kwargs,
    ) -> Tuple[DataLoader, DataLoader]:

        # Dataset parameters
        d_params = dict(
            root=os.path.join(dataset_root, "AudioSet/hdfs/mel_64x500"),
            rdcc_nbytes=rdcc_nbytes,
            data_shape=data_shape,
            data_key=data_key,
        )

        # Dataloader parameters
        l_params = dict(num_workers=num_workers, pin_memory=pin_memory,)

        # validation subset
        val_dataset = SingleAudioset(
            **d_params, version="eval", transform=val_transform
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, **l_params)

        # Training subset
        train_dataset = SingleAudioset(
            **d_params, version=version, transform=train_transform
        )

        # Calc the size of the Supervised and Unsupervised
        if unsupervised_ratio is None:
            unsupervised_ratio = 1 - supervised_ratio

        s_idx, u_idx = class_balance_split(
            train_dataset,
            supervised_ratio=supervised_ratio,
            unsupervised_ratio=unsupervised_ratio,
            batch_size=batch_size,
            verbose=True,
        )

        s_batch_size = int(np.floor(batch_size * supervised_ratio))
        u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

        s_sampler = SingleBalancedSampler(train_dataset, s_idx, shuffle=True)
        u_sampler = SingleBalancedSampler(train_dataset, u_idx, shuffle=True)

        s_train_loader = DataLoader(
            train_dataset, batch_size=s_batch_size, sampler=s_sampler, **l_params
        )
        u_train_loader = DataLoader(
            train_dataset, batch_size=u_batch_size, sampler=u_sampler, **l_params
        )

        train_loader = ZipCycleInfinite([s_train_loader, u_train_loader])

        return None, train_loader, val_loader

    return mean_teacher


def get_dct(version: str = "unbalanced", **kwargs):
    def dct(
        dataset_root: str,
        rdcc_nbytes: int = 512 * 1024 ** 2,
        data_shape: tuple = (64, 500,),
        data_key: str = "data",
        train_transform: Optional[nn.Module] = None,
        val_transform: Optional[nn.Module] = None,
        batch_size: int = 64,
        supervised_ratio: float = 1.0,
        unsupervised_ratio: float = None,
        balance: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False,
        **kwargs,
    ) -> Tuple[DataLoader, DataLoader]:

        # Dataset parameters
        d_params = dict(
            root=os.path.join(dataset_root, "AudioSet/hdfs/mel_64x500"),
            rdcc_nbytes=rdcc_nbytes,
            data_shape=data_shape,
            data_key=data_key,
        )

        # Dataloader parameters
        l_params = dict(num_workers=num_workers, pin_memory=pin_memory,)

        # validation subset
        val_dataset = SingleAudioset(
            **d_params, version="eval", transform=val_transform
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, **l_params)

        # Training subset
        train_dataset = SingleAudioset(
            **d_params, version=version, transform=train_transform
        )

        # Calc the size of the Supervised and Unsupervised
        if unsupervised_ratio is None:
            unsupervised_ratio = 1 - supervised_ratio

        s_idx, u_idx = class_balance_split(
            train_dataset,
            supervised_ratio=supervised_ratio,
            unsupervised_ratio=unsupervised_ratio,
            batch_size=batch_size,
            verbose=True,
        )

        s_batch_size = int(np.floor(batch_size * supervised_ratio))
        u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

        s_sampler = SingleBalancedSampler(train_dataset, s_idx, shuffle=True)
        u_sampler = SingleBalancedSampler(train_dataset, u_idx, shuffle=True)

        s1_train_loader = DataLoader(
            train_dataset, batch_size=s_batch_size, sampler=s_sampler, **l_params
        )
        s2_train_loader = DataLoader(
            train_dataset, batch_size=s_batch_size, sampler=s_sampler, **l_params
        )
        u_train_loader = DataLoader(
            train_dataset, batch_size=u_batch_size, sampler=u_sampler, **l_params
        )

        train_loader = ZipCycleInfinite(
            [s1_train_loader, s2_train_loader, u_train_loader]
        )

        return None, train_loader, val_loader

    return dct


def get_fixmatch(version: str = "unbalanced", **kwargs):
    def fixmatch(
        dataset_root: str,
        rdcc_nbytes: int = 512 * 1024 ** 2,
        data_shape: tuple = (320000,),
        data_key: str = "waveform",
        train_transform: Optional[nn.Module] = None,
        val_transform: Optional[nn.Module] = None,
        batch_size: int = 64,
        supervised_ratio: float = 0.1,
        unsupervised_ratio: float = None,
        balance: bool = True,
        num_workers: int = 10,
        pin_memory: bool = False,
        seed: int = 1234,
        **kwargs,
    ) -> Tuple[DataLoader, DataLoader]:

        # Dataset parameters
        d_params = dict(
            root=os.path.join(dataset_root, "AudioSet/hdfs/"),
            rdcc_nbytes=rdcc_nbytes,
            data_shape=data_shape,
            data_key=data_key,
        )

        # Dataloader parameters
        l_params = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
        )

        # validation subset
        print("creating validation dataset")
        val_dataset = SingleAudioset(
            **d_params, version="eval", transform=val_transform
        )
        val_loader = DataLoader(val_dataset, **l_params)

        # Create two dataset for managing weak and strong augmentation
        print("creating weak and strong augmented train dataset")
        weak_transform, strong_transform = train_transform
        train_weak_dataset = SingleAudioset(
            **d_params, version=version, transform=weak_transform
        )
        train_strong_dataset = SingleAudioset(
            **d_params, version=version, transform=strong_transform
        )

        # Split the dataset into two S and U
        print("spliting the dataset")
        if unsupervised_ratio is None:
            unsupervised_ratio = 1 - supervised_ratio

        s_indexes, u_indexes = class_balance_split(
            train_weak_dataset,
            supervised_ratio=supervised_ratio,
            unsupervised_ratio=unsupervised_ratio,
            batch_size=batch_size,
            verbose=True,
        )

        # Create the sampler sor S and U, make use of torch generator to ensure that the samplers for U
        # will return the same indexes
        s_batch_sampler = SingleBalancedSampler(
            train_weak_dataset, s_indexes, shuffle=True
        )
        u_weak_batch_sampler = SubsetRandomSampler(
            u_indexes, generator=torch.Generator().manual_seed(seed)
        )
        u_strong_batch_sampler = SubsetRandomSampler(
            u_indexes, generator=torch.Generator().manual_seed(seed)
        )

        print("merging dataloaders")
        s_weak_train_loader = DataLoader(
            train_weak_dataset, sampler=s_batch_sampler, **l_params
        )
        u_weak_train_loader = DataLoader(
            train_weak_dataset, sampler=u_weak_batch_sampler, **l_params
        )
        u_strong_train_loader = DataLoader(
            train_strong_dataset, sampler=u_strong_batch_sampler, **l_params
        )

        train_loader = ZipCycleInfinite(
            [s_weak_train_loader, u_weak_train_loader, u_strong_train_loader]
        )

        return None, train_loader, val_loader

    return fixmatch
