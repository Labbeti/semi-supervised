#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math
import os
import random

from typing import Any, Callable, Iterable, List, Optional, Tuple, Type

import numpy as np
import soundfile
import torchaudio

from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

from SSL.dataset.gsc import SPEECHCOMMANDS
from SSL.util.utils import Cacher, ZipCycle, ZipDataset


URL = "speech_commands_v0.02"
NOISE_FOLDER = "_background_noise_"
SILENCE_FOLDER = "silence"
EXCEPT_FOLDERS = ["_background_noise_", "silence"]

target_mapper = {
    "bed": 0,
    "bird": 1,
    "cat": 2,
    "dog": 3,
    "down": 4,
    "eight": 5,
    "five": 6,
    "follow": 7,
    "forward": 8,
    "four": 9,
    "go": 10,
    "happy": 11,
    "house": 12,
    "learn": 13,
    "left": 14,
    "marvin": 15,
    "nine": 16,
    "no": 17,
    "off": 18,
    "on": 19,
    "one": 20,
    "right": 21,
    "seven": 22,
    "sheila": 23,
    "six": 24,
    "stop": 25,
    "three": 26,
    "tree": 27,
    "two": 28,
    "up": 29,
    "visual": 30,
    "wow": 31,
    "yes": 32,
    "zero": 33,
    "backward": 34,
}
all_classes = target_mapper

# =============================================================================
# UTILITY FUNCTION
# =============================================================================


def _split_s_u(train_dataset: SPEECHCOMMANDS, s_ratio: float = 1.0) -> Tuple[List[int], List[int]]:
    _train_dataset = SpeechCommandsStats.from_dataset(train_dataset)

    nb_class = len(target_mapper)
    dataset_size = len(_train_dataset)

    if s_ratio == 1.0:
        return list(range(dataset_size)), []

    s_idx, u_idx = [], []
    nb_s = int(math.ceil(dataset_size * s_ratio) // nb_class)
    cls_idx = [[] for _ in range(nb_class)]

    # To each file, an index is assigned, then they are split into classes
    for i in trange(dataset_size):
        y, _, _ = _train_dataset[i]
        cls_idx[y].append(i)

    # Recover only the s_ratio % first as supervised, rest is unsupervised
    for i in trange(len(cls_idx)):
        random.shuffle(cls_idx[i])
        s_idx += cls_idx[i][:nb_s]
        u_idx += cls_idx[i][nb_s:]

    return s_idx, u_idx


def cache_feature(func: Callable):
    def decorator(*args, **kwargs):
        key = ",".join(map(str, args))

        if key not in decorator.cache:
            decorator.cache[key] = func(*args, **kwargs)

        return decorator.cache[key]

    decorator.cache = dict()
    decorator.func = func
    return decorator


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(
        self,
        root: str,
        subset: str = "train",
        url: str = URL,
        download: bool = False,
        transform: Optional[nn.Module] = None,
        cache: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(root, url, download)

        assert subset in ["train", "validation", "testing"]
        self.subset = subset
        self.transform_ = transform
        self.cache = cache
        self.root_path = self._walker[0].split("/")[:-2]

        self._keep_valid_files()

        # Prepare the cached method
        if cache:
            self.cached_getitem = Cacher(self._cacheable_getitem)
            self.cached_transform = Cacher(self._cacheable_transform)
        else:
            self.cached_getitem = self._cacheable_getitem
            self.cached_transform = self._cacheable_transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        data, label = self.cached_getitem(idx=idx)
        data = self.cached_transform(data, key=idx)
        return data, label

    def _cacheable_getitem(self, idx: int) -> Tuple:
        waveform, _, label, _, _ = super().__getitem__(idx)
        return waveform, target_mapper[label]

    def _cacheable_transform(self, x, key=None):
        if self.transform_ is not None:
            return self.transform_(x)
        return x

    def _keep_valid_files(self):
        # bn = os.path.basename
        def basename(x: str) -> str:
            return "/".join(x.split("/")[-2:])

        def file_list(filename: str) -> Iterable[str]:
            path = os.path.join(self._path, filename)
            with open(path, "r") as f:
                to_keep = f.read().splitlines()
            return dict.fromkeys(to_keep)

        # Recover file list for validaiton and testing.
        validation_list = file_list("validation_list.txt")
        testing_list = file_list("testing_list.txt")

        training_list = [
            basename(path)
            for path in self._walker
            if basename(path) not in validation_list
            and basename(path) not in testing_list
        ]

        if self.subset == "train":
            for p in training_list:
                if p in validation_list:
                    print("%s is train and validation" % p)
                    raise ValueError()

                if p in testing_list:
                    print("%s is in both train and testing list" % p)
                    raise ValueError()

        # Map the list to the corresponding subsets
        mapper = {
            "train": training_list,
            "validation": validation_list,
            "testing": testing_list,
        }

        # mapper[self.subset] : ["backward/file.wav", ...]
        self._walker = [
            os.path.join(self._path, path) for path in mapper[self.subset]
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(subset={self.subset})"


class SpeechCommandAug(SpeechCommands):
    def __init__(
        self,
        root: str,
        subset: str = "train",
        url: str = URL,
        download: bool = False,
        transform: Optional[nn.Module] = None,
        cache: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(root, subset, url, download, cache=cache, transform=transform)

        assert subset in ["train", "validation", "testing"]
        self.enable_cache = cache
        self.subset = subset
        self.root_path = self._walker[0].split("/")[:-2]

        self._keep_valid_files()

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        # loading waveform from disk can always be cached
        waveform, mapped_target = self.cached_getitem(index)

        # applying transformation
        data = waveform
        # if self.cached_getitem.func is not None:
        if isinstance(self.cached_getitem, Callable):
            data = self.cached_transform(data)
            data = data.squeeze()

        return data, mapped_target


class SpeechCommandsStats(SpeechCommands):
    @classmethod
    def from_dataset(cls, dataset: SPEECHCOMMANDS) -> "SpeechCommandsStats":
        root = dataset.root
        newone = cls(root=root)
        newone.__dict__.update(dataset.__dict__)
        return newone

    def _load_item(self, filepath: str, path: str) -> Tuple[str, str, int]:
        HASH_DIVIDER = "_nohash_"
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
        speaker, _ = os.path.splitext(filename)

        speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
        utterance_number = int(utterance_number)

        # remove Load audio
        return label, speaker_id, utterance_number

    def __getitem__(self, index: int) -> Tuple[int, str, int]:
        fileid = self._walker[index]
        label, speaker_id, utterance_number = self._load_item(fileid, self._path)
        return target_mapper[label], speaker_id, utterance_number


class SpeechCommand10(SpeechCommands):
    TRUE_CLASSES = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "off",
        "on",
        "go",
        "stop",
    ]

    def __init__(
        self,
        root: str,
        subset: str = "train",
        url: str = URL,
        download: bool = False,
        transform: Optional[nn.Module] = None,
        percent_to_drop: float = 0.5,
    ) -> None:
        super().__init__(root, subset, url, download, transform)

        assert 0.0 <= percent_to_drop < 1.0

        self.percent_to_drop = percent_to_drop

        self.target_mapper = {
            "yes": 0,
            "no": 1,
            "up": 2,
            "down": 3,
            "left": 4,
            "right": 5,
            "off": 6,
            "on": 7,
            "go": 8,
            "stop": 9,
            "silence": 10,
            "unknown": 11,
        }

        # the rest of the classes belong to the "junk / trash / poubelle class"
        for cmd in all_classes:
            if cmd not in self.target_mapper:
                self.target_mapper[cmd] = 11

        self.drop_some_trash()
        self.add_silence()

    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        filepath = self._walker[index]

        label = filepath.split("/")[-2]
        target = self.target_mapper[label]

        waveform, _ = super().__getitem__(index)

        return waveform, target

    def drop_some_trash(self):
        def is_trash(path):
            if self.target_mapper[path.split("/")[-2]] == 11:
                return True

            return False

        # Create the complete list of trash class
        trash_list = [path for path in self._walker if is_trash(path)]

        # choice only x% of it that will be removed
        nb_to_drop = int(len(trash_list) * self.percent_to_drop)
        to_drop = np.random.choice(trash_list, size=nb_to_drop, replace=False)

        # remove it from the _walker
        self._walker = list(set(self._walker) - set(to_drop))

        print("%d out of %s junk files were drop." % (len(to_drop), len(trash_list)))

    def add_silence(self):
        """For the class silence, a new directory is created called "silence"
        It will contain 1 seconds segment from the _background_noise_ directory
        If the directory already exist, do some verification and pass
        """
        silence_dir = os.path.join(*self.root_path, "silence")

        if os.path.isdir(silence_dir):
            self._check_silence_class()

        else:
            self._create_silence_class()
            self._check_silence_class()

    def _create_silence_class2(self):
        print("Silence class doesn't exist")
        silence_dir = os.path.join(*self.root_path, "silence")
        noise_path = os.path.join(*self.root_path, "_background_noise_")

        # the silence class directory doesn't exist, create it
        os.makedirs(silence_dir)

        # Split each noise files into 1 second long segment
        to_process = []
        for file in os.listdir(os.path.join(*self.root_path, NOISE_FOLDER)):
            if file[-4:] == ".wav":
                to_process.append(os.path.join(noise_path, file))

        # Basic way, split each files into 1 seconds long segment
        print("Creating silence samples...")
        for filepath in to_process:
            basename = os.path.basename(filepath)

            waveform, sr = torchaudio.load(filepath)  # type: ignore

            nb_full_segment = int(len(waveform[0]) / sr)
            rest = len(waveform[0]) % sr
            segments = np.split(waveform[0][:-rest], nb_full_segment)

            # write each segment as a wav file with a unique name
            for i, s in enumerate(segments):
                unique_id = f"{basename[:-4]}_nohash_{i}.wav"
                path = os.path.join(silence_dir, unique_id)
                soundfile.write(path, s, sr)
        print("done")

    def _create_silence_class(self):
        print("Silence class doesn't exist")
        silence_dir = os.path.join(*self.root_path, "silence")
        noise_path = os.path.join(*self.root_path, "_background_noise_")

        # the silence class directory doesn't exist, create it
        os.makedirs(silence_dir)

        # Split each noise files into 1 second long segment
        to_process = []
        for file in os.listdir(os.path.join(*self.root_path, NOISE_FOLDER)):
            if file[-4:] == ".wav":
                to_process.append(os.path.join(noise_path, file))

        # Basic way, split each files into 1 seconds long segment
        print("Creating silence samples...")
        for filepath in to_process:
            basename = os.path.basename(filepath)

            waveform, sr = torchaudio.load(filepath)  # type: ignore

            # write each segment, we will create 300 segments of 1 secondes
            start_timestamps = np.random.randint(0, len(waveform[0]) - sr, size=400)
            for i, st in enumerate(start_timestamps):
                unique_id = f"{basename[:-4]}_nohash_{i}.wav"
                path = os.path.join(silence_dir, unique_id)

                segment = waveform[0][st : st + sr]
                soundfile.write(path, segment, sr)

        print("done")

    def _check_silence_class(self):
        silence_dir = os.path.join(*self.root_path, "silence")
        all_files = os.listdir(silence_dir)

        print("Silence class already processed")
        print("%s samples present" % len(all_files))


def dct(
    dataset_root: str,
    supervised_ratio: float = 0.1,
    batch_size: int = 100,
    train_transform_s: Optional[nn.Module] = None,
    train_transform_u: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    download: bool = True,
    dataset_class: Type[SpeechCommands] = SpeechCommands,
    num_workers: int = 0,
    pin_memory: bool = False,
    verbose: int = 1,
    train_folds: Any = None,
    val_folds: Any = None,
) -> Tuple[None, Iterable, DataLoader, DataLoader]:

    loader_args = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Validation subset
    val_dataset = dataset_class(
        root=dataset_root, subset="validation", transform=val_transform, download=download
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )
    test_dataset = SpeechCommandAug(
        root=dataset_root,
        subset="testing",
        transform=val_transform,
        download=download,
        percent_to_drop=0.0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )

    # Training subset
    train_dataset_s = dataset_class(
        root=dataset_root, subset="train", transform=val_transform, download=download
    )
    train_dataset_u = copy.deepcopy(train_dataset_s)
    train_dataset_u.transform = train_transform_u

    s_idx, u_idx = _split_s_u(train_dataset_s, supervised_ratio)

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

    return None, train_loader, val_loader, test_loader


def dct_uniloss(
    **kwargs,
) -> Tuple[None, Iterable, DataLoader, DataLoader]:
    return dct(**kwargs)


def mean_teacher(
    dataset_root: str,
    supervised_ratio: float = 0.1,
    batch_size: int = 128,
    has_same_trans: bool = True,
    student_transform: Optional[nn.Module] = None,
    teacher_transform: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_class: Type[SpeechCommands] = SpeechCommandAug,
    verbose: int = 1,
    train_folds: Any = None,
    val_folds: Any = None,
) -> Tuple[None, ZipCycle, DataLoader, DataLoader]:
    """
    Load the SpeechCommand for a student teacher learning
    """
    loader_args = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataset_path = os.path.join(dataset_root)

    # validation subset
    val_dataset = dataset_class(
        root=dataset_path,
        subset="validation",
        transform=val_transform,
        download=download,
        percent_to_drop=0.0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )
    # Testing subset
    test_dataset = dataset_class(
        root=dataset_path,
        subset="testing",
        transform=val_transform,
        download=download,
        percent_to_drop=0.0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )

    # Training subset
    train_student_dataset = dataset_class(
        root=dataset_path,
        subset="train",
        transform=student_transform,
        download=download,
        percent_to_drop=0.93,
    )

    if has_same_trans:
        train_dataset = train_student_dataset
    else:
        train_teacher_dataset = copy.deepcopy(train_student_dataset)
        train_teacher_dataset.transform = teacher_transform
        train_dataset = ZipDataset(train_student_dataset, train_teacher_dataset)

    s_idx, u_idx = _split_s_u(train_student_dataset, supervised_ratio)

    s_batch_size = int(math.floor(batch_size * supervised_ratio))
    u_batch_size = int(math.ceil(batch_size * (1 - supervised_ratio)))

    sampler_s = SubsetRandomSampler(s_idx)
    sampler_u = SubsetRandomSampler(u_idx)

    train_s_loader = DataLoader(
        train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args
    )
    train_u_loader = DataLoader(
        train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args
    )

    train_loader = ZipCycle([train_s_loader, train_u_loader])
    return None, train_loader, val_loader, test_loader


# =============================================================================
#        SUPERVISED DATASETS
# =============================================================================
def supervised(
    dataset_root,
    supervised_ratio: float = 1.0,
    batch_size: int = 128,
    train_transform: Optional[nn.Module] = None,
    val_transform: Optional[nn.Module] = None,
    augmentation: Optional[str] = None,
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[None, DataLoader, DataLoader, DataLoader]:
    """
    Load the SppechCommand for a supervised training
    """
    use_cache = True
    if augmentation is not None:
        use_cache = False
        print("Augmentation are used, disabling transform cache ...")

    loader_args = {"num_workers": num_workers, "pin_memory": pin_memory}
    dataset_path = os.path.join(dataset_root)

    # validation subset
    val_dataset = SpeechCommands(
        root=dataset_path,
        subset="validation",
        transform=val_transform,
        cache=True,
        download=download,
        percent_to_drop=0.0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )
    # Testing subset
    test_dataset = SpeechCommandAug(
        root=dataset_path,
        subset="testing",
        transform=val_transform,
        download=download,
        percent_to_drop=0.0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_args
    )

    # Training subset
    train_dataset = SpeechCommands(
        root=dataset_path,
        subset="train",
        transform=train_transform,
        cache=use_cache,
        download=download,
    )

    if supervised_ratio == 1.0:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **loader_args
        )

    else:
        s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)

        sampler_s = SubsetRandomSampler(s_idx)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler_s, **loader_args
        )

    return None, train_loader, val_loader, test_loader


def fixmatch(**kwargs):
    raise NotImplementedError
