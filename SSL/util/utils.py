#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import inspect
import itertools
import logging
import os
import pickle
import random
import time

from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple
from zipfile import ZipFile, ZIP_DEFLATED

import bz2
import numpy as np
import torch

from torch.utils.data.dataset import Dataset

# TODO write q timer decorator that deppend on the logging level


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.__getitem__  # type: ignore
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


def timeit_logging(func):
    def decorator(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        logging.info(
            "%s executed in: %.3fs" % (func.__name__, time.time() - start_time)
        )

    return decorator


class Cacher:
    def __init__(self, func: Callable):
        self.func = func
        self.cached_func = self.cache_wrapper(func)

    def __call__(self, *args, caching: bool = True, **kwargs):
        if caching:
            return self.cached_func(*args, **kwargs)

        return self.func(*args, **kwargs)

    def cache_wrapper(self, func):
        def decorator(*args, **kwargs):
            key = ",".join(map(str, kwargs.values()))

            if key not in decorator.cache:
                decorator.cache[key] = func(*args, **kwargs)

            return decorator.cache[key]

        decorator.cache = dict()
        return decorator


def get_training_printers(
    losses: Dict[str, Any], metrics: Dict[str, Any]
) -> Tuple[str, str, str, str]:
    assert isinstance(losses, dict)
    assert isinstance(metrics, dict)

    UNDERLINE_SEQ = "\033[1;4m"
    RESET_SEQ = "\033[0m"

    text_form = (
        ": epoch {:<6.6} ({:<6.6}/{:>6.6}) - "
        + "{:<8.8} " * len(losses)
        + " | "
        + "{:<8.8} " * len(metrics)
        + "{:<6.6}"
    )  # time

    value_form = (
        ": epoch {:<6d} ({:<6d}/{:>6d}) - "
        + "{:<8.4f} " * len(losses)
        + " | "
        + "{:<8.4f} " * len(metrics)
        + "{:<6.2f}"
    )

    header = " " * 5 + text_form.format(
        "", "", "", *losses.keys(), *metrics.keys(), "Time"
    )
    train_form = "TRAIN" + value_form
    val_form = UNDERLINE_SEQ + "VALID" + value_form + RESET_SEQ
    test_form = val_form.replace("VALID", "TEST ")

    return header, train_form, val_form, test_form


def get_train_format(framework: str = "supervised"):
    assert framework in [
        "supervised",
        "mean-teacher",
        "dct",
        "audioset-sup",
        "audioset-fixmatch",
        "compare2021-prs-sup",
    ]

    UNDERLINE_SEQ = "\033[1;4m"
    RESET_SEQ = "\033[0m"

    if framework == "supervised":
        header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} - {:<9.9} {:<12.12}| {:<9.9}- {:<6.6}"
        value_form = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} - {:<9.9} {:<10.4f}| {:<9.4f}- {:<6.4f}"

        header = header_form.format(
            ".               ",
            "Epoch",
            "%",
            "Losses:",
            "ce",
            "metrics: ",
            "acc",
            "F1 ",
            "Time",
        )

    elif framework == "supervised-compare2021-prs":
        header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} - {:<9.9} {:<12.12}| {:<9.9}- {:<6.6}"
        value_form = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} - {:<9.9} {:<10.4f}| {:<9.4f}- {:<6.4f}"

        header = header_form.format(
            ".               ",
            "Epoch",
            "%",
            "Losses:",
            "ce",
            "metrics: ",
            "acc",
            "F1",
            "mAP",
            "Time",
        )

    elif framework == "mean-teacher":
        header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<10.8} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} | {:<10.8} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} - {:<8.6}"
        value_form = "{:<8.8} {:<6d} - {:<6d} - {:<10.8} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} | {:<10.8} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} - {:<8.4f}"
        header = header_form.format(
            ".               ",
            "Epoch",
            "%",
            "Student:",
            "ce",
            "ccost",
            "acc_s",
            "f1_s",
            "acc_u",
            "f1_u",
            "Teacher:",
            "ce",
            "acc_s",
            "f1_s",
            "acc_u",
            "f1_u",
            "Time",
        )

    elif framework == "dct":
        header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} | {:<6.6} | {:<6.6} | {:<6.6} - {:<9.9} {:<9.9} | {:<9.9}- {:<6.6}"
        value_form = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} - {:<9.9} {:<9.4f} | {:<9.4f}- {:<6.4f}"

        header = header_form.format(
            "",
            "Epoch",
            "%",
            "Losses:",
            "Lsup",
            "Lcot",
            "Ldiff",
            "total",
            "metrics: ",
            "acc_s1",
            "acc_u1",
            "Time",
        )

    elif framework == "audioset-sup":
        header_form = "{:<16.16} {:<5.5} - {:<5.5} / {:<5.5} - {:<7.7} {:<9.9} - {:<8.8} {:<12.12} {:<12.12} {:<12.12} - {:<6.6}"
        value_form = "{:<16.16} {:<5} - {:>5} / {:<5} - {:7.7} {:<9.4f} - {:<8.8} {:<12.3e} {:<12.3e} {:<12.3e} - {:<6.4f}"

        header = header_form.format(
            ".               ",
            "Epoch",
            "",
            "",
            "Losses:",
            "ce",
            "metrics: ",
            "acc",
            "F1",
            "mAP",
            "Time",
        )

    elif framework == "compare2021-prs-sup":
        return get_train_format("audioset-sup")

    elif framework == "audioset-fixmatch":
        header_form = "{:<16.16} {:<5.5} - {:<5.5} / {:<5.5} - {:<7.7} {:<9.9} - {:<8.8} {:<12.12} {:<12.12} {:<12.12} {:<12.12} {:<12.12} - {:<6.6}"
        value_form = "{:<16.16} {:<5} - {:>5} / {:<5} - {:7.7} {:<9.4f} - {:<8.8} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.3e} - {:<6.4f}"

        header = header_form.format(
            ".               ",
            "Epoch",
            "",
            "",
            "Losses:",
            "ce",
            "metrics: ",
            "acc_s",
            "F1_s",
            "acc_u",
            "F1_u",
            "mAP",
            "Time",
        )
    else:
        raise ValueError(f"Invalid argument {framework=}.")

    train_form = value_form
    val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

    return header, train_form, val_form


def cache_to_disk(path: Optional[str] = None) -> Callable:
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create a unique name for the cache base on the function arguments
            key = ".cache_" + "_".join(["=".join(map(str, i)) for i in kwargs.items()])
            path_ = key

            if path is not None:
                os.makedirs(path, exist_ok=True)
                path_ = os.path.join(path, key)

            # file do not exist, execute function, save result in file
            print("cache path: ", str(path_))
            if not os.path.isfile(path_):
                print("split not ready, generating ...")
                data = func(*args, **kwargs)

                with bz2.BZ2File(path_, "wb") as f:
                    print("saving split in cache file")
                    pickle.dump(data, f)

            # File exist, read file content
            else:
                print("split ready, loading cache file")
                with bz2.BZ2File(path_, "rb") as f:
                    data = pickle.load(f)

            return data

        return wrapper

    return decorator


def conditional_cache_v2(func):
    def decorator(*args, **kwargs):
        key = kwargs.get("key", None)
        cached = kwargs.get("cached", None)

        if cached is not None and key is not None:
            if key not in decorator.cache.keys():
                decorator.cache[key] = func(*args, **kwargs)

            return decorator.cache[key]

        return func(*args, **kwargs)

    decorator.cache = dict()

    return decorator


def cache_feature(func):
    def decorator(*args, **kwargs):
        key = ",".join(map(str, args + tuple(kwargs.values())))

        if key not in decorator.cache:
            decorator.cache[key] = func(*args, **kwargs)

        return decorator.cache[key]

    decorator.cache = dict()
    decorator.func = func
    return decorator


def track_maximum():
    def func(key, value):
        if key not in func.max:
            func.max[key] = value
        else:
            if func.max[key] < value:
                func.max[key] = value
        return func.max[key]

    func.max = dict()
    return func


def get_datetime(sep: str = "-") -> str:
    now = datetime.datetime.now()
    return now.strftime(f"%Y{sep}%m{sep}%d_%H{sep}%M{sep}%S")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_model_from_name(model_name: str):
    import SSL.models.ubs8k as ubs8k_models
    import SSL.models.ubs8k_test as ubs8k_models_test
    import SSL.models.cifar10 as cifar10_models

    all_members = []
    for module in [ubs8k_models, ubs8k_models_test, cifar10_models]:
        all_members += inspect.getmembers(module)

    for name, obj in all_members:
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__name__ == model_name:
                logging.info("Model loaded: %s" % model_name)
                return obj

    msg = "This model does not exist: %s\n" % model_name
    msg += "Available models are: %s" % [
        name
        for name, obj in all_members
        if inspect.isclass(obj) or inspect.isfunction(obj)
    ]
    raise AttributeError("This model does not exist: %s " % msg)


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


class ZipCycle:
    """
    Zip through a list of iterables and sized objects of different lengths.
    When a iterable smaller than the longest is over, this iterator is reset to the beginning.

    Example :
    ```
    r1 = range(1, 4)
    r2 = range(1, 6)
    iters = ZipCycle([r1, r2])
    for v1, v2 in iters:
        print(v1, v2)
    ```

    will print :
    ```
    1 1
    2 2
    3 3
    1 4
    2 5
    ```
    """

    def __init__(self, iterables: Iterable, align: str = "max"):
        assert align in ("min", "max")

        iterables = list(iterables)
        for iterable in iterables:
            if len(iterable) == 0:
                raise RuntimeError("An iterable is empty.")

        self._iterables = iterables

        f = max if align == "max" else min
        self._len = f([len(iterable) for iterable in self._iterables])

    def __iter__(self) -> Generator:
        cur_iters = [iter(iterable) for iterable in self._iterables]
        cur_count = [0 for _ in self._iterables]

        for _ in range(len(self)):
            items = []

            for i, _ in enumerate(cur_iters):
                if cur_count[i] >= len(self._iterables[i]):
                    cur_iters[i] = iter(self._iterables[i])

                item = next(cur_iters[i])
                cur_count[i] += 1

                items.append(item)

            yield items

    def __len__(self) -> int:
        return self._len


class ZipDataset(Dataset):
    def __init__(self, *datasets: Dataset,) -> None:
        if len(datasets) > 0 and any(
            len(dset) != len(datasets[0]) for dset in datasets  # type: ignore
        ):
            raise ValueError("Invalid datasets lengths for ZipDatasets.")

        super().__init__()
        self._datasets = datasets

    def __getitem__(self, index: int) -> List:
        item = []
        for dset in self._datasets:
            item.append(dset[index])
        return item

    def __len__(self) -> int:
        return min(map(len, self._datasets))  # type: ignore


class ZipCycleInfinite(ZipCycle):
    def __init__(self, iterables: List[Iterable]) -> None:
        super().__init__(iterables)

    def __iter__(self) -> Generator:
        infinite_iters = [itertools.cycle(it) for it in self._iterables]

        while True:
            items = [next(inf_it) for inf_it in infinite_iters]
            yield items


def create_bash_crossvalidation(nb_fold: int = 10):
    cross_validation = []
    end = nb_fold

    for i in range(nb_fold):
        train_folds = []
        start = i

        for i in range(nb_fold - 1):
            start = (start + 1) % nb_fold
            start = start if start != 0 else nb_fold
            train_folds.append(start)

        cross_validation.append(
            "-t " + " ".join(map(str, train_folds)) + " -v %d" % end
        )
        end = (end % nb_fold) + 1
        end = end if end != 0 else nb_fold

    print(";".join(cross_validation))


def save_source_as_img(sourcepath: str):
    # Create a zip file of the current source code

    with ZipFile(
        sourcepath + ".zip", "w", compression=ZIP_DEFLATED, compresslevel=9
    ) as myzip:
        myzip.write(sourcepath)

    # Read the just created zip file and store it into
    # a uint8 numpy array
    with open(sourcepath + ".zip", "rb") as myzip:
        zip_bin = myzip.read()

    zip_bin_n = np.array(list(map(int, zip_bin)), dtype=np.uint8)

    # Convert it into a 2d matrix
    desired_dimension = 500
    missing = desired_dimension - (zip_bin_n.size % desired_dimension)
    zip_bin_p = np.array(
        np.concatenate((zip_bin_n, np.array([0] * missing, dtype=np.uint8)))
    )
    zip_bin_i = np.array(zip_bin_p).reshape(
        (desired_dimension, zip_bin_p.size // desired_dimension)
    )

    # Cleaning (remove zip file)
    os.remove(sourcepath + ".zip")

    return zip_bin_i, missing


# from PIL import Image
#
# im = Image.open("tmp-232.png")
# source_bin = numpy.asarray(im, dtype=numpy.uint8)
# source_bin = source_bin[:, :, 0]
#
# source_bin = source_bin.flatten()[:-232]
#
# with open("student-teacher.ipynb.zip.bak", "wb") as mynewzip:
#     mynewzip.write(source_bin)
