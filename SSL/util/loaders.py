#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from SSL.util.utils import ZipCycle


def build_mapper(modules: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    dataset_mapper = dict()

    for dataset_name, dataset_module in modules.items():
        dataset_mapper[dataset_name] = {
            "supervised": dataset_module.supervised,
            "dct": dataset_module.dct,
            #             "dct_uniloss": dataset_module.dct_uniloss,
            "mean-teacher": dataset_module.mean_teacher,
            "fixmatch": dataset_module.fixmatch,
        }

    return dataset_mapper


def load_callbacks(dataset: str, framework: str, **kwargs) -> List[Callable]:
    import SSL.callbacks.esc as e
    import SSL.callbacks.ubs8k as u
    import SSL.callbacks.gsc as s
    import SSL.callbacks.audioset as a
    import SSL.callbacks.ComParE2021_PRS as c

    # get the corresping function mapper
    dataset_mapper = build_mapper(
        {
            "esc10": e,
            "esc50": e,
            "ubs8k": u,
            "gsc": s,
            "audioset-balanced": a,
            "audioset-unbalanced": a,
            "compare2021-prs": c,
        }
    )

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_optimizer(dataset: str, framework: str, **kwargs) -> Optimizer:
    import SSL.optimizer.esc as e
    import SSL.optimizer.ubs8k as u
    import SSL.optimizer.gsc as s
    import SSL.optimizer.audioset as a
    import SSL.optimizer.ComParE2021_PRS as c

    dataset_mapper = build_mapper(
        {
            "esc10": e,
            "esc50": e,
            "ubs8k": u,
            "gsc": s,
            "audioset-balanced": a,
            "audioset-unbalanced": a,
            "compare2021-prs": c,
        }
    )

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_preprocesser(
    dataset: str, framework: str, **kwargs
) -> Tuple[nn.Module, nn.Module]:
    import SSL.preprocessing.esc as esc
    import SSL.preprocessing.ubs8k as ubs8k
    import SSL.preprocessing.gsc as gsc
    import SSL.preprocessing.audioset as audioset
    import SSL.preprocessing.ComParE2021_PRS as prs

    dataset_mapper = build_mapper(
        {
            "esc10": esc,
            "esc50": esc,
            "ubs8k": ubs8k,
            "gsc": gsc,
            "audioset-balanced": audioset,
            "audioset-unbalanced": audioset,
            "compare2021-prs": prs,
        }
    )

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_dataset(
    dataset: str, framework: str, **kwargs
) -> Tuple[Any, ZipCycle, DataLoader, Optional[DataLoader]]:
    import SSL.dataset_loader.esc as esc
    import SSL.dataset_loader.gsc as gsc
    import SSL.dataset_loader.ubs8k as ubs8k
    import SSL.dataset_loader.audioset_balanced as a_bal
    import SSL.dataset_loader.audioset_unbalanced as a_unbal
    import SSL.dataset_loader.ComParE2021_PRS as c

    # Default dataset for audioset is the unsupervised version
    if dataset == "audioset":
        dataset = "audioset-unbalanced"

    dataset_mapper = build_mapper(
        {
            "esc10": esc,
            "esc50": esc,
            "ubs8k": ubs8k,
            "gsc": gsc,
            "audioset-balanced": a_bal,
            "audioset-unbalanced": a_unbal,
            "compare2021-prs": c,
        }
    )

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_helper(dataset: str, framework: str, mapper: dict, **kwargs):
    _dataset = dataset.lower()
    _framework = framework.lower()

    if _dataset not in mapper.keys():
        available_dataset = "{" + " | ".join(list(mapper.keys())) + "}"
        raise ValueError(
            f"dataset {_dataset} is not available. Available dataset are: {available_dataset}"
        )

    if _framework not in mapper[_dataset].keys():
        available_framework = "{" + " | ".join(list(mapper[_dataset].keys()))
        raise ValueError(
            f"framework {_framework} is not available. Available framework are: {available_framework}"
        )

    print(f"loading dataset: {_framework} | {_dataset}")
    return mapper[_dataset][_framework](**kwargs)
