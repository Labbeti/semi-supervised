#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import torch

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter


class CheckPoint:
    def __init__(
        self,
        model: List[nn.Module],
        optimizer: Optimizer,
        mode: str = "max",
        name: str = "best",
        nb_gpu: int = 1,
        verbose: bool = True,
    ) -> None:
        self.mode = mode

        self.name = name
        self.nb_gpu = nb_gpu
        self.verbose = verbose

        self.model = model
        self.optimizer = optimizer
        self.best_state: Dict[str, Any] = {}
        self.last_state: Dict[str, Any] = {}
        self.best_metric = None
        self.epoch_counter = 0

        # Preparation
        if not isinstance(self.model, list):
            self.model = [self.model]

        self._create_directory()
        self._init_message()
        self._init_state()

    def _create_directory(self) -> None:
        os.makedirs(os.path.dirname(self.name), exist_ok=True)

    def _init_state(self) -> None:
        self.best_state = {"state_dict": None, "optimizer": None, "epoch": None}
        self.last_state = {"state_dict": None, "optimizer": None, "epoch": None}

    def _init_message(self) -> None:
        if self.verbose:
            print("checkpoint initialise at: ", os.path.abspath(self.name))
            print("name: ", os.path.basename(self.name))
            print("mode: ", self.mode)

    def step(self, new_value: Union[None, float, Tensor] = None, iter: Optional[int] = None) -> None:
        # Save last epoch
        self.last_state = self._get_state(new_value, iter)
        torch.save(self.last_state, self.name + ".last")

        # Save best epoch
        if self._check_is_better(new_value):
            if self.verbose:
                print("\n better performance: saving ...")

            self.best_metric = new_value
            self.best_state = self._get_state(new_value)
            torch.save(self.best_state, self.name + ".best")

        self.epoch_counter += 1

    def _get_state(self, new_value: Union[None, float, Tensor] = None, iter: Optional[int] = None) -> Dict[str, Any]:
        state = {
            "state_dict": [m.state_dict() for m in self.model],
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch_counter if iter is None else iter,
        }
        if new_value is not None:
            state["best_metric"] = new_value
        return state

    def save(self) -> None:
        torch.save(self._get_state, self.name + ".last")

    def load(self, path: str, model_only: bool = False) -> None:
        data = torch.load(path)

        if not model_only:
            self._load_helper(data, self.last_state)
            self._load_helper(data, self.best_state)
        else:
            self._load_model_only(data)

    def load_best(self) -> None:
        if not os.path.isfile(self.name + ".best"):
            return

        data = torch.load(self.name + ".best")
        self._load_helper(data, self.best_state)

    def load_last(self) -> None:
        if not os.path.isfile(self.name + ".last"):
            print(f"File {self.name}.last doesn't exist")
            return

        data = torch.load(self.name + ".last")
        self._load_helper(data, self.last_state)
        print("Last save loaded ...")

    def _load_helper(self, state, destination):
        for k, v in state.items():
            destination[k] = v

        self.optimizer.load_state_dict(destination["optimizer"])
        self.epoch_counter = destination["epoch"]
        self.best_metric = destination["best_metric"]

        # Path to fit with previous version of checkpoint
        if not isinstance(destination["state_dict"], list):
            destination["state_dict"] = [destination["state_dict"]]

        # The name of the module change when DataParallel is used
        if self.nb_gpu > 1:
            for i in range(len(self.model)):
                state["state_dict"][i] = self._clean_state_dict(state["state_dict"][i])

        # Load the model(s)
        for i in range(len(self.model)):
            self.model[i].load_state_dict(destination["state_dict"][i])

    def _load_model_only(self, state):
        for i in range(len(self.model)):
            state["state_dict"][i] = self._clean_state_dict(state["state_dict"][i])

        for i in range(len(self.model)):
            self.model[i].load_state_dict(state["state_dict"][i])

    def _check_is_better(self, new_value):
        if self.best_metric is None:
            return True
        else:
            return self.best_metric < new_value

    def _clean_state_dict(self, state_dict) -> OrderedDict:
        new_state_dict = OrderedDict()
        for name, v in state_dict.items():
            if "module." in name[:7]:
                name = name[7:]  # remove `module.`

            new_state_dict[name] = v

        return new_state_dict


class mSummaryWriter(SummaryWriter):
    def __init__(
        self,
        log_dir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
    ):
        super().__init__(
            log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix
        )
        self.history = dict()

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        super().add_scalar(tag, scalar_value, global_step, walltime)

        if tag not in self.history:
            self.history[tag] = [scalar_value]
        else:
            self.history[tag].append(scalar_value)
