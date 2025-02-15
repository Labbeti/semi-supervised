#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import torch

from torch import nn, Tensor
from torch.distributions.beta import Beta


class MixUp(nn.Module):
    """
    Module MixUp that mix batch and labels with a parameter lambda sample from a beta distribution.

    Code overview :

    lambda ~ Beta(alpha, alpha) \n
    lambda = max(lambda, 1 - lambda) \n
    batch = batch_a * lambda + batch_b * (1 - lambda) \n
    label = label_a * lambda + label_b * (1 - lambda) \n

    Note:
        - if alpha -> 0 and apply_max == True, lambda sampled near 1,
        - if alpha -> 1 and apply_max == True, lambda sampled from an uniform distribution in [0.5, 1.0],
        - if alpha -> 0 and apply_max == False, lambda sampled near 1 or 0,
        - if alpha -> 1 and apply_max == False, lambda sampled from an uniform distribution in [0.0, 1.0],
    """

    def __init__(
        self, alpha: float = 0.4, apply_max: bool = False, mix_labels: bool = True
    ):
        """
        Build the MixUp Module.

        :param alpha: Controls the Beta distribution used to sampled the coefficient lambda. (default: 0.4)
        :param apply_max: If True, apply the "lambda = max(lambda, 1 - lambda)" after the sampling of lambda. (default: False)
            This operation is useful for having a mixed batch near to the first batch passed as input.
            It was set to True in MixMatch training but not in original MixUp training.
        :param mix_labels: If True, the labels of the two batches are mix using the same lambda.
            If False, the label of batch_a are kept as is.
        """
        super().__init__()
        self.alpha = alpha
        self.apply_max = apply_max
        self.mix_labels = mix_labels

        self._beta = Beta(alpha, alpha)
        self._lambda = 0.0

    def forward(
        self, batch_a: Tensor, batch_b: Tensor, labels_a: Tensor, labels_b: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply MixUp to batches and labels.
        """
        if batch_a.shape != batch_b.shape or labels_a.shape != labels_b.shape:
            raise RuntimeError(
                "Invalid shapes for MixUp : ({:s} != {:s} or {:s} != {:s})".format(
                    batch_a.shape, batch_b.shape, labels_a.shape, labels_b.shape
                )
            )

        # Sample from Beta distribution
        self._lambda = self._beta.sample().item() if self.alpha > 0.0 else 1.0

        if self.apply_max:
            self._lambda = max(self._lambda, 1.0 - self._lambda)

        labels_mix = labels_a
        if self.mix_labels:
            labels_mix = labels_a * self._lambda + labels_b * (1.0 - self._lambda)

        batch_mix = batch_a * self._lambda + batch_b * (1.0 - self._lambda)

        return batch_mix, labels_mix

    def get_last_lambda(self) -> float:
        """
        Returns the last lambda sampled. If no data has been passed to forward(), returns 0.0.
        """
        return self._lambda


class MixUpBatchShuffle(nn.Module):
    """
    Apply MixUp transform with the same batch in a different order. See MixUp module for details.
    """

    def __init__(
        self, alpha: float = 0.4, apply_max: bool = False, mix_labels: bool = True
    ):
        super().__init__()
        self.mixup = MixUp(alpha, apply_max, mix_labels)

    def forward(self, batch: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        assert batch.shape[0] == labels.shape[0], f"{batch=}, {labels=}"
        batch_size = batch.shape[0]
        indexes = torch.randperm(batch_size)
        batch_shuffle = batch[indexes]
        labels_shuffle = labels[indexes]
        return self.mixup(batch, batch_shuffle, labels, labels_shuffle)

    def get_last_lambda(self) -> float:
        return self.mixup.get_last_lambda()
