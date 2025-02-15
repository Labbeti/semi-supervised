#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import os.path as osp
import time

import hydra
import torch
from torch import nn
from torch.nn import functional as F

from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DataParallel
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary

from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage
from SSL.util.loaders import (
    load_callbacks,
    load_dataset,
    load_optimizer,
    load_preprocesser,
)
from SSL.util.model_loader import load_model
from SSL.util.checkpoint import CheckPoint, CustomSummaryWriter
from SSL.util.utils import reset_seed, get_datetime, track_maximum, get_lr
from SSL.util.utils import get_training_printers, DotDict


@hydra.main(
    config_path=osp.join("..", "..", "config", "supervised"), config_name="ubs8k.yaml"
)
def run(cfg: DictConfig) -> None:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))
    print("current dir: ", os.getcwd())

    reset_seed(cfg.train_param.seed)

    # -------- Get the pre-processer --------
    train_transform, val_transform = load_preprocesser(
        cfg.dataset.dataset,
        "supervised",
        use_augmentation=cfg.train_param.augmentation,
    )

    # -------- Get the dataset --------
    _manager, train_loader, val_loader, test_loader = load_dataset(
        cfg.dataset.dataset,
        "supervised",
        dataset_root=cfg.path.dataset_root,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,
        train_transform=train_transform,
        val_transform=val_transform,
        augmentation=cfg.train_param.augmentation,
        num_workers=cfg.hardware.nb_cpu,  # With the cache enable, it is faster to have run 0 worker
        pin_memory=True,
        verbose=1,
    )
    if test_loader is not None:
        raise NotImplementedError(f"TEST IS NOT IMPLEMTENTED FOR supervised.py script.")

    assert isinstance(train_loader, DataLoader)

    # The input shape of the data is used to generate the model
    input_shape = tuple(train_loader.dataset[0][0].shape)

    # -------- Prepare the model --------
    torch.cuda.empty_cache()  # type: ignore
    model_func = load_model(cfg.dataset.dataset, cfg.model.model)  # type: ignore
    model = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)
    model = model.cuda()

    if cfg.hardware.nb_cpu > 1:
        model = DataParallel(model)

    summary(model, input_shape)

    # -------- Tensorboard and checkpoint --------
    # -- Prepare suffix
    sufix_title = ""
    sufix_title += f"_{cfg.train_param.learning_rate}-lr"
    sufix_title += f"_{cfg.train_param.supervised_ratio}-sr"
    sufix_title += f"_{cfg.train_param.batch_size}-bs"
    sufix_title += f"_{cfg.train_param.seed}-seed"
    sufix_title += f"_{cfg.train_param.augmentation}-aug"

    # -------- Tensorboard logging --------
    tensorboard_sufix = sufix_title + f"_{cfg.train_param.epochs}-e"
    tensorboard_sufix += f"__{cfg.path.sufix}"
    tensorboard_title = f"{get_datetime()}_{cfg.model.model}_{tensorboard_sufix}"
    log_dir = f"{cfg.path.tensorboard_path}/{cfg.model.model}/{tensorboard_title}"
    print("Tensorboard log at: ", log_dir)

    tensorboard = CustomSummaryWriter(log_dir=log_dir, comment=model_func.__name__)

    # -------- Optimizer, callbacks, loss and checkpoint --------
    optimizer = load_optimizer(
        cfg.dataset.dataset,
        "supervised",
        learning_rate=cfg.train_param.learning_rate,
        model=model,
    )
    callbacks = load_callbacks(
        cfg.dataset.dataset,
        "supervised",
        optimizer=optimizer,
        epochs=cfg.train_param.epochs,
    )
    loss_ce = nn.CrossEntropyLoss(reduction="mean")

    checkpoint_sufix = sufix_title + f"__{cfg.path.sufix}"
    checkpoint_title = f"{cfg.model.model}_{checkpoint_sufix}"
    # checkpoint_path = f"{cfg.path.checkpoint_path}/{cfg.model.model}/{checkpoint_title}"
    checkpoint_path = osp.join(tensorboard.log_dir, checkpoint_title)
    checkpoint = CheckPoint(model, optimizer, mode="max", name=checkpoint_path)

    # -------- Metrics and print formater --------
    metrics = DotDict({"fscore": FScore(), "acc": CategoricalAccuracy(),})
    avg = ContinueAverage()

    def reset_metrics() -> None:
        for metric in [metrics.fscore, metrics.acc, avg]:
            metric.reset()

    maximum_tracker = track_maximum()

    header, train_formater, val_formater, test_formater = get_training_printers(
        {"ce": loss_ce}, metrics
    )

    # -------- Training and Validation function --------
    def train(epoch: int) -> None:
        start_time = time.time()
        nb_batch = len(train_loader)
        print("")

        reset_metrics()
        model.train()

        for i, (X, y) in enumerate(train_loader):
            X = X.cuda().float()
            y = y.cuda().long()

            logits = model(X)
            loss = loss_ce(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = torch.softmax(logits, dim=1)
                pred_arg = torch.argmax(logits, dim=1)
                y_one_hot = F.one_hot(y, num_classes=cfg.dataset.num_classes)

                acc = metrics.acc(pred_arg, y).mean(size=None)
                fscore = metrics.fscore(pred, y_one_hot).mean(size=None)
                avg_ce = avg(loss.item()).mean(size=None)

                # logs
                print(
                    train_formater.format(
                        epoch + 1,
                        i,
                        nb_batch,
                        avg_ce,
                        acc,
                        fscore,
                        time.time() - start_time,
                    ),
                    end="\r",
                )

        tensorboard.add_scalar("train/Lce", avg_ce, epoch)
        tensorboard.add_scalar("train/f1", fscore, epoch)
        tensorboard.add_scalar("train/acc", acc, epoch)

    def val(epoch: int) -> float:
        start_time = time.time()
        nb_batch = len(val_loader)
        print("")

        reset_metrics()
        model.eval()

        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                X = X.cuda().float()
                y = y.cuda().long()

                logits = model(X)
                loss = loss_ce(logits, y)

                # metrics
                pred = torch.softmax(logits, dim=1)
                pred_arg = torch.argmax(logits, dim=1)
                y_one_hot = F.one_hot(y, num_classes=cfg.dataset.num_classes)

                acc = metrics.acc(pred_arg, y).mean(size=None)
                fscore = metrics.fscore(pred, y_one_hot).mean(size=None)
                avg_ce = avg(loss.item()).mean(size=None)

                # logs
                print(
                    val_formater.format(
                        epoch + 1,
                        i,
                        nb_batch,
                        avg_ce,
                        acc,
                        fscore,
                        time.time() - start_time,
                    ),
                    end="\r",
                )

        tensorboard.add_scalar("val/Lce", avg_ce, epoch)
        tensorboard.add_scalar("val/f1", fscore, epoch)
        tensorboard.add_scalar("val/acc", acc, epoch)

        tensorboard.add_scalar(
            "hyperparameters/learning_rate", get_lr(optimizer), epoch
        )

        tensorboard.add_scalar("max/acc", maximum_tracker("acc", acc), epoch)
        tensorboard.add_scalar("max/f1", maximum_tracker("f1", fscore), epoch)

        return acc

    # -------- Training loop --------
    print(header)

    if cfg.train_param.resume:
        checkpoint.load_last()

    start_epoch = checkpoint.epoch_counter
    end_epoch = cfg.train_param.epochs

    for e in range(start_epoch, end_epoch):
        train(e)
        acc = val(e)

        # Callbacks and checkpoint
        for c in callbacks:
            c.step()

        checkpoint.step(acc, e)

        tensorboard.flush()

    # -------- Save the hyper parameters and the metrics --------
    hparams = dict(
        dataset=cfg.dataset.dataset,
        model=cfg.model.model,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        epochs=cfg.train_param.epochs,
        learning_rate=cfg.train_param.learning_rate,
        seed=cfg.train_param.seed,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,
    )

    # convert all value to str
    hparams = dict(zip(hparams.keys(), map(str, hparams.values())))

    final_metrics = {
        "max_acc": maximum_tracker.max["acc"],
        "max_f1": maximum_tracker.max["f1"],
    }

    tensorboard.add_hparams(hparams, final_metrics)

    tensorboard.flush()
    tensorboard.close()


if __name__ == "__main__":
    run()
