#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import time

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from omegaconf import DictConfig, OmegaConf
from torchsummary import summary

from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage
from SSL.loss.losses import JensenShanon
from SSL.util.loaders import (
    load_dataset,
    load_optimizer,
    load_preprocesser,
)
from SSL.util.model_loader import load_model
from SSL.util.checkpoint import mSummaryWriter
from SSL.util.utils import (
    reset_seed,
    get_datetime,
    DotDict,
    track_maximum,
    get_lr,
    get_training_printers,
)


@hydra.main(config_path="../../config/mean-teacher/", config_name="gsc.yaml")
def run(cfg: DictConfig) -> None:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))
    print("current dir: ", os.getcwd())

    reset_seed(cfg.train_param.seed)
    method_name = "mean-teacher"

    student_transform, val_transform = load_preprocesser(
        cfg.dataset.dataset,
        method_name,
        aug_cfg=cfg.stu_aug,
    )
    teacher_transform, _ = load_preprocesser(
        cfg.dataset.dataset,
        method_name,
        aug_cfg=cfg.tea_aug,
    )
    has_same_trans = cfg.stu_aug == cfg.tea_aug

    _manager, _train_loader, _val_loader, test_loader = load_dataset(
        cfg.dataset.dataset,
        method_name,
        dataset_root=cfg.path.dataset_root,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,
        student_transform=student_transform,
        teacher_transform=teacher_transform,
        has_same_trans=has_same_trans,
        val_transform=val_transform,
        num_workers=cfg.hardware.nb_cpu,
        pin_memory=True,
        verbose=1,
    )
    print(f"Test dataset: {test_loader.dataset}")

    input_shape = tuple(test_loader.dataset[0][0].shape)

    # -------- Prepare the model --------
    torch.cuda.empty_cache()  # type: ignore

    model_func = load_model(cfg.dataset.dataset, cfg.model.model)

    student = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)
    teacher = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.resume is None or not os.path.isfile(cfg.resume):
        raise ValueError(f"Invalid argument path {cfg.resume=}.")

    data = torch.load(cfg.resume, map_location=device)

    student_params, teacher_params = data["state_dict"]
    student.load_state_dict(student_params)
    teacher.load_state_dict(teacher_params)

    print(f"Best val score: {data['best_metric'].item()}")

    student = student.to(device=device)
    teacher = teacher.to(device=device)

    if cfg.hardware.nb_gpu > 1:
        student = nn.parallel.DataParallel(student)
        teacher = nn.parallel.DataParallel(teacher)

    summary(student, input_shape)

    # -------- Tensorboard and checkpoint --------
    # -- Prepare suffix
    sufix_title = ""
    sufix_title += f"_{cfg.train_param.learning_rate}-lr"
    sufix_title += f"_{cfg.train_param.supervised_ratio}-sr"
    sufix_title += f"_{cfg.train_param.nb_epoch}-e"
    sufix_title += f"_{cfg.train_param.batch_size}-bs"
    sufix_title += f"_{cfg.train_param.seed}-seed"

    # mean teacher parameters
    sufix_title += f"_{cfg.mt.alpha}a"
    sufix_title += f"-{cfg.mt.warmup_length}wl"
    sufix_title += f"-{cfg.mt.lambda_ccost_max}lcm"
    if cfg.mt.use_softmax:
        sufix_title += "-SOFTMAX"
    sufix_title += f"-{cfg.mt.ccost_method}"

    # mixup parameters
    if cfg.mixup.use:
        sufix_title += "_mixup"
        if cfg.mixup.max:
            sufix_title += "-max"
        if cfg.mixup.label:
            sufix_title += "-label"
        sufix_title += f"-{cfg.mixup.alpha}-a"

    sufix_title += cfg.tag

    # -------- Tensorboard logging --------
    tensorboard_title = f"{get_datetime()}_{cfg.model.model}_{sufix_title}"
    log_dir = f"{cfg.path.tensorboard_path}/{tensorboard_title}"
    print("Tensorboard log at: ", log_dir)

    tensorboard = mSummaryWriter(log_dir=log_dir, comment=model_func.__name__)

    # -------- Optimizer, callbacks, loss and checkpoint --------
    optimizer = load_optimizer(
        cfg.dataset.dataset,
        "mean-teacher",
        student=student,
        learning_rate=cfg.train_param.learning_rate,
    )
    callbacks = []
    loss_ce = nn.CrossEntropyLoss(reduction="mean")  # Supervised loss

    # Unsupervised loss
    if cfg.mt.ccost_method == "mse":
        consistency_cost = nn.MSELoss(reduction="mean")
    elif cfg.mt.ccost_method == "js":
        consistency_cost = JensenShanon
    else:
        raise ValueError(
            f'ccost methods can either be "mse" (Mean Square Error) or "js" (Jensen Shanon). ccost_method={cfg.mt.ccost_method}'
        )

    # -------- Metrics and print formater --------
    metrics = DotDict(
        {
            "sup": DotDict(
                {
                    "acc_s": CategoricalAccuracy(),
                    "acc_t": CategoricalAccuracy(),
                    "fscore_s": FScore(),
                    "fscore_t": FScore(),
                }
            ),
            "unsup": DotDict(
                {
                    "acc_s": CategoricalAccuracy(),
                    "acc_t": CategoricalAccuracy(),
                    "fscore_s": FScore(),
                    "fscore_t": FScore(),
                }
            ),
            "avg": DotDict(
                {
                    "sce": ContinueAverage(),
                    "tce": ContinueAverage(),
                    "ccost": ContinueAverage(),
                }
            ),
        }
    )

    def reset_metrics(metrics: dict):
        for k, v in metrics.items():
            if isinstance(v, dict):
                reset_metrics(v)

            else:
                v.reset()

    maximum_tracker = track_maximum()

    header, train_formater, val_formater = get_training_printers(
        metrics.avg, metrics.sup
    )

    # -------- Training and Validation function --------
    # use softmax or not
    softmax_fn = nn.Identity()
    if cfg.mt.use_softmax:
        softmax_fn = nn.Softmax(dim=1)

    def test(epoch):
        start_time = time.time()
        print("")
        nb_batch = len(test_loader)
        reset_metrics(metrics)
        student.eval()
        teacher.eval()

        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X = X.to(device=device).float()
                y = y.to(device=device)

                # Predictions
                student_logits = student(X)
                teacher_logits = teacher(X)

                # Calculate supervised loss (only student on S)
                loss = loss_ce(student_logits, y)
                _teacher_loss = loss_ce(teacher_logits, y)  # for metrics only
                ccost = consistency_cost(
                    softmax_fn(student_logits), softmax_fn(teacher_logits)
                )

                # Compute the metrics
                y_onehot = F.one_hot(y, num_classes=cfg.dataset.num_classes)

                acc_s = metrics.sup.acc_s(torch.argmax(student_logits, dim=1), y).mean(
                    size=None
                )
                acc_t = metrics.sup.acc_t(torch.argmax(teacher_logits, dim=1), y).mean(
                    size=None
                )
                fscore_s = metrics.sup.fscore_s(
                    torch.softmax(student_logits, dim=1), y_onehot
                ).mean(size=None)
                fscore_t = metrics.sup.fscore_t(
                    torch.softmax(teacher_logits, dim=1), y_onehot
                ).mean(size=None)

                # Running average of the two losses
                sce_avg = metrics.avg.sce(loss.item()).mean(size=None)
                tce_avg = metrics.avg.tce(_teacher_loss.item()).mean(size=None)
                ccost_avg = metrics.avg.ccost(ccost.item()).mean(size=None)

                # logs
                print(
                    val_formater.format(
                        epoch + 1,
                        i,
                        nb_batch,
                        sce_avg,
                        tce_avg,
                        ccost_avg,
                        acc_s,
                        acc_t,
                        fscore_s,
                        fscore_t,
                        time.time() - start_time,
                    ),
                    end="\r",
                )

        tensorboard.add_scalar("val/student_acc", acc_s, epoch)
        tensorboard.add_scalar("val/student_f1", fscore_s, epoch)
        tensorboard.add_scalar("val/teacher_acc", acc_t, epoch)
        tensorboard.add_scalar("val/teacher_f1", fscore_t, epoch)
        tensorboard.add_scalar("val/student_loss", sce_avg, epoch)
        tensorboard.add_scalar("val/teacher_loss", tce_avg, epoch)
        tensorboard.add_scalar("val/consistency_cost", ccost_avg, epoch)

        tensorboard.add_scalar(
            "hyperparameters/learning_rate", get_lr(optimizer), epoch
        )
        tensorboard.add_scalar(
            "max/student_acc", maximum_tracker("student_acc", acc_s), epoch
        )
        tensorboard.add_scalar(
            "max/teacher_acc", maximum_tracker("teacher_acc", acc_t), epoch
        )
        tensorboard.add_scalar(
            "max/student_f1", maximum_tracker("student_f1", fscore_s), epoch
        )
        tensorboard.add_scalar(
            "max/teacher_f1", maximum_tracker("teacher_f1", fscore_t), epoch
        )

        for c in callbacks:
            c.step()

    # -------- Training loop --------
    print(header)

    start_epoch = 0
    end_epoch = 1

    for e in range(start_epoch, end_epoch):
        test(e)
        tensorboard.flush()
    print()

    test_scores = {
        "Test scores": {
            "student_ce": metrics.avg.sce.mean(),
            "teacher_ce": metrics.avg.tce.mean(),
            "ccost": metrics.avg.ccost.mean(),
            "student_acc": metrics.sup.acc_s.mean().item(),
            "teacher_acc": metrics.sup.acc_t.mean().item(),
        },
    }
    print(yaml.dump(test_scores))

    # -------- Save the hyper parameters and the metrics --------
    hparams = {
        "dataset": cfg.dataset.dataset,
        "model": cfg.model.model,
        "supervised_ratio": cfg.train_param.supervised_ratio,
        "batch_size": cfg.train_param.batch_size,
        "nb_epoch": cfg.train_param.nb_epoch,
        "learning_rate": cfg.train_param.learning_rate,
        "seed": cfg.train_param.seed,
        "ema_alpha": cfg.mt.alpha,
        "warmup_length": cfg.mt.warmup_length,
        "lamda_ccost_max": cfg.mt.lambda_ccost_max,
        "use_softmax": cfg.mt.use_softmax,
        "ccost_method": cfg.mt.ccost_method,
        "mixup": cfg.mixup.use,
        "mixup-alpha": cfg.mixup.alpha,
        "mixup-max": cfg.mixup.max,
        "mixup-label": cfg.mixup.label,
    }

    # convert all value to str
    hparams = dict(zip(hparams.keys(), map(str, hparams.values())))

    final_metrics = {
        "max_acc_student": maximum_tracker.max["student_acc"],
        "max_f1_student": maximum_tracker.max["student_f1"],
        "max_acc_teacher": maximum_tracker.max["teacher_acc"],
        "max_f1_teacher": maximum_tracker.max["teacher_f1"],
    }

    tensorboard.add_hparams(hparams, final_metrics)

    tensorboard.flush()
    tensorboard.close()


if __name__ == "__main__":
    run()
