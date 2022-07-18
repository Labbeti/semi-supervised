#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import os.path as osp
import time

from typing import Any, Dict, Union

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn.parallel import DataParallel
from torchsummary import summary

from metric_utils.metrics import ContinueAverage, CategoricalAccuracy, FScore, Metrics
from SSL.loss.losses import JensenShanon
from SSL.ramps import Warmup, sigmoid_rampup
from SSL.util.checkpoint import CheckPoint, CustomSummaryWriter
from SSL.util.loaders import (
    load_callbacks,
    load_dataset,
    load_optimizer,
    load_preprocesser,
)
from SSL.util.mixup import MixUpBatchShuffle
from SSL.util.model_loader import load_model
from SSL.util.utils import (
    DotDict,
    ZipCycle,
    get_datetime,
    get_lr,
    get_training_printers,
    reset_seed,
    track_maximum,
)


@hydra.main(
    config_path=osp.join("..", "..", "config", "mean-teacher"), config_name="gsc"
)
def run(cfg: DictConfig) -> None:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))
    print("current dir: ", os.getcwd())

    reset_seed(cfg.train_param.seed)

    # -------- Get the pre-processer --------
    student_transform, val_transform = load_preprocesser(
        cfg.dataset.dataset, "mean-teacher", aug_cfg=cfg.stu_aug,
    )
    teacher_transform, _ = load_preprocesser(
        cfg.dataset.dataset, "mean-teacher", aug_cfg=cfg.tea_aug,
    )
    has_same_trans = cfg.stu_aug == cfg.tea_aug

    # -------- Get the dataset --------
    _manager, train_loader, val_loader, test_loader = load_dataset(
        cfg.dataset.dataset,
        "mean-teacher",
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
        download=cfg.download,
    )
    assert isinstance(train_loader, ZipCycle)

    if has_same_trans:
        # The input shape of the data is used to generate the model
        input_shape = tuple(train_loader._iterables[0].dataset[0][0].shape)
    else:
        input_shape = tuple(train_loader._iterables[0].dataset._datasets[0][0][0].shape)

    # -------- Prepare the model --------
    torch.cuda.empty_cache()  # type: ignore
    device = torch.device("cuda" if cfg.hardware.nb_gpu > 0 and torch.cuda.is_available() else "cpu")  # type: ignore

    model_func = load_model(cfg.dataset.dataset, cfg.model.model)

    student = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)
    teacher = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)

    if cfg.resume is not None:
        if not os.path.isfile(cfg.resume):
            raise ValueError(f"Invalid argument path {cfg.resume=}.")

        data = torch.load(cfg.resume, map_location=torch.device("cpu"))
        student_params, teacher_params = data["state_dict"]
        student.load_state_dict(student_params)
        teacher.load_state_dict(teacher_params)

    # We do not need gradient for the teacher model
    for p in teacher.parameters():
        p.detach_()

    student = student.eval().to(device)
    teacher = teacher.eval().to(device)

    if cfg.hardware.nb_gpu > 1:
        student = DataParallel(student)
        teacher = DataParallel(teacher)

    summary(student, input_shape, device=device.type)

    # -------- Tensorboard and checkpoint --------
    # -- Prepare suffix
    sufix_title = ""
    sufix_title += f"_{cfg.train_param.learning_rate}-lr"
    sufix_title += f"_{cfg.train_param.supervised_ratio}-sr"
    sufix_title += f"_{cfg.train_param.epochs}-e"
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

    sufix_title += f"_{cfg.tag}"

    # -------- Tensorboard logging --------
    tensorboard_title = f"{get_datetime()}_{cfg.model.model}_{sufix_title}"
    log_dir = f"{cfg.path.tensorboard_path}/{tensorboard_title}"
    print("Tensorboard log at: ", log_dir)

    tensorboard = CustomSummaryWriter(log_dir=log_dir, comment=model_func.__name__)

    # -------- Optimizer, callbacks, loss and checkpoint --------
    optimizer = load_optimizer(
        cfg.dataset.dataset,
        "mean-teacher",
        student=student,
        learning_rate=cfg.train_param.learning_rate,
    )
    callbacks = load_callbacks(
        cfg.dataset.dataset,
        "mean-teacher",
        optimizer=optimizer,
        epochs=cfg.train_param.epochs,
    )
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

    # Warmups
    lambda_cost = Warmup(cfg.mt.lambda_ccost_max, cfg.mt.warmup_length, sigmoid_rampup)
    callbacks += [lambda_cost]

    checkpoint_title = f"{cfg.model.model}_{sufix_title}"
    # checkpoint_path = f"{cfg.path.checkpoint_path}/{checkpoint_title}"
    checkpoint_path = osp.join(tensorboard.log_dir, checkpoint_title)
    checkpoint = CheckPoint(
        [student, teacher], optimizer, mode="max", name=checkpoint_path,
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

    def reset_metrics(metrics: Dict[Any, Union[Metrics, Dict]]) -> None:
        for k, v in metrics.items():
            if isinstance(v, dict):
                reset_metrics(v)
            else:
                v.reset()

    maximum_tracker = track_maximum()

    header, train_formater, val_formater, test_formater = get_training_printers(
        metrics.avg, metrics.sup  # type: ignore
    )

    # -------- Training and Validation function --------
    # use softmax or not
    if cfg.mt.use_softmax:
        ccost_activation = nn.Softmax(dim=1)
    else:
        ccost_activation = nn.Identity()

    if cfg.mt.use_buffer_sync:
        for (_, buffer), (_, ema_buffer) in zip(
            student.named_buffers(),
            teacher.named_buffers(),
        ):
            ema_buffer.set_(buffer.storage())

    # update the teacher using exponentiel moving average
    def update_teacher_model(
        student_model: nn.Module, teacher_model: nn.Module, alpha: float, epoch: int,
    ) -> None:
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (epoch + 1), alpha)

        for param, ema_param in zip(
            student_model.parameters(), teacher_model.parameters(),
        ):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    # For applying mixup
    mixup_fn = MixUpBatchShuffle(
        alpha=cfg.mixup.alpha, apply_max=cfg.mixup.max, mix_labels=cfg.mixup.label,
    )

    def noise_fn(x: Tensor) -> Tensor:
        noise_db = cfg.mt.noise_db
        return x + torch.rand_like(x) * noise_db + noise_db

    def train(epoch: int) -> None:
        start_time = time.time()
        print()

        nb_batch = len(train_loader)

        reset_metrics(metrics)
        student.train()
        if not cfg.mt.use_buffer_sync:
            teacher.train()

        for i, (batch_s, batch_u) in enumerate(train_loader):
            if has_same_trans:
                x_s, ys = batch_s
                x_u, yu = batch_u
                stu_xs = x_s
                tea_xs = x_s
                stu_xu = x_u
                tea_xu = x_u
            else:
                (stu_xs, ys), (tea_xs, _) = batch_s
                (stu_xu, yu), (tea_xu, _) = batch_u

            # Apply mixup if needed, otherwise no mixup.
            if cfg.mixup.use:
                tea_xs, _ = mixup_fn(tea_xs, ys)
                tea_xu, _ = mixup_fn(tea_xu, yu)

            if cfg.mt.noise_db > 0.0:
                tea_xs = noise_fn(tea_xs)
                tea_xu = noise_fn(tea_xu)

            stu_xs = stu_xs.to(device).float()
            stu_xu = stu_xu.to(device).float()
            tea_xs = tea_xs.to(device).float()
            tea_xu = tea_xu.to(device).float()
            ys = ys.to(device)
            yu = yu.to(device)

            # Predictions
            student_s_logits = student(stu_xs)
            student_u_logits = student(stu_xu)
            teacher_s_logits = teacher(tea_xs)
            teacher_u_logits = teacher(tea_xu)

            # Calculate supervised loss (only student on S)
            loss = loss_ce(student_s_logits, ys)

            # Calculate consistency cost (mse(student(x), teacher(x))) x is S + U
            student_logits = torch.cat((student_s_logits, student_u_logits), dim=0)
            teacher_logits = torch.cat((teacher_s_logits, teacher_u_logits), dim=0)
            ccost = consistency_cost(
                ccost_activation(student_logits), ccost_activation(teacher_logits)
            )

            total_loss = loss + lambda_cost() * ccost

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Teacher prediction (for metrics purpose)
                teacher_loss = loss_ce(teacher_s_logits, ys)

                # Update teacher
                update_teacher_model(
                    student, teacher, cfg.mt.alpha, epoch * nb_batch + i
                )

                # Compute the metrics for the student
                y_s_onehot = F.one_hot(ys, num_classes=cfg.dataset.num_classes)
                y_u_onehot = F.one_hot(yu, num_classes=cfg.dataset.num_classes)

                acc_ss = metrics.sup.acc_s(
                    torch.argmax(student_s_logits, dim=1), ys
                ).mean(size=None)
                acc_su = metrics.unsup.acc_s(
                    torch.argmax(student_u_logits, dim=1), yu
                ).mean(size=None)
                fscore_ss = metrics.sup.fscore_s(
                    torch.softmax(student_s_logits, dim=1), y_s_onehot
                ).mean(size=None)
                fscore_su = metrics.unsup.fscore_s(
                    torch.softmax(student_u_logits, dim=1), y_u_onehot
                ).mean(size=None)

                # Compute the metrics for the teacher
                acc_ts = metrics.sup.acc_t(
                    torch.argmax(teacher_s_logits, dim=1), ys
                ).mean(size=None)
                acc_tu = metrics.unsup.acc_t(
                    torch.argmax(teacher_u_logits, dim=1), yu
                ).mean(size=None)
                fscore_ts = metrics.sup.fscore_t(
                    torch.softmax(teacher_s_logits, dim=1), y_s_onehot
                ).mean(size=None)
                fscore_tu = metrics.unsup.fscore_t(
                    torch.softmax(teacher_u_logits, dim=1), y_u_onehot
                ).mean(size=None)

                # Running average of the two losses
                sce_avg = metrics.avg.sce(loss.item()).mean(size=None)
                tce_avg = metrics.avg.tce(teacher_loss.item()).mean(size=None)
                ccost_avg = metrics.avg.ccost(ccost.item()).mean(size=None)

                # logs
                print(
                    train_formater.format(
                        epoch + 1,
                        i,
                        nb_batch,
                        sce_avg,
                        tce_avg,
                        ccost_avg,
                        acc_ss,
                        acc_ts,
                        fscore_ss,
                        fscore_ts,
                        time.time() - start_time,
                    ),
                    end="\r",
                )

        tensorboard.add_scalar("train/student_acc_s", acc_ss, epoch)
        tensorboard.add_scalar("train/student_acc_u", acc_su, epoch)
        tensorboard.add_scalar("train/student_f1_s", fscore_ss, epoch)
        tensorboard.add_scalar("train/student_f1_u", fscore_su, epoch)

        tensorboard.add_scalar("train/teacher_acc_s", acc_ts, epoch)
        tensorboard.add_scalar("train/teacher_acc_u", acc_tu, epoch)
        tensorboard.add_scalar("train/teacher_f1_s", fscore_ts, epoch)
        tensorboard.add_scalar("train/teacher_f1_u", fscore_tu, epoch)

        tensorboard.add_scalar("train/student_loss", sce_avg, epoch)
        tensorboard.add_scalar("train/teacher_loss", tce_avg, epoch)
        tensorboard.add_scalar("train/consistency_cost", ccost_avg, epoch)

    def val(epoch: int) -> None:
        prefix = "val"
        start_time = time.time()
        print("")
        nb_batch = len(val_loader)
        reset_metrics(metrics)
        student.eval()
        teacher.eval()

        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                X = X.to(device).float()
                y = y.to(device)

                # Predictions
                student_logits = student(X)
                teacher_logits = teacher(X)

                # Calculate supervised loss (only student on S)
                loss = loss_ce(student_logits, y)
                teacher_loss = loss_ce(teacher_logits, y)  # for metrics only
                ccost = consistency_cost(
                    ccost_activation(student_logits),
                    ccost_activation(teacher_logits),
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
                tce_avg = metrics.avg.tce(teacher_loss.item()).mean(size=None)
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

        tensorboard.add_scalar(f"{prefix}/student_acc", acc_s, epoch)
        tensorboard.add_scalar(f"{prefix}/student_f1", fscore_s, epoch)
        tensorboard.add_scalar(f"{prefix}/teacher_acc", acc_t, epoch)
        tensorboard.add_scalar(f"{prefix}/teacher_f1", fscore_t, epoch)
        tensorboard.add_scalar(f"{prefix}/student_loss", sce_avg, epoch)
        tensorboard.add_scalar(f"{prefix}/teacher_loss", tce_avg, epoch)
        tensorboard.add_scalar(f"{prefix}/consistency_cost", ccost_avg, epoch)

        tensorboard.add_scalar("hparams/learning_rate", get_lr(optimizer), epoch)
        tensorboard.add_scalar("hparams/lambda_cost_max", lambda_cost(), epoch)

        tensorboard.add_scalar(
            f"{prefix}_max/student_acc",
            maximum_tracker(f"{prefix}/student_acc", acc_s),
            epoch,
        )
        tensorboard.add_scalar(
            f"{prefix}_max/teacher_acc",
            maximum_tracker(f"{prefix}/teacher_acc", acc_t),
            epoch,
        )
        tensorboard.add_scalar(
            f"{prefix}_max/student_f1",
            maximum_tracker(f"{prefix}/student_f1", fscore_s),
            epoch,
        )
        tensorboard.add_scalar(
            f"{prefix}_max/teacher_f1",
            maximum_tracker(f"{prefix}/teacher_f1", fscore_t),
            epoch,
        )

        checkpoint.step(acc_t)
        for c in callbacks:
            c.step()

    def test(epoch: int) -> None:
        if test_loader is None:
            return None

        prefix = "test"
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
                teacher_loss = loss_ce(teacher_logits, y)  # for metrics only
                ccost = consistency_cost(
                    ccost_activation(student_logits), ccost_activation(teacher_logits)
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
                tce_avg = metrics.avg.tce(teacher_loss.item()).mean(size=None)
                ccost_avg = metrics.avg.ccost(ccost.item()).mean(size=None)

                # logs
                print(
                    test_formater.format(
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

        tensorboard.add_scalar(f"{prefix}/student_acc", acc_s, epoch)
        tensorboard.add_scalar(f"{prefix}/student_f1", fscore_s, epoch)
        tensorboard.add_scalar(f"{prefix}/teacher_acc", acc_t, epoch)
        tensorboard.add_scalar(f"{prefix}/teacher_f1", fscore_t, epoch)
        tensorboard.add_scalar(f"{prefix}/student_loss", sce_avg, epoch)
        tensorboard.add_scalar(f"{prefix}/teacher_loss", tce_avg, epoch)
        tensorboard.add_scalar(f"{prefix}/consistency_cost", ccost_avg, epoch)

        maximum_tracker(f"{prefix}/student_acc", acc_s)
        maximum_tracker(f"{prefix}/student_f1", fscore_s)
        maximum_tracker(f"{prefix}/teacher_acc", acc_t)
        maximum_tracker(f"{prefix}/teacher_f1", fscore_t)

    # -------- Training loop --------
    print(header)

    if cfg.train_param.resume:
        checkpoint.load_last()

    start_epoch = checkpoint.epoch_counter
    end_epoch = cfg.train_param.epochs

    for e in range(start_epoch, end_epoch):
        train(e)
        val(e)
        tensorboard.flush()
    print()

    if test_loader is not None:
        best_epoch = checkpoint.best_state["epoch"]
        if best_epoch is None:
            best_epoch = -1
        print(f"Loading best model for testing... ({best_epoch=})\n")
        checkpoint.load_best()
        test(best_epoch)
        print()

    # -------- Save the hyper parameters and the metrics --------
    hparams = {
        "dataset": cfg.dataset.dataset,
        "model": cfg.model.model,
        "supervised_ratio": cfg.train_param.supervised_ratio,
        "batch_size": cfg.train_param.batch_size,
        "epochs": cfg.train_param.epochs,
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

    prefixes = ["val"]
    if test_loader is not None:
        prefixes.append("test")
    metric_names = ("student_acc", "teacher_acc", "student_f1", "teacher_f1")

    final_metrics = {}
    for prefix in prefixes:
        for metric_name in metric_names:
            final_metrics[f"{prefix}_max/{metric_name}"] = maximum_tracker.max[
                f"{prefix}/{metric_name}"
            ]
    final_metrics = {
        k: v.tolist() if isinstance(v, Tensor) else v for k, v in final_metrics.items()
    }

    print()
    print(f"Tensorboard logdir: {tensorboard.log_dir}.")
    print("Scores:")
    print(yaml.dump(final_metrics, sort_keys=False))

    metrics_fpath = osp.join(tensorboard.log_dir, "metrics.yaml")
    with open(metrics_fpath, "w") as file:
        yaml.dump(final_metrics, file)

    tensorboard.add_hparams(hparams, final_metrics)

    tensorboard.flush()
    tensorboard.close()


if __name__ == "__main__":
    run()
