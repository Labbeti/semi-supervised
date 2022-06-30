#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import math
import os.path as osp
import time

from typing import Any, Dict, Tuple

import hydra
import torch
import yaml

from advertorch.attacks import GradientSignAttack
from omegaconf import DictConfig, OmegaConf
from torch import nn, Tensor
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.parallel import DataParallel
from torchsummary import summary

from metric_utils.metrics import ContinueAverage, CategoricalAccuracy, FScore, Ratio
from SSL.loss import loss_cot, loss_diff, loss_sup
from SSL.ramps import Warmup, sigmoid_rampup
from SSL.util.checkpoint import CheckPoint, mSummaryWriter
from SSL.util.loaders import (
    load_callbacks,
    load_dataset,
    load_optimizer,
    load_preprocesser,
)
from SSL.util.mixup import MixUpBatchShuffle
from SSL.util.model_loader import load_model
from SSL.util.utils import (
    get_datetime,
    get_lr,
    get_train_format,
    reset_seed,
    track_maximum,
)


@hydra.main(
    config_path=osp.join("..", "..", "config", "deep-co-training"), config_name="gsc"
)
def run(cfg: DictConfig) -> None:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))
    print("current dir: ", os.getcwd())

    reset_seed(cfg.train_param.seed)

    # -------- Get the pre-processer --------
    train_transform_s, val_transform = load_preprocesser(
        cfg.dataset.dataset, "dct", aug_cfg=cfg.aug_s
    )
    train_transform_u, _ = load_preprocesser(
        cfg.dataset.dataset, "dct", aug_cfg=cfg.aug_u
    )

    # -------- Get the dataset --------
    manager, train_loader, val_loader, test_loader = load_dataset(
        cfg.dataset.dataset,
        "dct",
        dataset_root=cfg.path.dataset_root,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,
        train_transform_s=train_transform_s,
        train_transform_u=train_transform_u,
        val_transform=val_transform,
        num_workers=cfg.hardware.nb_cpu,
        pin_memory=True,
        verbose=1,
        download=cfg.download,
    )

    # The input shape of the data is used to generate the model
    input_shape = train_loader._iterables[0].dataset[0][0].shape

    # -------- Prepare the model --------
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_func = load_model(cfg.dataset.dataset, cfg.model.model)

    common_args = dict(
        manager=manager,
        num_classes=cfg.dataset.num_classes,
        input_shape=list(input_shape),
    )

    m1 = model_func(**common_args)
    m2 = model_func(**common_args)

    if cfg.resume is not None:
        if not os.path.isfile(cfg.resume):
            raise ValueError(f"Invalid argument path {cfg.resume=}.")

        data = torch.load(cfg.resume, map_location=torch.device("cpu"))
        m1_params, m2_params = data["state_dict"]
        m1.load_state_dict(m1_params)
        m2.load_state_dict(m2_params)

    m1 = m1.to(device)
    m2 = m2.to(device)

    if cfg.hardware.nb_gpu > 1:
        m1 = DataParallel(m1)
        m2 = DataParallel(m2)

    summary(m1, input_shape)

    # -------- Tensorboard and checkpoint --------
    # -- Prepare suffix
    sufix_title = ""
    sufix_title += f"_{cfg.train_param.learning_rate}-lr"
    sufix_title += f"_{cfg.train_param.supervised_ratio}-sr"
    sufix_title += f"_{cfg.train_param.nb_epoch}-e"
    sufix_title += f"_{cfg.train_param.batch_size}-bs"
    sufix_title += f"_{cfg.train_param.seed}-seed"

    # deep co training parameters
    sufix_title += f"_{cfg.dct.epsilon}eps"
    sufix_title += f"-{cfg.dct.warmup_length}wl"
    sufix_title += f"-{cfg.dct.lambda_cot_max}lcm"
    sufix_title += f"-{cfg.dct.lambda_diff_max}ldm"

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

    tensorboard = mSummaryWriter(log_dir=log_dir, comment=model_func.__name__)

    # -------- Optimizer, callbacks, loss, adversarial generator and checkpoint --------
    optimizer = load_optimizer(
        cfg.dataset.dataset,
        "dct",
        model1=m1,
        model2=m2,
        learning_rate=cfg.train_param.learning_rate,
    )
    callbacks = load_callbacks(
        cfg.dataset.dataset,
        "dct",
        optimizer=optimizer,
        nb_epoch=cfg.train_param.nb_epoch,
    )

    # Checkpoint
    checkpoint_title = f"{cfg.model.model}_{sufix_title}"
    checkpoint_path = f"{cfg.path.checkpoint_path}/{checkpoint_title}"
    checkpoint = CheckPoint([m1, m2], optimizer, mode="max", name=checkpoint_path)

    # define the warmups & add them to the callbacks (for update)
    lambda_cot = Warmup(cfg.dct.lambda_cot_max, cfg.dct.warmup_length, sigmoid_rampup)
    lambda_diff = Warmup(cfg.dct.lambda_diff_max, cfg.dct.warmup_length, sigmoid_rampup)
    callbacks += [lambda_cot, lambda_diff]

    # adversarial generation
    adv_generator_1 = GradientSignAttack(
        m1,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=cfg.dct.epsilon,
        clip_min=-math.inf,
        clip_max=math.inf,
        targeted=False,
    )

    adv_generator_2 = GradientSignAttack(
        m2,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=cfg.dct.epsilon,
        clip_min=-math.inf,
        clip_max=math.inf,
        targeted=False,
    )

    # -------- Metrics and print formater --------
    metrics_fn: Dict[str, Any] = dict(
        ratio_s=[Ratio(), Ratio()],
        ratio_u=[Ratio(), Ratio()],
        acc_s=[CategoricalAccuracy(), CategoricalAccuracy()],
        acc_u=[CategoricalAccuracy(), CategoricalAccuracy()],
        f1_s=[FScore(), FScore()],
        f1_u=[FScore(), FScore()],
        avg_total=ContinueAverage(),
        avg_sup=ContinueAverage(),
        avg_cot=ContinueAverage(),
        avg_diff=ContinueAverage(),
    )

    def reset_metrics() -> None:
        for item in metrics_fn.values():
            if isinstance(item, list):
                for f in item:
                    f.reset()
            else:
                item.reset()

    maximum_tracker = track_maximum()

    header, train_formater, val_formater = get_train_format("dct")

    # -------- Training and Validation function --------
    mixup_u_fn = MixUpBatchShuffle(
        alpha=cfg.mixup.alpha, apply_max=cfg.mixup.max, mix_labels=cfg.mixup.label
    )

    def train(epoch: int) -> float:
        prefix = "train"
        start_time = time.time()
        print("")

        reset_metrics()
        m1.train()
        m2.train()

        total_loss = torch.as_tensor(-1.0)

        for batch_idx, (S1, S2, U) in enumerate(train_loader):
            x_s1, y_s1 = S1
            x_s2, y_s2 = S2
            x_u, y_u = U

            # Apply mixup if needed, otherwise no mixup.
            if cfg.mixup.use:
                x_u, y_u = mixup_u_fn(x_u, y_u)

            x_s1, x_s2, x_u = x_s1.to(device), x_s2.to(device), x_u.to(device)
            y_s1, y_s2, y_u = y_s1.to(device), y_s2.to(device), y_u.to(device)

            with autocast():
                logits_s1 = m1(x_s1)
                logits_s2 = m2(x_s2)
                logits_u1 = m1(x_u)
                logits_u2 = m2(x_u)

            # pseudo labels of U
            pred_u1 = torch.argmax(logits_u1, 1)
            pred_u2 = torch.argmax(logits_u2, 1)

            # ======== Generate adversarial examples ========
            # fix batchnorm ----
            m1.eval()
            m2.eval()

            # generate adversarial examples ----
            adv_data_s1 = adv_generator_1.perturb(x_s1, y_s1)
            adv_data_u1 = adv_generator_1.perturb(x_u, pred_u1)

            adv_data_s2 = adv_generator_2.perturb(x_s2, y_s2)
            adv_data_u2 = adv_generator_2.perturb(x_u, pred_u2)

            m1.train()
            m2.train()

            # predict adversarial examples ----
            with autocast():
                adv_logits_s1 = m1(adv_data_s2)
                adv_logits_s2 = m2(adv_data_s1)

                adv_logits_u1 = m1(adv_data_u2)
                adv_logits_u2 = m2(adv_data_u1)

            # ======== calculate the differents loss ========
            # zero the parameter gradients ----
            for p in m1.parameters():
                p.grad = None  # zero grad
            for p in m2.parameters():
                p.grad = None

            # losses ----
            with autocast():
                l_sup = loss_sup(logits_s1, logits_s2, y_s1, y_s2)

                l_cot = loss_cot(logits_u1, logits_u2)

                l_diff = loss_diff(
                    logits_s1,
                    logits_s2,
                    adv_logits_s1,
                    adv_logits_s2,
                    logits_u1,
                    logits_u2,
                    adv_logits_u1,
                    adv_logits_u2,
                )

                total_loss = l_sup + lambda_cot() * l_cot + lambda_diff() * l_diff
            total_loss.backward()
            optimizer.step()

            # ======== Calc the metrics ========
            with torch.no_grad():
                # accuracies ----
                pred_s1 = torch.argmax(logits_s1, dim=1)
                pred_s2 = torch.argmax(logits_s2, dim=1)

                acc_s1 = metrics_fn["acc_s"][0](pred_s1, y_s1)
                acc_s2 = metrics_fn["acc_s"][1](pred_s2, y_s2)
                acc_u1 = metrics_fn["acc_u"][0](pred_u1, y_u)
                acc_u2 = metrics_fn["acc_u"][1](pred_u2, y_u)

                # ratios  ----
                adv_pred_s1 = torch.argmax(adv_logits_s1, 1)
                adv_pred_s2 = torch.argmax(adv_logits_s2, 1)
                adv_pred_u1 = torch.argmax(adv_logits_u1, 1)
                adv_pred_u2 = torch.argmax(adv_logits_u2, 1)

                ratio_s1 = metrics_fn["ratio_s"][0](adv_pred_s1, y_s1)
                ratio_s2 = metrics_fn["ratio_s"][1](adv_pred_s2, y_s2)
                ratio_u1 = metrics_fn["ratio_u"][0](adv_pred_u1, y_u)
                ratio_u2 = metrics_fn["ratio_u"][1](adv_pred_u2, y_u)
                # ========

                avg_total = metrics_fn["avg_total"](total_loss.item())
                avg_sup = metrics_fn["avg_sup"](l_sup.item())
                avg_diff = metrics_fn["avg_diff"](l_diff.item())
                avg_cot = metrics_fn["avg_cot"](l_cot.item())

                # logs
                print(
                    train_formater.format(
                        "Training: ",
                        epoch + 1,
                        int(100 * (batch_idx + 1) / len(train_loader)),
                        "",
                        avg_sup.mean(size=None),
                        avg_cot.mean(size=None),
                        avg_diff.mean(size=None),
                        avg_total.mean(size=None),
                        "",
                        acc_s1.mean(size=None),
                        acc_u1.mean(size=None),
                        time.time() - start_time,
                    ),
                    end="\r",
                )

                # TODO : rem
                break

        # Using tensorboard to monitor loss and acc\n",
        tensorboard.add_scalar(f"{prefix}/total_loss", avg_total.mean(size=None), epoch)
        tensorboard.add_scalar(f"{prefix}/Lsup", avg_sup.mean(size=None), epoch)
        tensorboard.add_scalar(f"{prefix}/Lcot", avg_cot.mean(size=None), epoch)
        tensorboard.add_scalar(f"{prefix}/Ldiff", avg_diff.mean(size=None), epoch)
        tensorboard.add_scalar(f"{prefix}/m1_acc", acc_s1.mean(size=None), epoch)
        tensorboard.add_scalar(f"{prefix}/m2_acc", acc_s2.mean(size=None), epoch)

        tensorboard.add_scalar("detail_acc/acc_s1", acc_s1.mean(size=None), epoch)
        tensorboard.add_scalar("detail_acc/acc_s2", acc_s2.mean(size=None), epoch)
        tensorboard.add_scalar("detail_acc/acc_u1", acc_u1.mean(size=None), epoch)
        tensorboard.add_scalar("detail_acc/acc_u2", acc_u2.mean(size=None), epoch)

        tensorboard.add_scalar("detail_ratio/ratio_s1", ratio_s1.mean(size=None), epoch)
        tensorboard.add_scalar("detail_ratio/ratio_s2", ratio_s2.mean(size=None), epoch)
        tensorboard.add_scalar("detail_ratio/ratio_u1", ratio_u1.mean(size=None), epoch)
        tensorboard.add_scalar("detail_ratio/ratio_u2", ratio_u2.mean(size=None), epoch)

        # Return the total loss to check for NaN
        return total_loss.item()

    def val(epoch: int) -> Tuple:
        prefix = "val"
        start_time = time.time()
        print()

        reset_metrics()
        m1.eval()
        m2.eval()

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_loader):
                x = X.to(device)
                y = y.to(device)

                with autocast():
                    logits_1 = m1(x)
                    logits_2 = m2(x)

                    # losses ----
                    l_sup = loss_sup(logits_1, logits_2, y, y)

                # ======== Calc the metrics ========
                # accuracies ----
                pred_1 = torch.argmax(logits_1, dim=1)
                pred_2 = torch.argmax(logits_2, dim=1)

                cont_m1_acc = metrics_fn["acc_s"][0](pred_1, y)
                cont_m2_acc = metrics_fn["acc_s"][1](pred_2, y)
                cont_avg_sup = metrics_fn["avg_sup"](l_sup.item())

                # logs
                print(
                    val_formater.format(
                        "Validation: ",
                        epoch + 1,
                        int(100 * (batch_idx + 1) / len(val_loader)),
                        "",
                        cont_avg_sup.mean(size=None),
                        0.0,
                        0.0,
                        cont_avg_sup.mean(size=None),
                        "",
                        cont_m1_acc.mean(size=None),
                        0.0,
                        time.time() - start_time,
                    ),
                    end="\r",
                )

        m1_acc = cont_m1_acc.mean(size=None)
        m2_acc = cont_m2_acc.mean(size=None)

        tensorboard.add_scalar(f"{prefix}/m1_acc", m1_acc, epoch)
        tensorboard.add_scalar(f"{prefix}/m2_acc", m2_acc, epoch)

        tensorboard.add_scalar(
            f"{prefix}_max/m1_acc", maximum_tracker(f"{prefix}/m1_acc", m1_acc), epoch
        )
        tensorboard.add_scalar(
            f"{prefix}_max/m2_acc", maximum_tracker(f"{prefix}/m2_acc", m2_acc), epoch
        )

        tensorboard.add_scalar("hparams/lambda_cot", lambda_cot(), epoch)
        tensorboard.add_scalar("hparams/lambda_diff", lambda_diff(), epoch)
        tensorboard.add_scalar("hparams/learning_rate", get_lr(optimizer), epoch)

        return m1_acc, m2_acc

    def test(epoch: int) -> None:
        if test_loader is None:
            return None

        prefix = "test"
        start_time = time.time()
        print("")

        reset_metrics()
        m1.eval()
        m2.eval()

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(test_loader):
                x = X.to(device)
                y = y.to(device)

                with autocast():
                    logits_1 = m1(x)
                    logits_2 = m2(x)

                    # losses ----
                    l_sup = loss_sup(logits_1, logits_2, y, y)

                # ======== Calc the metrics ========
                # accuracies ----
                pred_1 = torch.argmax(logits_1, dim=1)
                pred_2 = torch.argmax(logits_2, dim=1)

                cont_m1_acc = metrics_fn["acc_s"][0](pred_1, y)
                cont_m2_acc = metrics_fn["acc_s"][1](pred_2, y)
                cont_avg_sup = metrics_fn["avg_sup"](l_sup.item())

                # logs
                print(
                    val_formater.format(
                        "Testing: ",
                        epoch + 1,
                        int(100 * (batch_idx + 1) / len(train_loader)),
                        "",
                        cont_avg_sup.mean(size=None),
                        0.0,
                        0.0,
                        cont_avg_sup.mean(size=None),
                        "",
                        cont_m1_acc.mean(size=None),
                        0.0,
                        time.time() - start_time,
                    ),
                    end="\r",
                )

        m1_acc = cont_m1_acc.mean(size=None)
        m2_acc = cont_m2_acc.mean(size=None)

        tensorboard.add_scalar(f"{prefix}/m1_acc", m1_acc, epoch)
        tensorboard.add_scalar(f"{prefix}/m2_acc", m2_acc, epoch)

        maximum_tracker(f"{prefix}/m1_acc", m1_acc)
        maximum_tracker(f"{prefix}/m2_acc", m2_acc)

    # -------- Training loop ------
    print(header)

    start_epoch = checkpoint.epoch_counter
    end_epoch = cfg.train_param.nb_epoch

    for e in range(start_epoch, end_epoch):
        train(e)
        m1_acc, m2_acc = val(e)

        # Apply callbacks
        for c in callbacks:
            c.step()
        checkpoint.step(m1_acc)

        tensorboard.flush()
    print()

    if test_loader is not None:
        best_epoch = checkpoint.best_state["epoch"]
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
        "nb_epoch": cfg.train_param.nb_epoch,
        "learning_rate": cfg.train_param.learning_rate,
        "seed": cfg.train_param.seed,
        "epsilon": cfg.dct.epsilon,
        "warmup_length": cfg.dct.warmup_length,
        "lamda_cot_max": cfg.dct.lambda_cot_max,
        "lamda_diff_max": cfg.dct.lambda_diff_max,
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
    metric_names = ("m1_acc", "m2_acc")

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
    print("Scores:")
    print(yaml.dump(final_metrics, sort_keys=False))

    tensorboard.add_hparams(hparams, final_metrics)

    tensorboard.flush()
    tensorboard.close()


if __name__ == "__main__":
    run()
