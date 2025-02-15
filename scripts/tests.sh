#!/bin/bash

rd="${RANDOM}"
rd=`seq -f "%05g" ${rd} ${rd}`

# --------------------------------------
# RUN MT on GSC with augments
# common="--config-name gsc path.dataset_root=/projets/samova/leocances"
# ./run.sh mean-teacher ${common} tag="stu_aug_ident_tea_aug_ident"
# ./run.sh mean-teacher ${common} augpool@stu_aug=weak tag="stu_aug_weak_tea_aug_ident"
# ./run.sh mean-teacher ${common} augpool@tea_aug=weak tag="stu_aug_ident_tea_aug_weak"
# ./run.sh mean-teacher ${common} augpool@stu_aug=weak augpool@tea_aug=weak tag="stu_aug_weak_tea_aug_weak"

# --------------------------------------
# # RUN MT & DCT on GSC with augments
# data="gsc"
# dataset_root="/projets/samova/leocances"
# tensorboard_root="/users/samova/elabbe/root_sslh/semi-supervised/logs"

# common="--config-name ${data} path.dataset_root=${dataset_root} path.tensorboard_root=${tensorboard_root}"

# method="mean-teacher"
# for stu_aug in "ident" "weak"; do
#     for tea_aug in "ident" "weak"; do
#         ./run.sh ${method} ${common} augpool@stu_aug=${stu_aug} augpool@tea_aug=${tea_aug} tag="${rd}_data_${data}__method_${method}__stu_aug_${stu_aug}__tea_aug_${tea_aug}"
#     done
# done

# method="deep-co-training"
# for aug_s in "ident" "weak"; do
#     for aug_u in "ident" "weak"; do
#         ./run.sh ${method} ${common} augpool@aug_s=${aug_s} augpool@aug_u=${aug_u} tag="${rd}_data_${data}__method_${method}__aug_s_${aug_s}__aug_u_${aug_u}"
#     done
# done

# --------------------------------------
data="gsc"
dataset_root="/projets/samova/leocances"
tensorboard_root="/users/samova/elabbe/root_sslh/semi-supervised/logs"

common="--config-name ${data} path.dataset_root=${dataset_root} path.tensorboard_root=${tensorboard_root}"

# # Re-run DCT+mixup+uweak
# method="deep-co-training"
# aug_s="ident"
# aug_u="weak"

# args="augpool@aug_s=${aug_s} augpool@aug_u=${aug_u}"
# ./run.sh ${method} ${common} ${args} tag="${rd}_data_${data}__method_${method}__aug_s_${aug_s}__aug_u_${aug_u}"

# # Run MT+mixup with buffer sync
# method="mean-teacher"
# stu_aug="ident"
# tea_aug="ident"
# buffer_sync="true"

# args="augpool@stu_aug=${stu_aug} augpool@tea_aug=${tea_aug} mt.use_buffer_sync=${buffer_sync}"
# ./run.sh ${method} ${common} ${args} tag="${rd}_data_${data}__method_${method}__stu_aug_${stu_aug}__tea_aug_${tea_aug}__buffer_sync_${buffer_sync}"

# # Run MT with Gaussian noise
# method="mean-teacher"
# stu_aug="ident"
# tea_aug="ident"
# pre_trans="noise"
# use_mixup="false"

# args="augpool@stu_aug=${stu_aug} augpool@tea_aug=${tea_aug} aug@pre_trans=${pre_trans} mixup.use=${use_mixup}"
# ./run.sh ${method} ${common} ${args} tag="${rd}_data_${data}__method_${method}__stu_aug_${stu_aug}__tea_aug_${tea_aug}__pre_trans_${pre_trans}__use_mixup_${use_mixup}"

# # Run MT with SpecAug
# method="mean-teacher"
# stu_aug="ident"
# tea_aug="ident"
# post_trans="spec_aug"
# use_mixup="false"

# args="augpool@stu_aug=${stu_aug} augpool@tea_aug=${tea_aug} aug@post_trans=${post_trans} mixup.use=${use_mixup}"
# ./run.sh ${method} ${common} ${args} tag="${rd}_data_${data}__method_${method}__stu_aug_${stu_aug}__tea_aug_${tea_aug}__post_trans_${post_trans}__use_mixup_${use_mixup}"

# Run MT with Gaussian noise & mixup
method="mean-teacher"
stu_aug="ident"
tea_aug="ident"
pre_trans="noise"
use_mixup="true"

args="augpool@stu_aug=${stu_aug} augpool@tea_aug=${tea_aug} aug@pre_trans=${pre_trans} mixup.use=${use_mixup}"
./run.sh ${method} ${common} ${args} tag="${rd}_data_${data}__method_${method}__stu_aug_${stu_aug}__tea_aug_${tea_aug}__pre_trans_${pre_trans}__use_mixup_${use_mixup}"

# DCT classique pour vérifier
method="deep-co-training"
aug_s="ident"
aug_u="ident"
./run.sh ${method} ${common} augpool@aug_s=${aug_s} augpool@aug_u=${aug_u} tag="${rd}_data_${data}__method_${method}__aug_s_${aug_s}__aug_u_${aug_u}"
