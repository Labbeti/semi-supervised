#!/bin/bash

rd="${RANDOM}"

# RUN MT on GSC with augments
# common="--config-name gsc path.dataset_root=/projets/samova/leocances"
# ./run.sh mean-teacher ${common} tag="stu_aug_ident_tea_aug_ident"
# ./run.sh mean-teacher ${common} aug@stu_aug=weak tag="stu_aug_weak_tea_aug_ident"
# ./run.sh mean-teacher ${common} aug@tea_aug=weak tag="stu_aug_ident_tea_aug_weak"
# ./run.sh mean-teacher ${common} aug@stu_aug=weak aug@tea_aug=weak tag="stu_aug_weak_tea_aug_weak"

# RUN MT & DCT on GSC with augments
data="gsc"
dataset_root="/projets/samova/leocances"
tensorboard_root="/users/samova/elabbe/root_sslh/semi-supervised/tensorboard"

common="--config-name ${data} path.dataset_root=${dataset_root} path.tensorboard_root=${tensorboard_root}"

method="mean-teacher"
for stu_aug in "ident" "weak"; do
    for tea_aug in "ident" "weak"; do
        ./run.sh ${method} ${common} aug@stu_aug=${stu_aug} aug@tea_aug=${tea_aug} tag="${rd}_data_${data}__method_${method}__stu_aug_${stu_aug}__tea_aug_${tea_aug}"
    done
done

method="deep-co-training"
for aug_s in "ident" "weak"; do
    for aug_u in "ident" "weak"; do
        ./run.sh ${method} ${common} aug@aug_s=${aug_s} aug@aug_u=${aug_u} tag="${rd}_data_${data}__method_${method}__aug_s_${aug_s}__aug_u_${aug_u}"
    done
done
