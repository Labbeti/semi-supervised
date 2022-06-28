#!/bin/bash

common="--config-name speechcommand path.dataset_root=../../../../../projets/samova/leocances"
./run.sh ${common} tag="_stu_aug_ident_tea_aug_ident"
./run.sh ${common} aug@stu_aug=weak tag="_stu_aug_weak_tea_aug_ident"
./run.sh ${common} aug@tea_aug=weak tag="_stu_aug_ident_tea_aug_weak"
./run.sh ${common} aug@stu_aug=weak aug@tea_aug=weak tag="_stu_aug_weak_tea_aug_weak"
