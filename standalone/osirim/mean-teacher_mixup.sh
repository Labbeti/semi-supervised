source bash_scripts/parse_option.sh
source bash_scripts/cross_validation.sh

function show_help {
    echo "usage:  $BASH_SOURCE dataset model [-r | --ratio] [training options]"
    echo ""
    echo "Miscalleous arguments"
    echo "    -C | --crossval   CROSSVAL (default FALSE)"
    echo "    -R | --resume     RESUME (default FALSE)"
    echo "    -T | --tensorboard_path T_PATH (default=mean-teacher_mixup)"
    echo "    -h help"
    echo ""
    echo "Training parameters"
    echo "    --dataset            DATASET (default ubs8k)"
    echo "    --model              MODEL (default wideresnet28_4)"
    echo "    --supervised_ratio   SUPERVISED RATIO (default 1.0)"
    echo "    --batch_size         BATCH_SIZE (default 64)"
    echo "    --epochs              EPOCH (default 200)"
    echo "    --learning_rate      LR (default 0.001)"
    echo "    --seed               SEED (default 1234)"
    echo ""
    echo "    --ema_alpha          ALPHA value for the exponential moving average"
    echo "    --warmup_length      WL The length of the warmup"
    echo "    --lambda_cost_max    LCM The consistency cost maximum value"
    echo "    --ccost_method       CC_METHOD Uses JS or MSE for consistency cost"
    echo "    --ccost_softmax      FLAG"
    echo ""
    echo "    --mixup              F_MIXUP"
    echo "    --mixup_alpha        M_ALPHA"
    echo "    --mixup_max      "
    echo "    --mixup_label    "
    echo ""
    echo "Osirim related parameters"
    echo "    -n | --node NODE              On which node the job will be executed"
    echo "    -N | --nb_task NB TASK        On many parallel task"
    echo "    -g | --nb_gpu  NB GPU         On how many gpu this training should be done"
    echo "    -p | --partition PARTITION    On which partition the script will be executed"
    echo ""
    echo "Available partition"
    echo "    GPUNodes"
    echo "    RTX6000Node"
}

# osirim parameters
NODE=" "
NB_TASK=1
NB_GPU=1
PARTITION="GPUNodes"

# training parameters
DATASET="ubs8k"
MODEL=wideresnet28_2
SUPERVISED_RATIO=0.1
BATCH_SIZE=64
epochs=200
LR=0.001
SEED=1234

# Mean teacher parameters
EMA_ALPHA=0.999
WL=50
LCM=1
CC_METHOD=js

# mixup parameters
M_ALPHA=""

# Flag and miscallenous
RESUME=0
CROSSVAL=0
FLAG=""
F_MIXUP=""
F_MAX=""
F_LABEL=""
EXTRA_NAME=""

# Parse the optional parameters
while :; do
    # If no more option (o no option at all)
    if ! [ "$1" ]; then break; fi

    case $1 in
        -R | --resume)        FLAG="${FLAG} --resume"; shift;;
        -C | --crossval)      CROSSVAL=1; shift;;

        --dataset)            DATASET=$(parse_long $2); shift; shift;;
        --model)              MODEL=$(parse_long $2); shift; shift;;
        --supervised_ratio)   SUPERVISED_RATIO=$(parse_long $2); shift; shift;;
        --epochs)           epochs=$(parse_long $2); shift; shift;;
        --learning_rate)      LR=$(parse_long $2); shift; shift;;
        --batch_size)         BATCH_SIZE=$(parse_long $2); shift; shift;;
        --seed)               SEED=$(parse_long $2); shift; shift;;

        --ema_alpha)          EMA_ALPHA=$(parse_long $2); shift; shift;;
        --warmup_length)      WL=$(parse_long $2); shift; shift;;
        --lambda_cost_max)    LCM=$(parse_long $2); shift; shift;;
        --ccost_method)       CC_METHOD=$(parse_long $2); shift; shift;;
        --ccost_softmax)      FLAG="${FLAG} --ccost_softmax"; shift;;
        
        --mixup)              FLAG="${FLAG} --mixup";       EXTRA_NAME="${EXTRA_NAME}_mixup"; shift;;
        --mixup_alpha)        M_ALPHA=$(parse_long $2);     EXTRA_NAME="${EXTRA_NAME}-${M_ALPHA}a": shift; shift;;
        --mixup_max)          FLAG="${FLAG} --mixup_max";   EXTRA_NAME="${EXTRA_NAME}-max"; shift;;
        --mixup_label)        FLAG="${FLAG} --mixup_label"; EXTRA_NAME="${EXTRA_NAME}-label"; shift;;

        -n | --node)      NODE=$(parse_long $2); shift; shift;;
        -N | --nb_task)   NB_TASK=$(parse_long $2); shift; shift;;
        -g | --nb_gpu)    NB_GPU=$(parse_long $2); shift; shift;;
        -p | --partition) PARTITION=$(parse_long $2); shift; shift;;

        -?*) echo "WARN: unknown option" $1 >&2
    esac
done


if [ "${NODE}" = " " ]; then
   NODELINE=""
else
    NODELINE="#SBATCH --nodelist=${NODE}"
fi

# ___________________________________________________________________________________ #
LOG_DIR="logs"
SBATCH_JOB_NAME=mt_${DATASET}_${MODEL}_${SUPERVISED_RATIO}S_${EXTRA_NAME}

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LOG_DIR}/${SBATCH_JOB_NAME}.out
#SBATCH --error=${LOG_DIR}/${SBATCH_JOB_NAME}.err
#SBATCH --ntasks=$NB_TASK
#SBATCH --cpus-per-task=5
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$NB_GPU
#SBATCH --gres-flags=enforce-binding
$NODELINE


# sbatch configuration
# container=/logiciels/containerCollections/CUDA10/pytorch.sif
container=/users/samova/lcances/container/pytorch-dev.sif
python=/users/samova/lcances/.miniconda3/envs/pytorch-dev/bin/python
script=../mean-teacher/mean-teacher_mixup.py

source bash_scripts/add_option.sh

# prepare cross validation parameters
folds_str="$(cross_validation $DATASET $CROSSVAL)"
IFS=";" read -a folds <<< \$folds_str

# -------- dataset & model ------
common_args=\$(append "\$common_args" $DATASET '--dataset')
common_args=\$(append "\$common_args" $MODEL '--model')

# -------- training common_args --------
common_args=\$(append "\$common_args" $SUPERVISED_RATIO '--supervised_ratio')
common_args=\$(append "\$common_args" $epochs '--epochs')
common_args=\$(append "\$common_args" $LR '--learning_rate')
common_args=\$(append "\$common_args" $BATCH_SIZE '--batch_size')
common_args=\$(append "\$common_args" $SEED '--seed')

# -------- mean teacher parameters --------
common_args=\$(append "\$common_args" $EMA_ALPHA '--ema_alpha')
common_args=\$(append "\$common_args" $WL '--warmup_length')
common_args=\$(append "\$common_args" $LCM '--lambda_cost_max')
common_args=\$(append "\$common_args" $CC_METHOD '--ccost_method')

# -------- mixup parameters --------
common_args=\$(append "\$common_args" $M_ALPHA '--mixup_alpha')

# -------- flags --------
# should contain mixup_max, mixup_label, ccost_softmax
common_args="\${common_args} ${FLAG}"

# -------- resume training --------
if [ $RESUME -eq 1 ]; then
    common_args="\${common_args} --resume"
fi

# -------- dataset specific parameters --------
case $DATASET in
    ubs8k | esc10) dataset_args="--num_classes 10";;
    esc50) dataset_args="--num_classes 50";;
    speechcommand) dataset_args="--num_classes 35";;
    ?*) die "dataset ${DATASET} is not available"; exit 1;;
esac


run_number=0
for i in \${!folds[*]}
do
    run_number=\$(( \$run_number + 1 ))

    if [ $CROSSVAL -eq 1 ]; then
        tensorboard_sufix="--tensorboard_sufix run\${run_number}"
    else
        tensorboard_sufix=""
    fi

    extra_params="\${tensorboard_sufix} \${folds[\$i]}"
    
    echo srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
done


EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
# bash .sbatch_tmp.sh
