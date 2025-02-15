#!/bin/bash

# ___________________________________________________________________________________ #
function show_help {
    echo "usage:  $BASH_SOURCE [-m MODEL] [-r SUPERVISED RATIO] [-e EPOCH] [-R RESUME] [-l LEARNING_RATE] [-a AUGMENT ID] [-h]"
    echo "    -m MODEL (default cnn03)"
    echo "    -r SUPERVISED RATIO (default 0.1)"
    echo "    -e EPOCH (default 200)"
    echo "    -R RESUME (default FALSE)"
    echo "    -l LEARNING_RATE (default 0.003)"
    echo "    -a AUGMENT ID"
    echo "    -h help"

    echo "Osirm parameters"
    echo "    -n NODE (default gpu-nc07)"
    
    echo "Available models"
    echo "	cnn0"
    echo "	cnn03"
    echo "	scallable1"

    echo "Available augmentations"
    echo "see augmentation_list.py"
}

# default parameters
MODEL=cnn03
RATIO=0.1
epochs=200
LEARNING_RATE=0.003
RESUME=0
NODE=" "

while getopts "n:m:r:e:l:a::R::h" arg; do
  case $arg in
    n) NODE=$OPTARG;;
    m) MODEL=$OPTARG;;
    r) RATIO=$OPTARG;;
    e) epochs=$OPTARG;;
    l) LEARNING_RATE=$OPTARG;;
    R) RESUME=1;;
    a) AUGMENT+=("$OPTARG");;
    h) show_help;;
    *) 
        echo "invalide option" 1>&2
        show_help
        exit 1
        ;;
  esac
done

# Check augmentation
if [ "${#AUGMENT[@]}" = "1" ]; then
       AUGMENT_1=${AUGMENT[0]}       
       AUGMENT_2=${AUGMENT[0]}
elif [ "${#AUGMENT[@]}" = "2" ]; then
       AUGMENT_1=${AUGMENT[0]}       
       AUGMENT_2=${AUGMENT[1]}
else
	echo "Please provide at one or two augmentations for the adversarial generation"
	show_help
	exit 2
fi

if [ "${NODE}" = " " ]; then
   NODELINE=""
else
    NODELINE="#SBATCH --nodelist=${NODE}"
fi

folds="-t 1 2 3 4 5 6 7 8 9 -v 10"

# ___________________________________________________________________________________ #
LOG_DIR="logs"
SBATCH_JOB_NAME=a4a_${MODEL}_${AUGMENT_1}_${AUGMENT_2}

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LOG_DIR}/${SBATCH_JOB_NAME}.out
#SBATCH --error=${LOG_DIR}/${SBATCH_JOB_NAME}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
$NODELINE
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding


# sbatch configuration
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dct/bin/python
script=../co-training/co-training-AugAsAdv.py

tensorboard_path_root="--tensorboard_path ../../tensorboard/ubs8k/deep-co-training_aug4adv/${MODEL}/${RATIO}S"
checkpoint_path_root="--checkpoint_path ../../model_save/ubs8k/deep-co-training_aug4adv"

# ___________________________________________________________________________________ #
parameters=""

# -------- tensorboard and checkpoint path --------
tensorboard_path="\${tensorboard_path_root}/${MODEL}/${RATIO}S"
checkpoint_path=\${checkpoint_path_root}/${MODEL}/${RATIO}
parameters="${parameters} ${tensorboard_path} ${checkpoint_path}"

# -------- model --------
parameters="\${parameters} --model ${MODEL}"

# -------- training parameters --------
parameters="\${parameters} --supervised_ratio ${RATIO}"
parameters="\${parameters} --epochs ${epochs}"
parameters="\${parameters} --learning_rate ${LEARNING_RATE}"

# -------- augmentations --------
parameters="\${parameters} --augment_m1 ${AUGMENT_1}"
parameters="\${parameters} --augment_m2 ${AUGMENT_2}"

# -------- resume training --------
if [ $RESUME -eq 1 ]; then
    echo "$RESUME"
    parameters="\${parameters} --resume"
fi

echo python co-training-AugAsAdv.py ${folds} \${parameters}
srun -n 1 -N 1 singularity exec \${container} \${python} \${script} ${folds} \${parameters}

EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh

