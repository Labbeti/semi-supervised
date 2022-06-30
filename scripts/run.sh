#!/bin/bash

function get_skip_params() {
    if [ "$1" = "usage" ]; then
        echo "Usage: $0 NB_SKIP [PARAMS]"
        exit 0
    fi

    nb_skip_params=$(expr $1 + 1)
    it=0
    script_params=""

    for arg in "$@"
    do
        if [ $it -ge ${nb_skip_params} ]; then
            script_params="${script_params} ${arg}"
        fi
        it=$(expr $it + 1)
    done

    echo "${script_params}"
    return 0
}

function get_param() {
    name="$1"
    default_value="$2"
    pattern="^${name}=.*"

    value="${default_value}"
    found=false

    nb_skip_params=2
    it=0

    for arg in $@
    do
        if [ $it -ge ${nb_skip_params} ]; then
            result=`echo $arg | grep "$pattern"`
            if [ ! -z "$result" ]; then
                value=`echo "$arg" | cut -d "=" -f2`
            fi
        fi
        it=$(expr $it + 1)
    done

    echo "${value}"
    return 0
}

# --- PARAMS
job_name="$1"
fname_script="standalone/$1/$1.py"
script_params=`get_skip_params 1 $@`

tag=`get_param "tag" "NOTAG" $@`

cpus=4
gpus=1
slurm_partition="GPUNodes"
# Time format : days-hours:minutes:seconds. If "0", no time limit. Example for 3 days : 3-00:00:00
slurm_time="0"

dpath_project=`realpath $0 | xargs dirname | xargs dirname`
fpath_python="/users/samova/elabbe/miniconda3/envs/env_leo/bin/python"
fpath_script="${dpath_project}/${fname_script}"

dpath_log="${dpath_project}/logs/slurm"
fpath_out="${dpath_log}/%j_${tag}.out"
fpath_err="${dpath_log}/%j_${tag}.err"
fpath_singularity="/logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif"
srun="srun singularity exec ${fpath_singularity}"

mkdir -p ${dpath_log}

if [ "${slurm_partition}" = "24CPUNodes" ]; then
    mem_per_cpu="7500M"
elif [ "${slurm_partition}" = "48CPUNodes" ]; then
    mem_per_cpu="10000M"
elif [ "${slurm_partition}" = "64CPUNodes" ]; then
    mem_per_cpu="8000M"
elif [ "${slurm_partition}" = "GPUNodes" ]; then
    mem_per_cpu="9000M"
elif [ "${slurm_partition}" = "RTX6000Node" ]; then
    mem_per_cpu="4500M"
else
    echo "Invalid partition ${slurm_partition} for ${path}. (expected 24CPUNodes, 48CPUNodes, 64CPUNodes, GPUNodes or RTX6000Node)"
    exit 1
fi

# Memory format : number[K|M|G|T]. If "0", no memory limit, use all of memory in node. -- DISABLED
mem=""

module_load="module load singularity/3.0.3"
mkdir -p "tmp"
fpath_sbatch="tmp/${job_name}.sbatch"

slurm_params="+slurm.output=${fpath_out} +slurm.error=${fpath_err} +slurm.mem_per_cpu=${mem_per_cpu} +slurm.mem=${mem}"
extended_script_params="${script_params} ${slurm_params}"

# --- BUILD SBATCH FILE

cat << EOT > ${fpath_sbatch}
#!/bin/sh

# Minimal number of nodes (equiv: -N)
#SBATCH --nodes=1

# Number of tasks (equiv: -n)
#SBATCH --ntasks=1

# Job name (equiv: -J)
#SBATCH --job-name=${job_name}

# Log output file
#SBATCH --output=${fpath_out}

# Log err file
#SBATCH --error=${fpath_err}

# Number of CPUs
#SBATCH --cpus-per-task=${cpus}

# Memory limit (0 means no limit) -- DISABLED
## #SBATCH --mem=${mem}

# Memory per cpu
#SBATCH --mem-per-cpu=${mem_per_cpu}

# Duration limit (0 means no limit)
#SBATCH --time=${slurm_time}

# Mail for optional auto-sends -- DISABLED
## #SBATCH --mail-user=""

# Select partition
#SBATCH --partition=${slurm_partition}

# For GPU nodes, select the number of GPUs
#SBATCH --gres=gpu:${gpus}

# For GPU nodes, force job to start only when CPU and GPU are all available
#SBATCH --gres-flags=enforce-binding


# For testing the sbatch file -- DISABLED
## #SBATCH --test-only

# Specify a node list -- DISABLED
## #SBATCH --nodelist=gpu-nc04

# Others -- DISABLED
## #SBATCH --ntasks-per-node=4
## #SBATCH --ntasks-per-core=1
## #SBATCH --mail-type=END

module purge
${module_load}

${srun} ${fpath_python} ${fpath_script} ${extended_script_params}

EOT

# --- RUN
mkdir -p "${dpath_log}/start_logs"
echo "Sbatch job '${job_name}' with tag '${tag}'" | tee -a "${dpath_log}/start_logs/run_osirim_logs.txt"
sbatch ${fpath_sbatch}

exit 0
