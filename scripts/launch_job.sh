#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --constraint v100-16g
#SBATCH --cpus-per-task=10

set -e # fail fully on first line failure
set -x

echo "Running on $(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode
    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cpc3

echo $JOB_CMD
srun $JOB_CMD
