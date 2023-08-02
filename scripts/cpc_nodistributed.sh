#!/bin/bash
#SBATCH -J cpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --constraint v100-32g

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cpc3
module load sox

export CMD="python $WORK/CPC3/cpc/train.py \
	--pathDB $1 \
	--pathCheckpoint $2 \
	--file_extension .wav \
	--nLevelsGRU 2 \
	--multihead_rnn \
	--schedulerRamp 10 \
	--save_step 1 \
	--n_process_loader 1 \
	--max_size_loaded 300000000 \
	--nEpoch $3 \
	--augment_past \
	--augment_type pitch artificial_reverb \
	--samplingType samespeaker"

echo $CMD
srun $CMD
