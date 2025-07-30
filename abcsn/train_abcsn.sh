#!/bin/bash -l

#SBATCH --job-name=abcsn
#SBATCH --partition=idle
#SBATCH --time=7-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

#SBATCH --gpus=1
#SBATCH --constraint=nvidia-gpu

#SBATCH --requeue
#SBATCH --export=ALL

UD_QUIET_JOB_SETUP=YES

export PYTHONHASHSEED=0

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM RESTART COUNT: $SLURM_RESTART_COUNT"
echo "PYTHONHASHSEED: $PYTHONHASHSEED"

python ../abcsn_training.py --model_name="ABCSN" --mask_frac="0.15" --num_epochs_pretrain="10_000" --num_epochs_transfer="10_000" --batch_size_pretrain="64" --batch_size_transfer="64" --lr0_pretrain="1e-4" --lr0_transfer="1e-5" --patience_es_pretrain="25" --patience_es_transfer="25" --patience_rlrp_pretrain="10" --patience_rlrp_transfer="10" --factor_rlrp_pretrain="0.5" --factor_rlrp_transfer="0.5" --minlr_rlrp_pretrain="1e-7" --minlr_rlrp_transfer="1e-7" --mindelta_pretrain="0.0005" --mindelta_transfer="0.005" --PE="fourier" --intermediate_dim="64" --num_heads="8" --do_enc="0.50" --act_ff="leaky_relu" --do_ff="0.50" --l2="0.01" --l1="0"



