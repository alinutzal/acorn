#!/bin/bash
#SBATCH -A m4439_g
##SBATCH -C gpu&hbm80g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -c 32
#SBATCH --gpu-bind=none
#SBATCH -o logs/%x-%j.out
#SBATCH -J gnn
#SBATCH --mail-type=ALL
#SBATCH --signal=SIGUSR1@90
#SBATCH --time=24:00:00
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1

mkdir -p logs

export SLURM_CPU_BIND="cores"
export CUDA_LAUNCH_BLOCKING=1

WANDB__SERVICE_WAIT=300 srun -u g4i-train gnn_train.yaml