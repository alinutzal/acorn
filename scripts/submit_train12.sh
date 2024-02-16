#!/bin/bash
#SBATCH -A m4439_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 3
#SBATCH -q regular
#SBATCH -c 32
#SBATCH --gpu-bind=none
#SBATCH --requeue 
#SBATCH -o logs/%x-%j.out
#SBATCH -J train10
#SBATCH --mail-type=ALL
#SBATCH --signal=SIGUSR1@90
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --module=gpu,nccl-2.18

mkdir -p logs

export SLURM_CPU_BIND="cores"
export CUDA_LAUNCH_BLOCKING=1
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NET_GDR_LEVEL=PHB

# Setup software
module load python
conda activate acorn

WANDB__SERVICE_WAIT=300 srun -u acorn train examples/Example_1/gnn_train10.yaml