#!/bin/bash
#SBATCH -A m4439_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -c 32
#SBATCH --gpu-bind=none
#SBATCH --requeue 
#SBATCH -o logs/%x-%j.out
#SBATCH -J test_all
#SBATCH --mail-type=ALL
#SBATCH --signal=SIGUSR1@90
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --module=gpu,nccl-2.18

mkdir -p logs

export SLURM_CPU_BIND="cores"
export CUDA_LAUNCH_BLOCKING=1
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NET_GDR_LEVEL=PHB

# Setup software
module load python
conda activate acorn

srun -u python notebooks/inference/infEx1.py