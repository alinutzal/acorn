#!/bin/bash
#SBATCH -A m4439_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -c 32
#SBATCH --gpu-bind=none
#SBATCH -o logs/%x-%j.out
#SBATCH -J trainml
#SBATCH --mail-type=ALL
#SBATCH --signal=SIGUSR1@90
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --module=gpu,nccl-2.15

mkdir -p logs

export SLURM_CPU_BIND="cores"
export CUDA_LAUNCH_BLOCKING=1
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NET_GDR_LEVEL=PHB

# Setup software
module load python
conda activate gnn4itk

WANDB__SERVICE_WAIT=300 srun -u g4i-train examples/Example_2/gnn_train.yaml
