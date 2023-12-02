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
#SBATCH --time=1:00:00
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1

mkdir -p logs

export SLURM_CPU_BIND="cores"
export CUDA_LAUNCH_BLOCKING=1

WANDB__SERVICE_WAIT=300 srun podman-hpc run -it --rm --gpu -v $PWD:$PWD -v $SCRATCH:/scratch -v /global/cfs/cdirs/m4439:/m4439 -w $PWD docker.io/docexoty/exatrkx:cuda12-pytorch2.1 g4i-train example/Example_1/gnn_train.yaml