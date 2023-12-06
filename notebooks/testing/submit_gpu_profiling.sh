#!/bin/sh

#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=pls0144
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task 20
#SBATCH -J traingnn
#SBATCH -o gnn-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ialadutska@student.ysu.edu

module load miniconda3
module load cuda
source activate umap3

srun --gpu_cmode=exclusive nsys profile -o profile9 -t cuda,cublas,nvtx python profiling_mini.py