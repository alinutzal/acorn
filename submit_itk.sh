#!/bin/bash 
#SBATCH -C gpu 
#SBATCH -q shared
#SBATCH -A m4439_g
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 1
#SBATCH --gpu-bind none
#SBATCH --time=01:00:00
##SBATCH --image=nvcr.io/nvdlfwea/pyg/pyg:23.10-py3
#SBATCH --module=gpu
#SBATCH -J itk-era5
#SBATCH -o %x-%j.out

DATADIR=/global/cfs/cdirs/m4439/portal.nersc.gov/project/dasrepo/pharring/sc23_data
LOGDIR=${SCRATCH}/cf/logs
mkdir -p ${LOGDIR}
args="${@}"

export FI_MR_CACHE_MONITOR=userfaultfd
export HDF5_USE_FILE_LOCKING=FALSE

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

export MASTER_ADDR=$(hostname)

# Reversing order of GPUs to match default CPU affinities from Slurm
#export CUDA_VISIBLE_DEVICES=3,2,1,0

set -x
# srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
#     bash -c "
#     source export_DDP_vars.sh
#     ${PROFILE_CMD} python train.py ${args}
#     "

srun -u podman-hpc run --rm --gpu \
        -e XDG_RUNTIME_DIR=/tmp/user \
        -v /tmp:/tmp \
        -v $HOME:$HOME \
        -v $PSCRATCH:/scratch \
        -v /global/cfs/cdirs/m4439:/m4439 \
        -v ${DATADIR}:/data \
        -v ${LOGDIR}:/logs \
        -e HOME -w $HOME \
        --net host \
        docker.io/docexoty/exatrkx:cuda12-pytorch2.1 \
        bash -c "
        source activate gnn4itk; 
        cd $HOME/acorn_new; 
        pip install -e .;
        ${PROFILE_CMD} ${args}
        "        
#source export_DDP_vars.sh;