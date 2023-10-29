# base image
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# local and envs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ARG DEBIAN_FRONTEND=noninteractive

# add some packages
RUN apt-get update
RUN apt-get install -y git h5utils wget vim libc6

# update python pip
RUN python -m pip install --upgrade pip
RUN python --version
RUN python -m pip --version

#RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

# Install PyG.
RUN CPATH=/usr/local/cuda/include:$CPATH \
 && LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
 && DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN pip install scipy

RUN pip install pyg_lib -f https://data.pyg.org/whl/torch-1.13.0+cu116.html \
 && pip install --no-index torch_scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html \
 && pip install --no-index torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html \
 && pip install --no-index torch_cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.html \
 && pip install --no-index torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html \
 && pip install torch-geometric

# copy and install package
COPY . .
RUN python -m pip install -e .
